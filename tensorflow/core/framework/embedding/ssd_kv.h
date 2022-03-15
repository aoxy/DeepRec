#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_

#include <fstream>
#include <sstream>
#include <vector>

#include "sparsehash/dense_hash_map"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/io/path.h"

namespace tensorflow {

template <class V>
class ValuePtr;

template <class K, class V>
class SSDKV : public KVInterface<K, V> {
 public:
  SSDKV(std::string path) {
    path_ = io::JoinPath(
        path, "ssd_kv_" + std::to_string(Env::Default()->NowMicros()) + "_");
    hash_map.max_load_factor(0.8);
    hash_map.set_empty_key(-1);
    hash_map.set_deleted_key(-2);
    current_version = 0;
    buffer_size = 4 * 1024; // Write 4KB at once.
    write_buffer = new char[buffer_size];
    curr_buffer_off = 0;
    fs.push_back(std::fstream(
        path_ + std::to_string(current_version),
        std::ios::app | std::ios::in | std::ios::out | std::ios::binary));
    CHECK(fs[current_version].good());

    counter_ = new SizeCounter<K>(8);
    app_counter_ = new SizeCounter<K>(8);
    new_value_ptr_fn_ = [](size_t size) {
      return new NormalContiguousValuePtr<V>(size);
    };
  }

  void SetTotalDims(int total_dims) {
    total_dims_ = total_dims;
    val_len = sizeof(FixedLengthHeader) + total_dims_ * sizeof(V);
  }

  ~SSDKV() {
    for (int i = 0; i < fs.size(); i++) {
      fs[i].close();
    }
    delete[] write_buffer;
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) {
    auto iter = hash_map.find(key);
    if (iter == hash_map.end()) {
      return errors::NotFound("Unable to find Key: ", key, " in SSDKV.");
    } else {
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      OffsetVersion posi = iter->second;
      fs[posi.version].seekg(posi.offset, std::ios::beg);
      fs[posi.version].read((char*)(val->GetPtr()), val_len);
      *value_ptr = val;
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) {
    auto iter = hash_map.find(key);
    if (iter == hash_map.end()) {
      fs[current_version].seekp(0, std::ios::end);
      size_t offset = fs[current_version].tellp();
      hash_map[key] = OffsetVersion(offset, current_version);;
      fs[current_version].write((char*)value_ptr->GetPtr(), val_len);
      counter_->add(key, 1);
      app_counter_->add(key, 1);
      return Status::OK();
    } else {
      return errors::AlreadyExists("already exists Key: ", key, " in SSDKV.");
    }
  }

  Status BatchInsert(std::vector<K> keys,
                     std::vector<ValuePtr<V>*> value_ptrs) {
    return BatchCommit(keys, value_ptrs);
  }

  Status BatchCommit(std::vector<K> keys,
                     std::vector<ValuePtr<V>*> value_ptrs) {
    fs[current_version].seekp(0, std::ios::end);// seek to end
    size_t offset = fs[current_version].tellp();// first offset
    for (int i = 0; i < keys.size(); i++) {
      app_counter_->add(keys[i], 1);
      hash_map[keys[i]] = OffsetVersion(offset + val_len * i, current_version);
      if(curr_buffer_off + val_len >= buffer_size){
        fs[current_version].write(write_buffer, curr_buffer_off);
        // LOG(INFO) << "write: " << curr_buffer_off << std::endl;
        curr_buffer_off = 0;
      }
      memcpy(write_buffer + curr_buffer_off, (char*)value_ptrs[i]->GetPtr(), val_len);
      curr_buffer_off += val_len;
    }
    if(curr_buffer_off > 0){
      fs[current_version].write(write_buffer, curr_buffer_off);
      curr_buffer_off = 0;
    }
    return Status::OK();
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) {
    app_counter_->add(key, 1);
    fs[current_version].seekp(0, std::ios::end);
    size_t offset = fs[current_version].tellp();
    hash_map[key] = OffsetVersion(offset, current_version);  // Update offset.
    fs[current_version].write((char*)value_ptr->GetPtr(), val_len);
    delete value_ptr;
    return Status::OK();
  }

  Status Remove(K key) {
    counter_->sub(key, 1);
    if (hash_map.erase(key)) {
      return Status::OK();
    } else {
      return errors::NotFound("Unable to find Key: ", key, " in SSDKV.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>*>* value_ptr_list) {
    spin_rd_lock l(mu);
    for (const auto it : hash_map) {
      key_list->push_back(it.first);
      OffsetVersion posi = it.second;
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      fs[posi.version].seekg(posi.offset, std::ios::beg);
      fs[posi.version].read((char*)(val->GetPtr()), val_len);
      value_ptr_list->push_back(val);
    }
    return Status::OK();
  }

  int64 Size() const { return counter_->size(); }

  void FreeValuePtr(ValuePtr<V>* value_ptr) { delete value_ptr; }

  std::string DebugString() const {
    return strings::StrCat("counter_->size(): ", counter_->size(),
                           "app_counter_->size(): ", app_counter_->size());
  }

 private:
  size_t val_len;
  char *write_buffer;
  size_t buffer_size;
  size_t curr_buffer_off;
  SizeCounter<K>* counter_;
  SizeCounter<K>* app_counter_;
  std::string path_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;
  int total_dims_;

  mutable easy_spinrwlock_t mu = EASY_SPINRWLOCK_INITIALIZER;
  class OffsetVersion {
   public:
    size_t offset;   // 在文件中的偏移
    size_t version;  // 存储在哪个文件中
    OffsetVersion(size_t o, size_t v) : offset(o), version(v) {}
    OffsetVersion() : offset(-1), version(-1) {}
  };
  google::dense_hash_map<K, OffsetVersion> hash_map;
  std::vector<std::fstream> fs;
  size_t current_version;
};

}  // namespace tensorflow

#endif TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_
