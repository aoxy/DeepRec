#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_

#include "tensorflow/core/lib/io/path.h"
#include "sparsehash/dense_hash_map"

#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/framework/embedding/value_ptr.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"

#include <sstream>
#include <vector>
#include <fstream>



namespace tensorflow {
  
template <class V>
class ValuePtr;

template <class K, class V>
class SSDKV : public KVInterface<K, V> {
 public:
  SSDKV(std::string path) {
    path_ = io::JoinPath(path, "ssd_kv_" + std::to_string(Env::Default()->NowMicros()) + "_");
    hash_map_ = new dense_hash_map[partition_num_];
    for (int i = 0; i< partition_num_; i++) {
      hash_map_[i].hash_map.max_load_factor(0.8);
      hash_map_[i].hash_map.set_empty_key(-1);
      hash_map_[i].hash_map.set_deleted_key(-2);
      // hash_map_[i].fs.open(path_ + std::to_string(i), std::ios::app | std::ios::in | std::ios::out | std::ios::binary);
      fs.push_back(std::fstream(path_ + std::to_string(i), std::ios::app | std::ios::in | std::ios::out | std::ios::binary));
      CHECK(fs[i].good());
    }
    counter_ =  new SizeCounter<K>(8);
    app_counter_ =  new SizeCounter<K>(8);
    new_value_ptr_fn_ = [] (size_t size) { return new NormalContiguousValuePtr<V>(size); };
  }

  void SetTotalDims(int total_dims) {
    total_dims_ = total_dims;
    val_len = sizeof(FixedLengthHeader) + total_dims_ * sizeof(V);
  }

  ~SSDKV() {
    LOG(INFO) << "~SSDKV()";
    for (int i = 0; i< partition_num_; i++) {
      fs[i].close();
    }
    delete []hash_map_;
  }

  Status Lookup(K key, ValuePtr<V>** value_ptr) {
    int64 l_id = std::abs(key)%partition_num_;
    spin_rd_lock l(hash_map_[l_id].mu);
    auto iter = hash_map_[l_id].hash_map.find(key);
    if (iter == hash_map_[l_id].hash_map.end()) {
      return errors::NotFound(
          "Unable to find Key: ", key, " in SSDKV.");
    } else {
      ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
      int64 offset = iter->second;
      fs[l_id].seekg(offset, std::ios::beg);
      fs[l_id].read((char*)(val->GetPtr()), val_len);
      *value_ptr = val;
      return Status::OK();
    }
  }

  Status Insert(K key, const ValuePtr<V>* value_ptr) {
    int64 l_id = std::abs(key)%partition_num_;
    spin_wr_lock l(hash_map_[l_id].mu);
    auto iter = hash_map_[l_id].hash_map.find(key);
    if (iter == hash_map_[l_id].hash_map.end()) {
      fs[l_id].seekp(0, std::ios::end);
      int64 offset = fs[l_id].tellp();
      hash_map_[l_id].hash_map[key] = offset;
      fs[l_id].write((char*)value_ptr->GetPtr(), val_len);
      counter_->add(key, 1);
      app_counter_->add(key, 1);
      return Status::OK();
    } else {
      return errors::AlreadyExists(
          "already exists Key: ", key, " in SSDKV.");
    }
  }

  Status BatchInsert(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    return BatchCommit(keys, value_ptrs);
  } 

  Status BatchCommit(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    for (int i = 0; i < keys.size(); i++) {
      Commit(keys[i], value_ptrs[i]);
    }
    return Status::OK();
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) {
    app_counter_->add(key, 1);
    int64 l_id = std::abs(key)%partition_num_;
    spin_wr_lock l(hash_map_[l_id].mu);
    fs[l_id].seekp(0, std::ios::end);
    int64 offset = fs[l_id].tellp();
    hash_map_[l_id].hash_map[key] = offset; // Update offset.
    fs[l_id].write((char*)value_ptr->GetPtr(), val_len);
    delete value_ptr;
    return Status::OK();
  }

  Status Remove(K key) {
    counter_->sub(key, 1);
    int64 l_id = std::abs(key)%partition_num_;
    spin_wr_lock l(hash_map_[l_id].mu);
    if (hash_map_[l_id].hash_map.erase(key)) {
      return Status::OK();
    } else {
      return errors::NotFound(
          "Unable to find Key: ", key, " in SSDKV.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list, std::vector<ValuePtr<V>* >* value_ptr_list) {
    dense_hash_map hash_map_dump[partition_num_];
    // std::vector<std::fstream> fs_dump;
    int64 offset;
    for (int i = 0; i< partition_num_; i++) {
      spin_rd_lock l(hash_map_[i].mu);
      hash_map_dump[i].hash_map = hash_map_[i].hash_map;
      // fs_dump.push_back(fs[i]);
    }
    for (int i = 0; i< partition_num_; i++) {
      for (const auto it : hash_map_dump[i].hash_map) {
        key_list->push_back(it.first);
        offset = it.second;
        ValuePtr<V>* val = new_value_ptr_fn_(total_dims_);
        fs[i].seekg(offset, std::ios::beg);
        fs[i].read((char*)(val->GetPtr()), val_len);
        value_ptr_list->push_back(val);
      }
    }
    return Status::OK();
  }

  int64 Size() const {
    return counter_->size();
  }

  void FreeValuePtr(ValuePtr<V>* value_ptr) {
    delete value_ptr;
  }

  std::string DebugString() const {
    return strings::StrCat("counter_->size(): ", counter_->size(),
                           "app_counter_->size(): ", app_counter_->size());
  }
 private:
  size_t val_len;
  const int partition_num_ = 1;
  SizeCounter<K>* counter_;
  SizeCounter<K>* app_counter_;
  std::string path_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;
  int total_dims_;

  struct dense_hash_map {
    mutable easy_spinrwlock_t mu = EASY_SPINRWLOCK_INITIALIZER;
    google::dense_hash_map<K, size_t> hash_map;
    // std::fstream fs;
  };
  dense_hash_map* hash_map_;
  std::vector<std::fstream> fs;
};

} //namespace tensorflow

#endif  TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_KV_H_
