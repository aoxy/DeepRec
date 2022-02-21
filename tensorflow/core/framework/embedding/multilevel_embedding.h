#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTILEVEL_EMBEDDING_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTILEVEL_EMBEDDING_H_

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/dense_hash_map.h"
#include "tensorflow/core/framework/embedding/leveldb_kv.h"
#include "tensorflow/core/framework/embedding/lockless_hash_map.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
template <class V>
class ValuePtr;

namespace embedding {

struct StorageConfig {
  StorageConfig() : type(StorageType::INVALID), path("") {}
  StorageConfig(StorageType t,
                const std::string& p) : type(t), path(p) {}
  StorageType type;
  std::string path;
};

template <class K, class V>
class StorageManager {
 public:
  StorageManager(const string& name,
                 StorageConfig sc,
                 size_t cap = -1)
: hash_table_count_(0),
  name_(name),
  sc_(sc),
  cache_(nullptr),
  cache_capacity_(cap),
  eviction_thread_(nullptr),
  total_dims_(0) {}

  ~StorageManager() {
    if (eviction_thread_) {
      mutex_lock l(mu_);
      shutdown_cv_.notify_all();
      shutdown_ = true;
    }
    delete eviction_thread_;
    for (auto kv: kvs_) {
      delete kv;
    }
    delete cache_;
  }

  Status Init() {
    new_value_ptr_fn_ = [] (size_t size) { return new NormalContiguousValuePtr<V>(size); };
    switch (sc_.type) {
      case StorageType::DRAM:
      case StorageType::PMEM_LIBPMEM:
        LOG(INFO) << "StorageManager::DRAM: " << name_;
        kvs_.push_back(new LocklessHashMap<K, V>());
        break;
      case StorageType::LEVELDB:
        LOG(INFO) << "StorageManager::LEVELDB: " << name_;
        kvs_.push_back(new LevelDBKV<K, V>(sc_.path));
        break;
      case StorageType::DRAM_LEVELDB:
        LOG(INFO) << "StorageManager::DRAM_LEVELDB: " << name_;
        kvs_.push_back(new LocklessHashMap<K, V>());
        kvs_.push_back(new LevelDBKV<K, V>(sc_.path));
        break;
      default:
        LOG(INFO) << "StorageManager::default";
        kvs_.push_back(new LocklessHashMap<K, V>());
        break;
    }
    hash_table_count_ = kvs_.size();
    if (hash_table_count_ > 1) {
      cache_ = new LRUCache<K>();
      eviction_thread_ = Env::Default()->StartThread(ThreadOptions(), "EV_Eviction",
                                                     [this]() { BatchEviction(); });
      thread_pool_.reset(new thread::ThreadPool(Env::Default(), ThreadOptions(),
                                               "MultiLevel_Embedding_Cache", 2,
                                               /*low_latency_hint=*/false));
    }
    // DebugString();
    CHECK(2 >= hash_table_count_) << "Not support multi-level(>2) embedding.";

    return Status::OK();
  }

  void DebugString() {
    LOG(INFO) << "Level Number: " << hash_table_count_;
    LOG(INFO) << "Storage Type: " << sc_.type;
    LOG(INFO) << "Storage Path: " << sc_.path;
  }

  void Schedule(std::function<void()> fn) {
    if (hash_table_count_ > 1) {
      thread_pool_->Schedule(std::move(fn));
    }
  }

  Status GetOrCreate(K key, ValuePtr<V>** value_ptr, size_t size) {
    bool found = false;
    int level = 0;
    for (; level < hash_table_count_; ++level) {
      Status s = kvs_[level]->Lookup(key, value_ptr);
      if (s.ok()) {
        found = true;
        break;
      }
    }
    if (!found) {
      *value_ptr = new_value_ptr_fn_(size);
    }
    if (level || !found) {
      Status s = kvs_[0]->Insert(key, *value_ptr);
      if (s.ok()) {
        // Insert Success
        //LOG(INFO) << "Insert Success, size: " <<kvs_[0]->Size();
        return s;
      } else {
        // Insert Failed, key already exist
        delete *value_ptr;
        s = kvs_[0]->Lookup(key, value_ptr);
        return s;
      }
    }
    return Status::OK();
  }

  Status Remove(K key) {
    for (auto kv : kvs_) {
      kv->Remove(key);
    }
    return Status::OK();
  }

  int64 Size() const {
    int64 total_size = 0;
    for (auto kv : kvs_) {
      total_size += kv->Size();
    }
    return total_size;
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<ValuePtr<V>* >* value_ptr_list) {
    for (auto kv : kvs_) {
      TF_CHECK_OK(kv->GetSnapshot(key_list, value_ptr_list));
    }
    return Status::OK();
  }

  Status BatchCommit(std::vector<K> keys, std::vector<ValuePtr<V>*> value_ptrs) {
    for (auto kv : kvs_) {
      TF_CHECK_OK(kv->BatchCommit(keys, value_ptrs));
    }
    return Status::OK();
  }

  BatchCache<K>* Cache() {
    return cache_;
  }

  void SetDim(int index, int dim, int slotnum) {
    int i;
    while (flag_.test_and_set(std::memory_order_acquire));
    if (slotnum != slot_dims_.size()) {
      for (i = slot_dims_.size(); i < slotnum; i++) {
        slot_dims_.emplace_back(0);
        slot_offset_.emplace_back(0);
      }
    }
    dim +=  (16 - (sizeof(V) * dim) % 16) / sizeof(V);
    slot_dims_[index] = dim;
    total_dims_ += dim;
    for (i = 0; i < slotnum; i++) {
      if (slot_dims_[i] == 0)
        break;
    }
    if (i == slotnum) {
      for (int j = 1; j < slotnum; j++) {
        slot_offset_[j] += slot_dims_[j-1] + slot_offset_[j-1];
      }
      // set slot_dims/slot_offset done
      mutex_lock l(mu_);
      done_ = true;
      for (auto kv : kvs_) {
        kv->SetTotalDims(total_dims_);
      }
    }
    flag_.clear(std::memory_order_release);
  }

  int GetOffset(int index) {
    if (slot_offset_.size() == 0)
      return 0;
    else
      return slot_offset_[index];
  }

  int GetTotalDims() {
    return total_dims_;
  }

  Status Commit(K key, const ValuePtr<V>* value_ptr) {
    TF_CHECK_OK(kvs_[0]->Commit(key, value_ptr));
    return Status::OK();
  }

  void FreeValuePtr(ValuePtr<V>* value_ptr) {
    for (auto kv : kvs_) {
      kv->FreeValuePtr(value_ptr);
    }
  }

 private:
  void BatchEviction() {
    Env* env = Env::Default();
    const int kSize = 1000;
    if (cache_capacity_ == -1) {
      while (true) {
        mutex_lock l(mu_);
        if (done_) {
          break;
        }
      }
      // default 1GB cache size approximately
      cache_capacity_ = 1024 * 1024 * 1024 / (GetTotalDims() * sizeof(V));
    }
    LOG(INFO) << "Cache cache_capacity: " << cache_capacity_;
    K evic_ids[kSize];
    while (true) {
      mutex_lock l(mu_);
      if (shutdown_) {
        break;
      }
      const int kTimeoutMilliseconds = 10 * 1;
      WaitForMilliseconds(&l, &shutdown_cv_, kTimeoutMilliseconds);

      int cache_count = cache_->size();
      if (cache_count > cache_capacity_) {
        // eviction
        int k_size = cache_count - cache_capacity_;
        k_size = std::min(k_size, kSize);
        //K* evic_ids = new K[k_size];
        size_t true_size = cache_->get_evic_ids(evic_ids, k_size);
        ValuePtr<V>* value_ptr;
        for (int64 i = 0; i < true_size; ++i) {
          if (kvs_[0]->Lookup(evic_ids[i], &value_ptr).ok()) {
            TF_CHECK_OK(kvs_[0]->Remove(evic_ids[i]));
            //TF_CHECK_OK(kvs_[1]->Insert(evic_ids[i], value_ptr));
            TF_CHECK_OK(kvs_[1]->Commit(evic_ids[i], value_ptr));
            // delete value_ptr;
          } else {
            // bypass
          }
        }
        //LOG(INFO) << "kvs_[0] size:"<< kvs_[0]->Size() << ", kvs_[1] size: " << kvs_[1]->Size();
        //delete[] evic_ids;
      }
    }
  }

 private:
  int32 hash_table_count_;
  std::string name_;
  std::vector<KVInterface<K, V>*> kvs_;
  std::function<ValuePtr<V>*(size_t)> new_value_ptr_fn_;
  StorageConfig sc_;

  std::unique_ptr<thread::ThreadPool> thread_pool_;
  Thread* eviction_thread_;
  BatchCache<K>* cache_;
  size_t cache_capacity_;
  mutex mu_;
  condition_variable shutdown_cv_;
  bool shutdown_ GUARDED_BY(mu_) = false;

  std::vector<int> slot_dims_;
  std::vector<int> slot_offset_;
  int total_dims_;
  bool done_ = false;

  std::atomic_flag flag_ = ATOMIC_FLAG_INIT;


};

} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_MULTILEVEL_EMBEDDING_H_
