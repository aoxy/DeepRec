#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BLOCK_LOCK_LRU_CACHE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BLOCK_LOCK_LRU_CACHE_H_
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "sparsehash/dense_hash_set_lockless"
#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {

template <class K>
class BatchCache;

struct alignas(64) SizeDataBlock {
  size_t val;  // Padding to fill one cache line (64 bytes)
  char _padding[64 - sizeof(size_t)];
  SizeDataBlock() : val(0) {}
};

// __thread static int thread_idx = -1;
static thread_local int thread_idx = -1;

template <class K>
class BlockLockLRUCache : public BatchCache<K> {
 public:
  BlockLockLRUCache(size_t capacity, size_t way, int num_threads = 8)
      : evic_idx_(0),
        num_threads_(num_threads),
        way_(way),
        thread_count_idx(0),
        global_version_(0),
        is_expanding_(false),
        is_rehash_(false) {
    block_count_ = (capacity + way_ - 1) / way_;
    capacity_ = block_count_ * way_;
    base_capacity_ = capacity_;
    cache_.resize(block_count_);
    for (size_t i = 0; i < block_count_; i++) {
      cache_[i] = new CacheBlock(way_);
    }
    size_ = new SizeDataBlock[num_threads_]();
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_CACHE_RECORD_HITRATE", false,
                                   &is_record_hitrate_));
    BatchCache<K>::num_hit = new int64[num_threads_ * 16]();
    BatchCache<K>::num_miss = new int64[num_threads_ * 16]();
    evicted.max_load_factor(0.8);
    evicted.set_empty_key_and_value(EMPTY_CACHE_KEY, -1);
    evicted.set_counternum(16);
    evicted.set_deleted_key(DELETED_CACHE_KEY);
  }

  ~BlockLockLRUCache() override {
    delete[] size_;
    delete[] BatchCache<K>::num_hit;
    delete[] BatchCache<K>::num_miss;
  }

  size_t get_capacity() override { return capacity_; }

  void set_capacity(size_t new_capacity) override {}

  size_t size() {
    size_t total_size = size_[0].val;
    for (size_t j = 1; j < num_threads_; ++j) {
      total_size += size_[j].val;
    }
    return total_size;
  }

  float hit_rate() override {
    float hit_rate = 0.0;
    size_t total_hit = this->num_hit[0];
    size_t total_miss = this->num_miss[0];
    for (size_t j = 1; j < num_threads_; ++j) {
      total_hit += this->num_hit[j * 16];
      total_miss += this->num_miss[j * 16];
    }
    if (total_hit > 0 || total_miss > 0) {
      hit_rate = total_hit * 100.0 / (total_hit + total_miss);
    }
    return hit_rate;
  }

  size_t get_cached_ids(K* cached_ids, size_t k_size, int64* cached_versions,
                        int64* cached_freqs) override {
    size_t true_size = 0;
    return true_size;
  }

  size_t get_evic_ids(K* evic_ids, size_t k_size) {
    size_t true_size = 0;
    for (auto id : evicted) {
      if (true_size < k_size) {
        evic_ids[true_size] = id;
        ++true_size;
        evicted.erase_lockless(id);
      }
    }
    if (thread_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      thread_idx = (thread_count_idx++) % num_threads_;
    }
    size_[thread_idx].val -= true_size;
    return true_size;
  }

  void update(const K* batch_ids, size_t batch_size, bool use_locking = true) {
    __sync_fetch_and_add(&global_version_, 1);
    bool found;
    bool insert;
    size_t min_j;
    size_t min_version;
    int batch_hit = 0;
    int batch_miss = 0;
    int size_add = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      CacheBlock& curr_block = *cache_[id % block_count_];
      std::vector<CacheItem>& curr_cached = curr_block.cached;
      found = false;
      mutex_lock l(curr_block.mtx_cached);
      for (size_t j = 0; j < curr_cached.size(); ++j) {
        if (id == curr_cached[j].id) {
          // LOG(INFO) << "Found ID = " << id
          //           << ", Version: " << curr_cached[j].value << " --> "
          //           << global_version_;
          curr_cached[j].value = global_version_;
          found = true;
          ++batch_hit;
          break;
        }
      }
      if (!found) {
        batch_miss++;
        if (!evicted.erase_lockless(id)) {
          ++size_add;
          // LOG(INFO) << "Size++ By " << id;
        }
        if (is_expanding_ && curr_cached.size() < new_way_) {
          curr_cached.emplace_back(id, global_version_);
        } else {
          insert = false;
          min_j = 0;
          min_version = std::numeric_limits<size_t>::max();
          for (size_t j = 0; j < curr_cached.size(); ++j) {
            if (BlockLockLRUCache<K>::EMPTY_CACHE_KEY == curr_cached[j].id) {
              // LOG(INFO) << "Insert ID = " << id
              //           << ", Version: " << curr_cached[j].value << " --> "
              //           << global_version_;
              curr_cached[j].id = id;
              curr_cached[j].value = global_version_;
              insert = true;
              break;
            } else if (min_version > curr_cached[j].value) {
              min_version = curr_cached[j].value;
              min_j = j;
            }
          }
          if (!insert) {
            // LOG(INFO) << "Replace ID: " << curr_cached[min_j].id << " --> "
            //           << id << ", Version: " << curr_cached[min_j].value
            //           << " --> " << global_version_;
            evicted.insert_lockless(curr_cached[min_j].id);
            curr_cached[min_j].id = id;
            curr_cached[min_j].value = global_version_;
          }
        }
      }
    }
    if (thread_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      thread_idx = (thread_count_idx++) % num_threads_;
    }
    size_[thread_idx].val += size_add;
    if (is_record_hitrate_) {
      this->num_hit[thread_idx * 16] += batch_hit;
      this->num_miss[thread_idx * 16] += batch_miss;
    }
  }

  void update(const K* batch_ids, size_t batch_size, const int64* batch_version,
              const int64* batch_freqs, bool use_locking = true) override {
    update(batch_ids, batch_size, false);
  }

  void add_to_prefetch_list(const K* batch_ids, const size_t batch_size) {}

  void add_to_cache(const K* batch_ids, const size_t batch_size) {
    update(batch_ids, batch_size, false);
  }

 private:
  class CacheItem {
   public:
    K id;
    size_t value;
    CacheItem(K id, size_t version) : id(id), value(version) {}
    CacheItem() : id(BlockLockLRUCache<K>::EMPTY_CACHE_KEY), value(0) {}
  };
  class CacheBlock {
   public:
    std::vector<CacheItem> cached;
    mutex mtx_cached;
    CacheBlock(size_t way) { cached.resize(way); }
    CacheBlock() = delete;
    CacheBlock(CacheBlock&) = delete;
    CacheBlock& operator=(CacheBlock&) = delete;
  };

  std::vector<CacheBlock*> cache_;
  typedef google::dense_hash_set_lockless<K> LocklessHashSet;
  LocklessHashSet evicted;
  mutex sync_idx_mu_;
  size_t block_count_;
  size_t evic_idx_;
  size_t ext_idx_;
  int num_threads_;
  size_t way_;
  size_t new_way_;
  size_t capacity_;
  size_t base_capacity_;
  bool is_expanding_;
  bool is_rehash_;
  mutex rehash_mu_;
  SizeDataBlock* size_;
  size_t global_version_;
  bool is_record_hitrate_;
  unsigned int thread_count_idx;
  static const K EMPTY_CACHE_KEY;
  static const K DELETED_CACHE_KEY;
};

template <class K>
const K BlockLockLRUCache<K>::EMPTY_CACHE_KEY = -1;

template <class K>
const K BlockLockLRUCache<K>::DELETED_CACHE_KEY = -2;
}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
