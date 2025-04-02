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

// __thread static int thread_idx = -1;
static thread_local int thread_idx = -1;

template <class K>
class BlockLockLRUCache : public BatchCache<K> {
 public:
  BlockLockLRUCache(const std::string& name, size_t capacity, size_t way, int num_threads = 8)
      : name_(name),
        base_capacity_(capacity),
        capacity_(capacity),
        num_threads_(num_threads),
        way_(way),
        thread_count_idx(0),
        // global_version_(0),
        is_shrinking_(false),
        is_expanding_(false),
        is_rehash_(false) {
    block_count_ = (capacity_ + way_ - 1) / way_;
    cache_.resize(block_count_);
    for (size_t i = 0; i < block_count_ - 1; i++) {
      cache_[i] = new CacheBlock(way_);
    }
    cache_[block_count_ - 1] = new CacheBlock(way_ + capacity_ - block_count_ * way_);
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_CACHE_RECORD_HITRATE", false,
                                   &is_record_hitrate_));
    BatchCache<K>::size_data_.resize(num_threads_);
    evicted.max_load_factor(0.8);
    evicted.set_empty_key_and_value(EMPTY_CACHE_KEY, -1);
    evicted.set_counternum(16);
    evicted.set_deleted_key(DELETED_CACHE_KEY);
    LOG(INFO) << "BlockLockLRUCache Init: capacity = " << capacity << ", way = " << way << ", Name = " << name_;
  }

  // BlockLockLRUCache(BlockLockLRUCache&&) = delete;
  
  // BlockLockLRUCache& operator=(BlockLockLRUCache&&) = delete;

  ~BlockLockLRUCache() override {
    LOG(INFO) << "~~BLRU Evicted Size = " << evicted.size_lockless()
              << ", Cache Size = " << this->size()
              << ", capacity_ = " << capacity_
              << ", Name = " << name_;
  }

  size_t get_capacity() override { return capacity_; }

  void set_capacity(size_t new_capacity) override {
    if (new_capacity == capacity_) {
      return;
    }
    LOG(INFO) << "Cache " << name_;
    if (new_capacity < capacity_) {
      LOG(INFO) << "Use shrink to change cache capacity.=================== " << capacity_ << " To " << new_capacity;
      new_way_ = (new_capacity + block_count_ - 1) / block_count_;
      full_ways_ = new_capacity + block_count_ - block_count_ * new_way_;
      __sync_bool_compare_and_swap(&is_shrinking_, false, true);
      __sync_bool_compare_and_swap(&is_expanding_, true, false);
    } else if (new_capacity > base_capacity_ * 2) {
      LOG(INFO) << "Use rehash to change cache capacity.++++++++++++++ " << capacity_ << " To " << new_capacity;
      size_t new_block_count_ = (new_capacity + way_ - 1) / way_;
      __sync_bool_compare_and_swap(&is_rehash_, false, true);
      __sync_bool_compare_and_swap(&is_expanding_, true, false);
      std::vector<CacheBlock*> new_cache_;
      new_cache_.resize(new_block_count_);
      for (size_t i = 0; i < new_block_count_ - 1; i++) {
        new_cache_[i] = new CacheBlock(way_);
      }
      new_cache_[new_block_count_ - 1] = new CacheBlock(way_ + new_capacity - new_block_count_ * way_);
      for (size_t i = 0; i < cache_.size(); i++) {
        CacheBlock& curr_block = *cache_[i];
        std::vector<CacheItem>& curr_cached = curr_block.cached;
        {
          mutex_lock l(curr_block.mtx_cached);
          for (const CacheItem& ci : curr_cached) {
            if (BlockLockLRUCache<K>::EMPTY_CACHE_KEY == ci.id) {
              continue;
            }
            CacheBlock& new_block = *new_cache_[ci.id % new_block_count_];
            std::vector<CacheItem>& new_cached = new_block.cached;
            for (size_t j = 0; j < new_cached.size(); ++j) {
              if (BlockLockLRUCache<K>::EMPTY_CACHE_KEY == new_cached[j].id) {
                new_cached[j].id = ci.id;
                new_cached[j].value = ci.value;
                break;
              }
            }
          }
        }
      }
      {
        mutex_lock l(rehash_mu_);
        cache_.swap(new_cache_);
        block_count_ = new_block_count_;
        base_capacity_ = capacity_;
      }
      __sync_bool_compare_and_swap(&is_rehash_, true, false);
    } else {
      LOG(INFO) << "Use append to change cache capacity.=================== " << capacity_ << " To " << new_capacity;
      new_way_ = (new_capacity + block_count_ - 1) / block_count_;
      full_ways_ = new_capacity + block_count_ - block_count_ * new_way_;
      __sync_bool_compare_and_swap(&is_expanding_, false, true);
      __sync_bool_compare_and_swap(&is_shrinking_, true, false);
    }
    capacity_ = new_capacity;
  }

  size_t get_cached_ids(K* cached_ids, size_t k_size, int64* cached_versions,
                        int64* cached_freqs) override {
    size_t true_size = 0;
    return true_size;
  }

  size_t get_evic_ids(K* evic_ids, size_t k_size) {
    mutex_lock l(evic_mu_);
    size_t evicted_size = evicted.size_lockless();
    if (evicted_size == 0) {
      return 0;
    }
    size_t true_size = 0;
    for (auto id : evicted) {
      if (true_size < k_size) {
        evic_ids[true_size] = id;
        ++true_size;
      }
    }
    for (size_t i = 0; i < true_size; i++) {
      evicted.erase_lockless(evic_ids[i]);
    }
    if (thread_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      thread_idx = (thread_count_idx++) % num_threads_;
    }
    BatchCache<K>::size_data_[thread_idx].size_ -= true_size;
    return true_size;
  }

  void update(const K* batch_ids, size_t batch_size, bool use_locking = true) {
    // __sync_fetch_and_add(&global_version_, 1);
    bool found;
    bool insert;
    size_t min_j;
    size_t min_version;
    int batch_hit = 0;
    int batch_miss = 0;
    int size_add = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      unsigned int bidx = id % block_count_;
      CacheBlock& curr_block = *cache_[bidx];
      std::vector<CacheItem>& curr_cached = curr_block.cached;
      found = false;
      mutex_lock l(curr_block.mtx_cached);
      for (size_t j = 0; j < curr_cached.size(); ++j) {
        if (id == curr_cached[j].id) {
          curr_cached[j].value = Env::Default()->NowMicros();
          found = true;
          ++batch_hit;
          break;
        }
      }
      if (!found) {
        batch_miss++;
        if (!evicted.erase_lockless(id)) {
          ++size_add;
        }
        if (is_expanding_ && curr_cached.size() < (new_way_ - (bidx >= full_ways_))) {
          curr_cached.emplace_back(id, Env::Default()->NowMicros());
        } else {
          unsigned int curr_size = curr_cached.size();
          if (is_shrinking_) {
            curr_size = (new_way_ - (bidx >= full_ways_));
          }
          insert = false;
          min_j = 0;
          min_version = std::numeric_limits<size_t>::max();
          for (size_t j = 0; j < curr_size; ++j) {
            if (BlockLockLRUCache<K>::EMPTY_CACHE_KEY == curr_cached[j].id) {
              curr_cached[j].id = id;
              curr_cached[j].value = Env::Default()->NowMicros();
              insert = true;
              break;
            } else if (min_version > curr_cached[j].value) {
              min_version = curr_cached[j].value;
              min_j = j;
            }
          }
          if (!insert) {
            evicted.insert_lockless(curr_cached[min_j].id);
            curr_cached[min_j].id = id;
            curr_cached[min_j].value = Env::Default()->NowMicros();
          }
          if (is_shrinking_) {
            for (size_t j = curr_size; j < curr_cached.size(); ++j) {
              if (BlockLockLRUCache<K>::EMPTY_CACHE_KEY != curr_cached[j].id) {
                evicted.insert_lockless(curr_cached[j].id);
              }
            }
          }
        }
      }
    }
    if (thread_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      thread_idx = (thread_count_idx++) % num_threads_;
    }
    BatchCache<K>::size_data_[thread_idx].size_ += size_add;
    if (is_record_hitrate_) {
      BatchCache<K>::size_data_[thread_idx].num_hit_ += batch_hit;
      BatchCache<K>::size_data_[thread_idx].num_miss_ += batch_miss;
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

  std::string name_;
  std::vector<CacheBlock*> cache_;
  typedef google::dense_hash_set_lockless<K> LocklessHashSet;
  LocklessHashSet evicted;
  mutex sync_idx_mu_;
  unsigned int block_count_;
  unsigned int full_ways_;
  unsigned int num_threads_;
  unsigned int way_;
  unsigned int new_way_;
  size_t capacity_;
  size_t base_capacity_;
  bool is_shrinking_;
  bool is_expanding_;
  bool is_rehash_;
  mutex evic_mu_;
  mutex rehash_mu_;
  // size_t global_version_;
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
