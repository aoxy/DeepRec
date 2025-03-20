#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BLOCK_LOCK_LFU_CACHE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_BLOCK_LOCK_LFU_CACHE_H_
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {

template <class K>
class BatchCache;

template <class K>
class PrefetchLFUNode;

// __thread static int sync_idx = -1;
static thread_local int sync_idx = -1;

template <class K>
class BlockLockLFUCache : public BatchCache<K> {
 public:
  BlockLockLFUCache(const std::string& name, size_t capacity, size_t way, int num_threads = 8)
      : name_(name),
        evic_idx_(0),
        num_threads_(num_threads),
        way_(way),
        sync_idx_count(0),
        is_expanding_(false),
        is_rehash_(false) {
    block_count_ = (capacity + way_ - 1) / way_;
    capacity_ = block_count_ * way_;
    base_capacity_ = capacity_;
    cache_.resize(block_count_);
    for (size_t i = 0; i < block_count_; i++) {
      cache_[i] = new CacheBlock(way_);
    }
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_CACHE_RECORD_HITRATE", false,
                                   &is_record_hitrate_));
    BatchCache<K>::size_data_.resize(num_threads_);
    evicted.max_load_factor(0.8);
    evicted.set_empty_key_and_value(EMPTY_CACHE_KEY, -1);
    evicted.set_counternum(16);
    evicted.set_deleted_key(DELETED_CACHE_KEY);
    LOG(INFO) << "BlockLockLFUCache Init: capacity = " << capacity << ", way = " << way << ", Name = " << name_;
  }

  // BlockLockLFUCache(BlockLockLFUCache&&) = delete;
  
  // BlockLockLFUCache& operator=(BlockLockLFUCache&&) = delete;

  ~BlockLockLFUCache() override {
    LOG(INFO) << "~~BLFU Evicted Size = " << evicted.size_lockless()
              << ", Cache Size = " << this->size()
              << ", Name = " << name_;
  }

  size_t get_capacity() override { return capacity_; }

  void set_capacity(size_t new_capacity) override {
    if (new_capacity <= capacity_) {
      return;
    }
    capacity_ = new_capacity;
    if (new_capacity > base_capacity_ * 2) {
      LOG(INFO) << "Use rehash to change cache capacity.++++++++++++++";
      size_t new_block_count_ = (new_capacity + way_ - 1) / way_;
      __sync_bool_compare_and_swap(&is_rehash_, false, true);
      __sync_bool_compare_and_swap(&is_expanding_, true, false);
      std::vector<CacheBlock*> new_cache_;
      new_cache_.resize(new_block_count_);
      for (size_t i = 0; i < new_block_count_; i++) {
        new_cache_[i] = new CacheBlock(way_);
      }
      for (size_t i = 0; i < cache_.size(); i++) {
        CacheBlock& curr_block = *cache_[i];
        {
          std::vector<CacheItem>& curr_cached = curr_block.cached;
          mutex_lock l(curr_block.mtx_cached);
          for (const CacheItem& ci : curr_cached) {
            CacheBlock& new_block = *new_cache_[ci.id % new_block_count_];
            std::vector<CacheItem>& new_cached = new_block.cached;
            bool insert = false;
            for (size_t j = 0; j < new_cached.size(); ++j) {
              if (BlockLockLFUCache<K>::EMPTY_CACHE_KEY == new_cached[j].id) {
                insert = true;
                new_cached[j].id = ci.id;
                new_cached[j].count = ci.count;
                break;
              }
            }
          }
        }
      }
      {
        // mutex_lock l(rehash_mu_);
        cache_.swap(new_cache_);
        block_count_ = new_block_count_;
        capacity_ = new_block_count_ * way_;
        base_capacity_ = capacity_;
      }
      __sync_bool_compare_and_swap(&is_rehash_, true, false);
    } else {
      LOG(INFO) << "Use append to change cache capacity.===================";
      new_way_ = (new_capacity + block_count_ - 1) / block_count_;
      __sync_bool_compare_and_swap(&is_expanding_, false, true);
    }
  }

  size_t get_cached_ids(K* cached_ids, size_t k_size, int64* cached_versions,
                        int64* cached_freqs) override {
    size_t true_size = 0;
    size_t total_cached_size = 0;
    std::unordered_set<K> cached_set;
    for (size_t i = 0; true_size < k_size; ++i) {
      CacheBlock& curr_block = *cache_[i % block_count_];
      std::vector<CacheItem>& curr_cached = curr_block.cached;
      size_t max_j = 0;
      size_t max_count = 0;
      mutex_lock l(curr_block.mtx_cached);
      for (size_t j = 0; j < curr_cached.size(); ++j) {
        if (curr_cached[j].id > 0 && !cached_set.count(curr_cached[j].id)) {
          ++total_cached_size;
          if (max_count < curr_cached[j].count) {
            max_count = curr_cached[j].count;
            max_j = j;
          }
        }
      }
      K max_id = curr_cached[max_j].id;
      if (max_id != BlockLockLFUCache<K>::EMPTY_CACHE_KEY &&
          !cached_set.count(max_id)) {
        --total_cached_size;
        cached_set.insert(max_id);
        cached_ids[true_size] = max_id;
        cached_freqs[true_size++] = curr_cached[max_j].count;
      }
      if ((i + 1) % block_count_ == 0) {
        if (total_cached_size == 0) {
          break;
        }
        total_cached_size = 0;
      }
    }
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
    if (sync_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      sync_idx = (sync_idx_count++) % num_threads_;
    }
    BatchCache<K>::size_data_[sync_idx].size_ -= true_size;
    return true_size;
  }

  void update(const K* batch_ids, size_t batch_size, bool use_locking = true) {
    bool found;
    bool insert;
    size_t min_j;
    size_t min_count;
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
          ++curr_cached[j].count;
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
        if (is_expanding_ && curr_cached.size() < new_way_) {
          curr_cached.emplace_back(id);
        } else {
          insert = false;
          min_j = 0;
          min_count = std::numeric_limits<size_t>::max();
          for (size_t j = 0; j < curr_cached.size(); ++j) {
            if (BlockLockLFUCache<K>::EMPTY_CACHE_KEY == curr_cached[j].id) {
              curr_cached[j].id = id;
              curr_cached[j].count = 1;
              insert = true;
              break;
            } else if (min_count > curr_cached[j].count) {
              min_count = curr_cached[j].count;
              min_j = j;
            }
          }
          if (!insert) {
            evicted.insert_lockless(curr_cached[min_j].id);
            curr_cached[min_j].id = id;
            curr_cached[min_j].count = 1;
          }
        }
      }
    }
    if (sync_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      sync_idx = (sync_idx_count++) % num_threads_;
    }
    BatchCache<K>::size_data_[sync_idx].size_ += size_add;
    if (is_record_hitrate_) {
      BatchCache<K>::size_data_[sync_idx].num_hit_ += batch_hit;
      BatchCache<K>::size_data_[sync_idx].num_miss_ += batch_miss;
    }
  }

  void update(const K* batch_ids, size_t batch_size, const int64* batch_version,
              const int64* batch_freqs, bool use_locking = true) override {
    bool found;
    bool insert;
    size_t min_j;
    size_t min_count;
    int batch_hit = 0;
    int batch_miss = 0;
    int size_add = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      int64 freq = batch_freqs[i];
      CacheBlock& curr_block = *cache_[id % block_count_];
      std::vector<CacheItem>& curr_cached = curr_block.cached;
      found = false;
      mutex_lock l(curr_block.mtx_cached);
      for (size_t j = 0; j < curr_cached.size(); ++j) {
        if (id == curr_cached[j].id) {
          curr_cached[j].count += freq;
          found = true;
          batch_hit += freq;
          break;
        }
      }
      if (!found) {
        batch_miss++;
        if (!evicted.erase_lockless(id)) {
          ++size_add;
        }
        if (is_expanding_ && curr_cached.size() < new_way_) {
          curr_cached.emplace_back(id);
        } else {
          insert = false;
          min_j = 0;
          min_count = std::numeric_limits<size_t>::max();
          for (size_t j = 0; j < curr_cached.size(); ++j) {
            if (BlockLockLFUCache<K>::EMPTY_CACHE_KEY == curr_cached[j].id) {
              curr_cached[j].id = id;
              curr_cached[j].count = freq;
              insert = true;
              break;
            } else if (min_count > curr_cached[j].count) {
              min_count = curr_cached[j].count;
              min_j = j;
            }
          }
          if (!insert) {
            evicted.insert_lockless(curr_cached[min_j].id);
            curr_cached[min_j].id = id;
            curr_cached[min_j].count = freq;
          }
        }
      }
    }
    if (sync_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      sync_idx = (sync_idx_count++) % num_threads_;
    }
    BatchCache<K>::size_data_[sync_idx].size_ += size_add;
    if (is_record_hitrate_) {
      BatchCache<K>::size_data_[sync_idx].num_hit_ += batch_hit;
      BatchCache<K>::size_data_[sync_idx].num_miss_ += batch_miss;
    }
  }

  void add_to_prefetch_list(const K* batch_ids, const size_t batch_size) {
    mutex_lock l(prefetch_mu_);
    if (sync_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      sync_idx = (sync_idx_count++) % num_threads_;
    }
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it_prefetch = prefetch_id_table.find(id);
      if (it_prefetch == prefetch_id_table.end()) {
        int64 freq = 1;
        CacheBlock& curr_block = *cache_[id % block_count_];
        std::vector<CacheItem>& curr_cached = curr_block.cached;
        mutex_lock l(curr_block.mtx_cached);
        for (size_t j = 0; j < curr_cached.size(); ++j) {
          if (id == curr_cached[j].id) {
            curr_cached[j].id = BlockLockLFUCache<K>::EMPTY_CACHE_KEY;
            freq = curr_cached[j].count;
            BatchCache<K>::size_data_[sync_idx].size_--;
            break;
          }
        }
        prefetch_id_table[id] = new PrefetchLFUNode<K>(id, freq);
      } else {
        it_prefetch->second->Ref();
      }
    }
  }

  void add_to_cache(const K* batch_ids, const size_t batch_size) {
    mutex_lock l(prefetch_mu_);
    std::vector<K> ids_to_cache(batch_size);
    std::vector<int64> freqs_to_cache(batch_size);
    int64 nums_to_cache = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it_prefetch = prefetch_id_table.find(id);
      if (it_prefetch == prefetch_id_table.end()) {
        LOG(FATAL) << "The id should be prefetched before being used.";
      }
      it_prefetch->second->UnRef();
      if (it_prefetch->second->ref_count() == 0) {
        int64 freq = it_prefetch->second->freq();
        delete it_prefetch->second;
        prefetch_id_table.erase(id);
        ids_to_cache[nums_to_cache] = id;
        freqs_to_cache[nums_to_cache] = freq;
        nums_to_cache++;
      }
    }
    const int64* versions_to_cache = nullptr;
    update(ids_to_cache.data(), nums_to_cache, versions_to_cache,
           freqs_to_cache.data(), false);
  }

 private:
  class CacheItem {
   public:
    K id;
    size_t count;
    CacheItem(K id) : id(id), count(1) {}
    CacheItem() : id(BlockLockLFUCache<K>::EMPTY_CACHE_KEY), count(0) {}
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
  std::unordered_map<K, PrefetchLFUNode<K>*> prefetch_id_table;
  std::vector<CacheBlock*> cache_;
  typedef google::dense_hash_set_lockless<K> LocklessHashSet;
  LocklessHashSet evicted;
  mutex prefetch_mu_;
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
  mutex evic_mu_;
  mutex rehash_mu_;
  bool is_record_hitrate_;
  unsigned int sync_idx_count;
  static const K EMPTY_CACHE_KEY;
  static const K DELETED_CACHE_KEY;
};
template <class K>
const K BlockLockLFUCache<K>::EMPTY_CACHE_KEY = -1;

template <class K>
const K BlockLockLFUCache<K>::DELETED_CACHE_KEY = -2;

}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
