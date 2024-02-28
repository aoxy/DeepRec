#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
#include <iostream>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <list>
#include <limits>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {

template <class K>
class BatchCache {
 public:
  BatchCache() {}
  virtual ~BatchCache() {}
  void update(const Tensor& t) {
    update((K*)t.data(), t.NumElements());
  }
  void add_to_prefetch_list(const Tensor& t) {
    add_to_prefetch_list((K*)t.data(), t.NumElements());
  }
  void add_to_cache(const Tensor& t) {
    add_to_cache((K*)t.data(), t.NumElements());
  }

  void update(const Tensor& t, const Tensor& counts_tensor) {
    update((K*)t.data(), t.NumElements(),
           nullptr, (int64*)counts_tensor.data());
  }

  virtual size_t get_evic_ids(K* evic_ids, size_t k_size) = 0;
  virtual size_t get_cached_ids(K* cached_ids, size_t k_size,
                                int64* cached_versions,
                                int64* cached_freqs) = 0;
  virtual void update(const K* batch_ids, size_t batch_size,
                      bool use_locking=true) = 0;
  virtual void update(const K* batch_ids, size_t batch_size,
                      const int64* batch_versions,
                      const int64* batch_freqs,
                      bool use_locking=true) = 0;
  virtual void add_to_prefetch_list(
      const K* batch_ids, size_t batch_size) = 0;
  virtual void add_to_cache(
      const K* batch_ids, size_t batch_size) = 0;
  virtual size_t size() = 0;
  virtual size_t get_capacity() { return 0; }
  virtual void set_capacity(size_t new_capacity) {}
  virtual void reset_status() {
     *num_hit = 0;
     *num_miss = 0;
  }
  virtual float hit_rate() {
    float hit_rate = 0.0;
    if (*num_hit > 0 || *num_miss > 0) {
      hit_rate = *num_hit * 100.0 / (*num_hit + *num_miss);
    }
    return hit_rate;
  }
  std::string DebugString() {
    return strings::StrCat("HitRate = " , hit_rate(),
                          " %, visit_count = ", *num_hit + *num_miss,
                           ", hit_count = ", *num_hit);
  }
  virtual mutex_lock maybe_lock_cache(
      mutex& mu, mutex& temp_mu,bool use_locking) {
    if (use_locking) {
      mutex_lock l(mu);
      return l;
    } else {
      mutex_lock l(temp_mu);
      return l;
    }
  }

 protected:
  int64 *num_hit;
  int64 *num_miss;
};

template<class K>
class PrefetchNode {
 public:
  explicit PrefetchNode(): key_(-1), ref_count_(1) {}
  explicit PrefetchNode(K id): key_(id), ref_count_(1) {}
  virtual ~PrefetchNode() {}
  virtual void Ref() {
    ref_count_++;
  };
  virtual void UnRef() {
    ref_count_--;
  };
  virtual K key() {
    return key_;
  }
  virtual int64 ref_count() {
    return ref_count_;
  }
 protected:
   K key_;
   int64 ref_count_;
};

template<class K>
class PrefetchLFUNode: public PrefetchNode<K>{
 public:
  explicit PrefetchLFUNode(K id){
    PrefetchNode<K>::key_ = id;
    PrefetchNode<K>::ref_count_ = 1;
    freq_ = 1;
  }

  PrefetchLFUNode(K id, int64 freq){
    PrefetchNode<K>::key_ = id;
    PrefetchNode<K>::ref_count_ = 1;
    freq_ = freq;
  }

  void Ref() override {
    PrefetchNode<K>::ref_count_++;
    freq_++;
  }

  int64 freq() {
    return freq_;
  }
 private:
  int64 freq_;
};

template <class K>
class LRUCache : public BatchCache<K> {
 public:
  LRUCache() {
    mp.clear();
    head = new LRUNode(0);
    tail = new LRUNode(0);
    head->next = tail;
    tail->pre = head;
    BatchCache<K>::num_hit = new int64();
    BatchCache<K>::num_miss = new int64();
  }

  size_t size() {
    mutex_lock l(mu_);
    return mp.size();
  }

  size_t get_evic_ids(K* evic_ids, size_t k_size) {
    mutex_lock l(mu_);
    size_t true_size = 0;
    LRUNode *evic_node = tail->pre;
    LRUNode *rm_node = evic_node;
    for (size_t i = 0; i < k_size && evic_node != head; ++i) {
      evic_ids[i] = evic_node->id;
      rm_node = evic_node;
      evic_node = evic_node->pre;
      mp.erase(rm_node->id);
      delete rm_node;
      true_size++;
    }
    evic_node->next = tail;
    tail->pre = evic_node;
    return true_size;
  }

  size_t get_cached_ids(K* cached_ids, size_t k_size,
                        int64* cached_versions,
                        int64* cached_freqs) override {
    mutex_lock l(mu_);
    LRUNode* it = head->next;
    size_t i;
    for (i = 0; i < k_size && it != tail; i++, it = it->next) {
      cached_ids[i] = it->id;
    }
    return i;
  }

  void update(const K* batch_ids, size_t batch_size,
              bool use_locking=true) {
    mutex temp_mu;
    auto lock = BatchCache<K>::maybe_lock_cache(mu_, temp_mu, use_locking);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      typename std::map<K, LRUNode *>::iterator it = mp.find(id);
      if (it != mp.end()) {
        LRUNode *node = it->second;
        node->pre->next = node->next;
        node->next->pre = node->pre;
        head->next->pre = node;
        node->next = head->next;
        head->next = node;
        node->pre = head;
        *BatchCache<K>::num_hit++;
      } else {
        LRUNode *newNode = new LRUNode(id);
        head->next->pre = newNode;
        newNode->next = head->next;
        head->next = newNode;
        newNode->pre = head;
        mp[id] = newNode;
        *BatchCache<K>::num_miss++;
      }
    }
  }

  void update(const K* batch_ids, size_t batch_size,
              const int64* batch_version,
              const int64* batch_freqs,
              bool use_locking = true) override {
    //TODO: add to rank accroding to the version of ids
    update(batch_ids, batch_size);
  }

  void add_to_prefetch_list(const K* batch_ids, const size_t batch_size) {
    mutex_lock l(mu_);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it_prefetch = prefetch_id_table.find(id);
      if (it_prefetch == prefetch_id_table.end()) {
        auto it_cache = mp.find(id);
        if (it_cache != mp.end()) {
          LRUNode *node = it_cache->second;
          node->pre->next = node->next;
          node->next->pre = node->pre;
          delete node;
          mp.erase(id);
        }
        prefetch_id_table[id] = new PrefetchNode<K>(id);
      } else {
        it_prefetch->second->Ref();
      }
    }
  }

  void add_to_cache(const K* batch_ids, const size_t batch_size) {
    mutex_lock l(mu_);
    std::vector<K> ids_to_cache(batch_size);
    int64 nums_to_cache = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it_prefetch = prefetch_id_table.find(id);
      if (it_prefetch == prefetch_id_table.end()) {
        LOG(FATAL)<<"The id should be prefetched before being used.";
      }
      it_prefetch->second->UnRef();
      if (it_prefetch->second->ref_count() == 0) {
        delete it_prefetch->second;
        prefetch_id_table.erase(id);
        ids_to_cache[nums_to_cache] = id;
        nums_to_cache++;
      }
    }
    update(ids_to_cache.data(), nums_to_cache, false);
  }

 private:
  class LRUNode {
   public:
     K id;
     LRUNode *pre, *next;
     LRUNode(K id) : id(id), pre(nullptr), next(nullptr) {}
  };
  LRUNode *head, *tail;
  std::map<K, LRUNode*> mp;
  std::unordered_map<K, PrefetchNode<K>*> prefetch_id_table;
  mutex mu_;
};

// __thread static int sync_idx = -1;
static thread_local int sync_idx = -1;

template <class K>
class BlockLockLFUCache : public BatchCache<K> {
 public:
  BlockLockLFUCache(size_t capacity, size_t way, int num_threads=8)
      : evic_idx_(0),
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
    size_ = new size_t[num_threads_ * 16]();
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_CACHE_RECORD_HITRATE", false,
                                   &is_record_hitrate_));
    BatchCache<K>::num_hit = new int64[num_threads_ * 16]();
    BatchCache<K>::num_miss = new int64[num_threads_ * 16]();
  }

  ~BlockLockLFUCache() override {
    delete[] size_;
    delete[] BatchCache<K>::num_hit;
    delete[] BatchCache<K>::num_miss;
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
            std::vector<K>& new_evicted = new_block.evicted;
            bool insert = false;
            for (size_t j = 0; j < new_cached.size(); ++j) {
              if (BlockLockLFUCache<K>::EMPTY_CACHE_ == new_cached[j].id) {
                insert = true;
                new_cached[j].id = ci.id;
                new_cached[j].count = ci.count;
                break;
              }
            }
            if (!insert) {
              new_evicted.emplace_back(ci.id);
            }
          }
        }
        {
          std::vector<K>& curr_evicted = curr_block.evicted;
          mutex_lock l(curr_block.mtx_evicted);
          for (const K& id : curr_evicted) {
            CacheBlock& new_block = *new_cache_[id % new_block_count_];
            std::vector<K>& new_evicted = new_block.evicted;
            new_evicted.emplace_back(id);
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

  size_t size() {
    size_t total_size = size_[0];
    for (size_t j = 1; j < num_threads_; ++j) {
      total_size += size_[j * 16];
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
  
  size_t get_cached_ids(K* cached_ids, size_t k_size,
                        int64* cached_versions,
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
      if (max_id != BlockLockLFUCache<K>::EMPTY_CACHE_ && !cached_set.count(max_id)) {
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
    size_t true_size = 0;
    for (size_t i = 0; true_size < k_size && i < block_count_; ++i) {
      CacheBlock& curr_block = *cache_[evic_idx_];
      std::vector<K>& curr_evicted = curr_block.evicted;
      mutex_lock l(curr_block.mtx_evicted);
      while (true_size < k_size && !curr_evicted.empty()) {
        evic_ids[true_size++] = *curr_evicted.end();
        curr_evicted.pop_back();
      }
      evic_idx_ = (evic_idx_ + 1) % block_count_;
    }
    if (true_size < k_size) {
      size_t total_cached_size = 0;
      for (size_t i = 0; true_size < k_size; ++i) {
        CacheBlock& curr_block = *cache_[i % block_count_];
        std::vector<CacheItem>& curr_cached = curr_block.cached;
        size_t min_j = 0;
        size_t min_count = std::numeric_limits<size_t>::max();
        mutex_lock l(curr_block.mtx_cached);
        for (size_t j = 0; j < curr_cached.size(); ++j) {
          if (curr_cached[j].id != BlockLockLFUCache<K>::EMPTY_CACHE_) {
            ++total_cached_size;
            if (min_count >= curr_cached[j].count) {
              min_count = curr_cached[j].count;
              min_j = j;
            }
          }
        }
        if (curr_cached[min_j].id != BlockLockLFUCache<K>::EMPTY_CACHE_) {
          --total_cached_size;
          evic_ids[true_size++] = curr_cached[min_j].id;
          curr_cached[min_j].id = BlockLockLFUCache<K>::EMPTY_CACHE_;
          curr_block.full = false;
        }
        if ((i + 1) % block_count_ == 0) {
          if (total_cached_size == 0) {
            break;
          }
          total_cached_size = 0;
        }
      }
    }
    if (sync_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      sync_idx = (sync_idx_count++) % num_threads_;
    }
    size_[sync_idx * 16] -= true_size;
    return true_size;
  }

  void update(const K* batch_ids, size_t batch_size,
                   bool use_locking=true) {
    bool found;
    bool insert;
    size_t min_j;
    size_t min_count;
    int batch_hit = 0;
    int batch_miss = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      CacheBlock& curr_block = *cache_[id % block_count_];
      std::vector<CacheItem>& curr_cached = curr_block.cached;
      found = false;
      mutex_lock l(curr_block.mtx_cached);
      for (size_t j = 0; j < curr_cached.size(); ++j) {
        if (id == curr_cached[j].id) {
          found = true;
          ++curr_cached[j].count;
          ++batch_hit;
          break;
        }
      }
      if (!found) {
        batch_miss++;
        if (!curr_block.full) {
          insert = false;
          for (size_t j = 0; j < curr_cached.size(); ++j) {
            if (BlockLockLFUCache<K>::EMPTY_CACHE_ == curr_cached[j].id) {
              insert = true;
              curr_cached[j].id = id;
              curr_cached[j].count = 1;
              break;
            }
          }
          curr_block.full = !insert;
        } else if (is_expanding_ && curr_cached.size() < new_way_) {
          curr_cached.emplace_back(id);
        } else {
          min_j = 0;
          min_count = std::numeric_limits<size_t>::max();
          for (size_t j = 0; j < curr_cached.size(); ++j) {
            if (min_count > curr_cached[j].count) {
              min_count = curr_cached[j].count;
              min_j = j;
            }
          }
          {
            mutex_lock l(curr_block.mtx_evicted);
            curr_block.evicted.push_back(curr_cached[min_j].id);
          }
          curr_cached[min_j].id = id;
          curr_cached[min_j].count = 1;
        }
      }
    }
    if (sync_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      sync_idx = (sync_idx_count++) % num_threads_;
    }
    size_[sync_idx * 16] += batch_miss;
    if (is_record_hitrate_) {
      this->num_hit[sync_idx * 16] += batch_hit;
      this->num_miss[sync_idx * 16] += batch_miss;
    }
  }

  void update(const K* batch_ids, size_t batch_size,
                    const int64* batch_version,
                    const int64* batch_freqs,
                    bool use_locking = true) override {
    bool found;
    bool insert;
    size_t min_j;
    size_t min_count;
    int batch_hit = 0;
    int batch_miss = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      int64 freq = batch_freqs[i];
      CacheBlock& curr_block = *cache_[id % block_count_];
      std::vector<CacheItem>& curr_cached = curr_block.cached;
      found = false;
      mutex_lock l(curr_block.mtx_cached);
      for (size_t j = 0; j < curr_cached.size(); ++j) {
        if (id == curr_cached[j].id) {
          found = true;
          curr_cached[j].count += freq;
          batch_hit += freq;
          break;
        }
      }
      if (!found) {
        batch_miss++;
        if (!curr_block.full) {
          insert = false;
          for (size_t j = 0; j < curr_cached.size(); ++j) {
            if (BlockLockLFUCache<K>::EMPTY_CACHE_ == curr_cached[j].id) {
              insert = true;
              curr_cached[j].id = id;
              curr_cached[j].count = freq;
              break;
            }
          }
          curr_block.full = !insert;
        } else {
          min_j = 0;
          min_count = std::numeric_limits<size_t>::max();
          for (size_t j = 0; j < curr_cached.size(); ++j) {
            if (min_count > curr_cached[j].count) {
              min_count = curr_cached[j].count;
              min_j = j;
            }
          }
          {
            mutex_lock l(curr_block.mtx_evicted);
            curr_block.evicted.push_back(curr_cached[min_j].id);
          }
          curr_cached[min_j].id = id;
          curr_cached[min_j].count = freq;
        }
      }
    }
    if (sync_idx < 0) {
      mutex_lock l(sync_idx_mu_);
      sync_idx = (sync_idx_count++) % num_threads_;
    }
    size_[sync_idx * 16] += batch_miss;
    // __sync_fetch_and_add(size_ + (sync_idx % num_threads_), batch_miss);
    if (is_record_hitrate_) {
      this->num_hit[sync_idx * 16] += batch_hit;
      this->num_miss[sync_idx * 16] += batch_miss;
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
            curr_cached[j].id = BlockLockLFUCache<K>::EMPTY_CACHE_;
            curr_block.full = false;
            freq = curr_cached[j].count;
            // __sync_fetch_and_sub(size_ + (sync_idx % num_threads_), 1);
            size_[sync_idx * 16]--;
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
    update(ids_to_cache.data(), nums_to_cache,
           versions_to_cache, freqs_to_cache.data(),
           false);
  }

 private:
  class CacheItem {
   public:
    K id;
    size_t count;
    CacheItem(K id) : id(id), count(1) {}
    CacheItem() : id(BlockLockLFUCache<K>::EMPTY_CACHE_), count(0) {}
  };
  class CacheBlock {
   public:
    std::vector<CacheItem> cached;
    std::vector<K> evicted;
    mutex mtx_cached;
    mutex mtx_evicted;
    bool full;
    CacheBlock(size_t way) : full(false) { cached.resize(way); }
    CacheBlock() = delete;
    CacheBlock(CacheBlock&) = delete;
    CacheBlock& operator=(CacheBlock&) = delete;
  };

  std::unordered_map<K, PrefetchLFUNode<K>*> prefetch_id_table;
  std::vector<CacheBlock*> cache_;
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
  mutex rehash_mu_;
  size_t *size_;
  bool is_record_hitrate_;
  unsigned int sync_idx_count;
  static const K EMPTY_CACHE_;
};
template <class K>
const K BlockLockLFUCache<K>::EMPTY_CACHE_ = -1;

template <class K>
class LFUCache : public BatchCache<K> {
 public:
  LFUCache() {
    min_freq = std::numeric_limits<size_t>::max();
    max_freq = 0;
    freq_table.emplace_back(std::pair<std::list<LFUNode>*, int64>(
      new std::list<LFUNode>, 0));
    BatchCache<K>::num_hit = new int64();
    BatchCache<K>::num_miss = new int64();
  }

  size_t size() {
    mutex_lock l(mu_);
    return key_table.size();
  }

  size_t get_cached_ids(K* cached_ids, size_t k_size,
                        int64* cached_versions,
                        int64* cached_freqs) override {
    mutex_lock l(mu_);
    size_t i = 0;
    size_t curr_freq = max_freq;
    auto it = freq_table[max_freq - 1].first->begin();
    while (i < k_size && curr_freq >= min_freq) {
      cached_ids[i] = (*it).key;
      cached_freqs[i] = (*it).freq;
      i++;
      it++;
      if (it == freq_table[curr_freq - 1].first->end()) {
        do {
          curr_freq--;
        } while (freq_table[curr_freq - 1].second == 0
            && curr_freq >= min_freq);
        if (curr_freq >= min_freq) {
          it = freq_table[curr_freq - 1].first->begin();
        }
      }
    }
    return i;
  }

  size_t get_evic_ids(K *evic_ids, size_t k_size) {
    mutex_lock l(mu_);
    size_t true_size = 0;
    size_t st_freq = min_freq;
    for (size_t i = 0; i < k_size && key_table.size() > 0; ++i) {
      auto rm_it = freq_table[st_freq-1].first->back();
      key_table.erase(rm_it.key);
      evic_ids[i] = rm_it.key;
      ++true_size;
      freq_table[st_freq-1].first->pop_back();
      freq_table[st_freq-1].second--;
      if (freq_table[st_freq-1].second == 0) {
        ++st_freq;
        while (st_freq <= max_freq) {
          if (freq_table[st_freq-1].second == 0) {
            ++st_freq;
          } else {
            min_freq = st_freq;
            break;
          }
        }
        if (st_freq > max_freq) {
          reset_min_and_max_freq();
        }
      }
    }
    return true_size;
  }

  void update(const K *batch_ids, size_t batch_size,
              bool use_locking=true) {
    mutex temp_mu;
    auto lock = BatchCache<K>::maybe_lock_cache(mu_, temp_mu, use_locking);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it = key_table.find(id);
      if (it == key_table.end()) {
        freq_table[0].first->emplace_front(LFUNode(id, 1));
        freq_table[0].second++;
        key_table[id] = freq_table[0].first->begin();
        min_freq = 1;
        max_freq = std::max(max_freq, min_freq);
        *BatchCache<K>::num_miss++;
      } else {
        typename std::list<LFUNode>::iterator node = it->second;
        size_t freq = node->freq;
        freq_table[freq-1].first->erase(node);
        freq_table[freq-1].second--;
        if (freq_table[freq-1].second == 0) {
          if (min_freq == freq)
            min_freq += 1;
        }
        if (freq == freq_table.size()) {
          freq_table.emplace_back(std::pair<std::list<LFUNode>*, int64>(
           new std::list<LFUNode>, 0));
        }
        max_freq = std::max(max_freq, freq + 1);
        freq_table[freq].first->emplace_front(LFUNode(id, freq + 1));
        freq_table[freq].second++;
        key_table[id] = freq_table[freq].first->begin();
        *BatchCache<K>::num_hit++;
      }
    }
  }

  void update(const K *batch_ids, const size_t batch_size,
              const int64* batch_versions,
              const int64* batch_freqs,
              bool use_locking = true) override {
    mutex temp_mu;
    auto lock = BatchCache<K>::maybe_lock_cache(mu_, temp_mu, use_locking);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it = key_table.find(id);
      size_t freq =  batch_freqs[i];
      if (it == key_table.end()) {
        if (freq < min_freq) {
          min_freq = freq;
        }

        if (freq > max_freq) {
          max_freq = freq;
          int64 prev_size = freq_table.size();
          if (max_freq > prev_size) {
            freq_table.resize(max_freq, std::pair<std::list<LFUNode>*, int64>(
                nullptr, 0));
            for (int64 j = prev_size; j < max_freq; j++) {
              freq_table[j].first = new std::list<LFUNode>;
            }
          }
        }
        freq_table[freq-1].first->emplace_front(LFUNode(id, freq));
        freq_table[freq-1].second++;
        key_table[id] = freq_table[freq-1].first->begin();
        *BatchCache<K>::num_miss++;
      } else {
        typename std::list<LFUNode>::iterator node = it->second;
        size_t last_freq = node->freq;
        size_t curr_freq = last_freq + freq;
        freq_table[last_freq-1].first->erase(node);
        freq_table[last_freq-1].second--;
        
        if (curr_freq > max_freq) {
          max_freq = curr_freq;
          freq_table.resize(max_freq, std::pair<std::list<LFUNode>*, int64>(
           new std::list<LFUNode>, 0));
        }

        if (freq_table[last_freq-1].second == 0) {
          if (min_freq == last_freq){
            update_min_freq();
          }
        }
       
        freq_table[curr_freq-1].first->emplace_front(LFUNode(id, curr_freq));
        freq_table[curr_freq-1].second++;
        key_table[id] = freq_table[curr_freq-1].first->begin();
        *BatchCache<K>::num_hit++;
      }
    }
  }

  void add_to_prefetch_list(const K* batch_ids, const size_t batch_size) {
    mutex_lock l(mu_);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it_prefetch = prefetch_id_table.find(id);
      if (it_prefetch == prefetch_id_table.end()) {
        auto it_cache = key_table.find(id);
        if (it_cache != key_table.end()) {
          auto cache_node = it_cache->second;
          int64 freq = cache_node->freq;
          freq_table[freq - 1].first->erase(cache_node);
          freq_table[freq - 1].second--;
          key_table.erase(id);
          if (freq_table[freq - 1].second == 0) {
            if (freq == max_freq) {
              update_max_freq();
            }
            if (freq == min_freq) {
              update_min_freq();
            }
          }
          prefetch_id_table[id] = new PrefetchLFUNode<K>(id, freq);
        } else {
          prefetch_id_table[id] = new PrefetchLFUNode<K>(id);
        }
      } else {
        it_prefetch->second->Ref();
      }
    }
  }

  void add_to_cache(const K* batch_ids, const size_t batch_size) {
    mutex_lock l(mu_);
    std::vector<K> ids_to_cache(batch_size);
    std::vector<int64> freqs_to_cache(batch_size);
    int64 nums_to_cache = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it_prefetch = prefetch_id_table.find(id);
      if (it_prefetch == prefetch_id_table.end()) {
        LOG(FATAL)<<"The id should be prefetched before being used.";
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
    update(ids_to_cache.data(), nums_to_cache,
           versions_to_cache, freqs_to_cache.data(),
           false);
  }


 private:
  void reset_min_and_max_freq() {
    min_freq = std::numeric_limits<size_t>::max();
    max_freq = 0;
  }

  void update_min_freq() {
    size_t i;
    for (i = min_freq + 1; i <= max_freq; i++) {
      if(freq_table[i-1].second != 0) {
        min_freq = i;
        break;
      }
    }
    if (i > max_freq) {
      reset_min_and_max_freq();
    }
  }

  void update_max_freq() {
    size_t i;
    for (i = max_freq - 1; i >=min_freq; i--) {
      if(freq_table[i-1].second != 0) {
        max_freq = i;
        break;
      }
    }
    if (i < min_freq) {
      reset_min_and_max_freq();
    }
  }


  class LFUNode {
   public:
    K key;
    size_t freq;
    LFUNode(K key, size_t freq) : key(key), freq(freq) {}
  };
  size_t min_freq;
  size_t max_freq;
  std::vector<std::pair<std::list<LFUNode>*, int64>> freq_table;
  std::unordered_map<K, typename std::list<LFUNode>::iterator> key_table;
  std::unordered_map<K, PrefetchLFUNode<K>*> prefetch_id_table;
  mutex mu_;
};

} // embedding
} // tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
