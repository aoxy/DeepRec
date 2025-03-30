#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
#include <iostream>
#include <limits>
#include <list>
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {

struct alignas(64) SizeDataBlock {
  size_t size_;
  size_t num_hit_;
  size_t num_miss_;  // Padding to fill one cache line (64 bytes)
  char _padding[64 - 3 * sizeof(size_t)];
  SizeDataBlock() : size_(0), num_hit_(0), num_miss_(0) {}
  void reset() {
    size_ = 0;
    num_hit_ = 0;
    num_miss_ = 0;
  }
};

template <class K>
class BatchCache {
 public:
  BatchCache() {}
  virtual ~BatchCache() {}
  void update(const Tensor& t) { update((K*)t.data(), t.NumElements()); }
  void add_to_prefetch_list(const Tensor& t) {
    add_to_prefetch_list((K*)t.data(), t.NumElements());
  }
  void add_to_cache(const Tensor& t) {
    add_to_cache((K*)t.data(), t.NumElements());
  }

  void update(const Tensor& t, const Tensor& counts_tensor) {
    update((K*)t.data(), t.NumElements(), nullptr,
           (int64*)counts_tensor.data());
  }

  virtual size_t get_evic_ids(K* evic_ids, size_t k_size) = 0;
  virtual size_t get_cached_ids(K* cached_ids, size_t k_size,
                                int64* cached_versions,
                                int64* cached_freqs) = 0;
  virtual void update(const K* batch_ids, size_t batch_size,
                      bool use_locking = true) = 0;
  virtual void update(const K* batch_ids, size_t batch_size,
                      const int64* batch_versions, const int64* batch_freqs,
                      bool use_locking = true) = 0;
  virtual void add_to_prefetch_list(const K* batch_ids, size_t batch_size) = 0;
  virtual void add_to_cache(const K* batch_ids, size_t batch_size) = 0;
  virtual size_t size() {
    size_t total_size = 0;
    for (SizeDataBlock& sdb : size_data_) {
      total_size += sdb.size_;
    }
    return total_size;
  }
  virtual size_t get_capacity() = 0;
  virtual void set_capacity(size_t new_capacity) = 0;
  virtual void reset_status() {
    for (SizeDataBlock& sdb : size_data_) {
      sdb.reset();
    }
  }
  virtual float hit_rate() {
    float hit_rate = 0.0;
    size_t total_num_hit = 0;
    size_t total_num_access = 0;
    for (SizeDataBlock& sdb : size_data_) {
      total_num_hit += sdb.num_hit_;
      total_num_access += sdb.num_miss_;
    }
    total_num_access += total_num_hit;
    if (total_num_access > 0) {
      hit_rate = total_num_hit * 100.0 / (total_num_access);
    }
    return hit_rate;
  }
  std::string DebugString() {
    float hit_rate = 0.0;
    size_t total_num_hit = 0;
    size_t total_num_access = 0;
    for (SizeDataBlock& sdb : size_data_) {
      total_num_hit += sdb.num_hit_;
      total_num_access += sdb.num_miss_;
    }
    total_num_access += total_num_hit;
    if (total_num_access > 0) {
      hit_rate = total_num_hit * 100.0 / (total_num_access);
    }
    return strings::StrCat("HitRate = ", hit_rate,
                           " %, visit_count = ", total_num_access,
                           ", hit_count = ", total_num_hit);
  }
  virtual mutex_lock maybe_lock_cache(mutex& mu, mutex& temp_mu,
                                      bool use_locking) {
    if (use_locking) {
      mutex_lock l(mu);
      return l;
    } else {
      mutex_lock l(temp_mu);
      return l;
    }
  }

 protected:
  std::vector<SizeDataBlock> size_data_;
};

template <class K>
class PrefetchNode {
 public:
  explicit PrefetchNode() : key_(-1), ref_count_(1) {}
  explicit PrefetchNode(K id) : key_(id), ref_count_(1) {}
  virtual ~PrefetchNode() {}
  virtual void Ref() { ref_count_++; };
  virtual void UnRef() { ref_count_--; };
  virtual K key() { return key_; }
  virtual int64 ref_count() { return ref_count_; }

 protected:
  K key_;
  int64 ref_count_;
};

template <class K>
class PrefetchLFUNode : public PrefetchNode<K> {
 public:
  explicit PrefetchLFUNode(K id) {
    PrefetchNode<K>::key_ = id;
    PrefetchNode<K>::ref_count_ = 1;
    freq_ = 1;
  }

  PrefetchLFUNode(K id, int64 freq) {
    PrefetchNode<K>::key_ = id;
    PrefetchNode<K>::ref_count_ = 1;
    freq_ = freq;
  }

  void Ref() override {
    PrefetchNode<K>::ref_count_++;
    freq_++;
  }

  int64 freq() { return freq_; }

 private:
  int64 freq_;
};

template <class K>
class LRUCache : public BatchCache<K> {
 public:
  LRUCache(size_t capacity) : capacity_(capacity) {
    mp.clear();
    head = new LRUNode(0);
    tail = new LRUNode(0);
    head->next = tail;
    tail->pre = head;
    BatchCache<K>::size_data_.resize(1);
  }

  size_t size() override {
    mutex_lock l(mu_);
    return mp.size();
  }

  size_t get_capacity() override { return capacity_; }

  void set_capacity(size_t new_capacity) override {}

  size_t get_evic_ids(K* evic_ids, size_t k_size) {
    mutex_lock l(mu_);
    size_t true_size = 0;
    LRUNode* evic_node = tail->pre;
    LRUNode* rm_node = evic_node;
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

  size_t get_cached_ids(K* cached_ids, size_t k_size, int64* cached_versions,
                        int64* cached_freqs) override {
    mutex_lock l(mu_);
    LRUNode* it = head->next;
    size_t i;
    for (i = 0; i < k_size && it != tail; i++, it = it->next) {
      cached_ids[i] = it->id;
    }
    return i;
  }

  void update(const K* batch_ids, size_t batch_size, bool use_locking = true) {
    mutex temp_mu;
    auto lock = BatchCache<K>::maybe_lock_cache(mu_, temp_mu, use_locking);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      typename std::map<K, LRUNode*>::iterator it = mp.find(id);
      if (it != mp.end()) {
        LRUNode* node = it->second;
        node->pre->next = node->next;
        node->next->pre = node->pre;
        head->next->pre = node;
        node->next = head->next;
        head->next = node;
        node->pre = head;
        BatchCache<K>::size_data_[0].num_hit_++;
      } else {
        LRUNode* newNode = new LRUNode(id);
        head->next->pre = newNode;
        newNode->next = head->next;
        head->next = newNode;
        newNode->pre = head;
        mp[id] = newNode;
        BatchCache<K>::size_data_[0].num_miss_++;
      }
    }
  }

  void update(const K* batch_ids, size_t batch_size, const int64* batch_version,
              const int64* batch_freqs, bool use_locking = true) override {
    // TODO: add to rank accroding to the version of ids
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
          LRUNode* node = it_cache->second;
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
        LOG(FATAL) << "The id should be prefetched before being used.";
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
  size_t capacity_;
  std::map<K, LRUNode*> mp;
  std::unordered_map<K, PrefetchNode<K>*> prefetch_id_table;
  mutex mu_;
};


template <typename K>
class ShardedLRUCache: public BatchCache<K> {
 public:
  ShardedLRUCache(const std::string& name, size_t capacity, const unsigned int num_threads = 0): name_(name), capacity_(capacity) {
    int shard_shift = (sizeof(num_threads) * 8 - 1) - __builtin_clz(num_threads);
    if (shard_shift < 0) {
      LOG(FATAL) << "Invalid argument shard_shift=" << shard_shift << " when constructing ShardedLRUCache for " << name;
    }
    int shard_num = 1 << shard_shift;
    LOG(INFO) << "Sharded LRU for " << name << " with " << shard_num << " shards";
    shard_mask_ = shard_num - 1;
    shards_.resize(shard_num);
    BatchCache<K>::size_data_.resize(shard_num);
    for (auto& shard : shards_) {
      shard.reset(new LRUShard());
      LRUNode *head = new LRUNode(0);
      LRUNode *tail = new LRUNode(0);
      head->next = tail;
      tail->prev = head;
      shard->head = head;
      shard->tail = tail;
    }
  }

  ~ShardedLRUCache() override {
    LOG(INFO) << "Sharded LRU Cache for " << name_ << " Destroying: " <<  BatchCache<K>::DebugString();
    const size_t num_shards = shards_.size();
    for (size_t i = 0; i < num_shards; ++i) {
      LRUShard& shard = *shards_[i];
      mutex_lock l_prefetch(shard.prefetch_mu_);
      mutex_lock l_cache(shard.mu_);
      for (auto iter = shard.prefetch_id_table.begin(); iter != shard.prefetch_id_table.end(); ++iter) {
        delete iter->second;
      }
      shard.prefetch_id_table.clear();
      for (LRUNode *node = shard.head->next; node != shard.tail;) {
        LRUNode* rm_node = node;
        node = node->next;
        delete rm_node;
      }
      delete shard.head;
      shard.head = nullptr;
      delete shard.tail;
      shard.tail = nullptr;
      shard.size = 0;
    }
  }

  size_t size() override {
    size_t total_size = 0;
    for (const auto& shard : shards_) {
      total_size += shard->size;
    }
    return total_size;
  }

  size_t get_capacity() override { return capacity_; }

  void set_capacity(size_t new_capacity) override { capacity_ = new_capacity; }

  size_t get_evic_ids(K* evic_ids, size_t k_size) override {
    const size_t num_shards = shards_.size();
    size_t num_per_shard = k_size / num_shards;
    size_t remaining = k_size % num_shards;
    size_t true_size = 0;
    for (size_t i = 0; i < num_shards; ++i) {
      LRUShard& shard = *shards_[i];
      const size_t num_this_shard = num_per_shard + (i < remaining ? 1 : 0);
      size_t true_num = 0;
      mutex_lock l(shard.mu_);
      LRUNode *evic_node = shard.tail->prev;
      LRUNode *rm_node = evic_node;
      for (size_t j = 0; j < num_this_shard && evic_node != shard.head; ++j) {
        evic_ids[true_size] = evic_node->id;
        rm_node = evic_node;
        evic_node = evic_node->prev;
        shard.mp.erase(rm_node->id);
        delete rm_node;
        true_size++;
        true_num++;
      }
      evic_node->next = shard.tail;
      shard.tail->prev = evic_node;
      shard.size -= true_num;
    }
    return true_size;
  }

  size_t get_cached_ids(K *cached_ids, size_t k_size, int64* cached_versions, int64* cached_freqs) override {
    const size_t num_shards = shards_.size();
    size_t num_per_shard = k_size / num_shards;
    size_t remaining = k_size % num_shards;
    size_t true_size = 0;
    for (size_t i = 0; i < num_shards; ++i) {
      LRUShard& shard = *shards_[i];
      const size_t num_this_shard = num_per_shard + (i < remaining ? 1 : 0);
      mutex_lock l(shard.mu_);
      LRUNode *it = shard.head->next;
      for (size_t j = 0; j < num_this_shard && it != shard.tail; ++j, it = it->next) {
        cached_ids[true_size++] = it->id; 
      }
    }
    return true_size;
  }

  void update(const K* batch_ids, size_t batch_size, bool use_locking = true) override {
    mutex temp_mu;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      const size_t shard_idx = id & shard_mask_;
      LRUShard& shard = *shards_[shard_idx];
      auto lock = BatchCache<K>::maybe_lock_cache(shard.mu_, temp_mu, use_locking);
      typename std::map<K, LRUNode*>::iterator it = shard.mp.find(id);
      if (it != shard.mp.end()) {
        LRUNode* node = it->second;
        node->prev->next = node->next;
        node->next->prev = node->prev;
        shard.head->next->prev = node;
        node->next = shard.head->next;
        shard.head->next = node;
        node->prev = shard.head;
        BatchCache<K>::size_data_[shard_idx].num_hit_++;
      } else {
        LRUNode* newNode = new LRUNode(id);
        shard.head->next->prev = newNode;
        newNode->next = shard.head->next;
        shard.head->next = newNode;
        newNode->prev = shard.head;
        shard.mp[id] = newNode;
        shard.size++;
        BatchCache<K>::size_data_[shard_idx].num_miss_++;
      }
    }
  }

  void update(const K* batch_ids, size_t batch_size, const int64 *batch_version,
              const int64* batch_freqs, bool use_locking = true) override {
    update(batch_ids, batch_size, use_locking);
  }

  void add_to_prefetch_list(const K* batch_ids, const size_t batch_size) override {
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      const size_t shard_idx = id & shard_mask_;
      LRUShard& shard = *shards_[shard_idx];
      mutex_lock l(shard.prefetch_mu_);
      auto it_prefetch = shard.prefetch_id_table.find(id);
      if (it_prefetch == shard.prefetch_id_table.end()) {
        mutex_lock ll(shard.mu_);
        auto it_cache = shard.mp.find(id);
        if (it_cache != shard.mp.end()) {
          LRUNode *node = it_cache->second;
          node->prev->next = node->next;
          node->next->prev = node->prev;
          delete node;
          shard.mp.erase(id);
        }
        
        //unlock
        (mutex_lock)(std::move(ll));
        shard.prefetch_id_table[id] = new PrefetchNode<K>(id);
      } else {
        it_prefetch->second->Ref();
      }
    }
  }

  void add_to_cache(const K* batch_ids, const size_t batch_size) override {
    std::vector<K> ids_to_cache(batch_size);
    int64 nums_to_cache = 0;
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      const size_t shard_idx = id & shard_mask_;
      LRUShard& shard = *shards_[shard_idx];
      mutex_lock l(shard.prefetch_mu_);
      auto it_prefetch = shard.prefetch_id_table.find(id);
      if (it_prefetch == shard.prefetch_id_table.end()) {
        LOG(FATAL) << "The id should be prefetched before being used.";
      }
      it_prefetch->second->UnRef();
      if (it_prefetch->second->ref_count() == 0) {
        delete it_prefetch->second;
        shard.prefetch_id_table.erase(id);
        ids_to_cache[nums_to_cache++] = id;
      }
    }
    update(ids_to_cache.data(), nums_to_cache, true);
  }



 protected:
 private:
  class LRUNode {
   public:
    K id;
    LRUNode *prev, *next;

    LRUNode(const K& id) : id(id), prev(nullptr), next(nullptr) {}

    LRUNode(const K& id, const LRUNode* prev, const LRUNode* next)
        : id(id), prev(prev), next(next) {}
  };

  class LRUShard {
   public:
    LRUNode *head = nullptr, *tail = nullptr;
    std::map<K, LRUNode*> mp;
    std::unordered_map<K, PrefetchNode<K>*> prefetch_id_table;
    mutex mu_;
    mutex prefetch_mu_;
    uint64 size = 0;
  };

  std::vector<std::unique_ptr<LRUShard>> shards_;
  uint64 shard_mask_;
  size_t capacity_;

  std::string name_;
};

template <class K>
class LFUCache : public BatchCache<K> {
 public:
  LFUCache(size_t capacity) : capacity_(capacity) {
    min_freq = std::numeric_limits<size_t>::max();
    max_freq = 0;
    freq_table.emplace_back(
        std::pair<std::list<LFUNode>*, int64>(new std::list<LFUNode>, 0));
    BatchCache<K>::size_data_.resize(1);
  }

  size_t get_capacity() override { return capacity_; }

  void set_capacity(size_t new_capacity) override {}

  size_t size() override {
    mutex_lock l(mu_);
    return key_table.size();
  }

  size_t get_cached_ids(K* cached_ids, size_t k_size, int64* cached_versions,
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
        } while (freq_table[curr_freq - 1].second == 0 &&
                 curr_freq >= min_freq);
        if (curr_freq >= min_freq) {
          it = freq_table[curr_freq - 1].first->begin();
        }
      }
    }
    return i;
  }

  size_t get_evic_ids(K* evic_ids, size_t k_size) {
    mutex_lock l(mu_);
    size_t true_size = 0;
    size_t st_freq = min_freq;
    // for (int i = st_freq - 1; i < freq_table.size(); i++) {
    //   LOG(INFO) << "Len " << i;
    //   for (auto it : *freq_table[i].first) {
    //     std::cout << it.key << ", " << it.freq << "->";
    //   }
    //   std::cout << std::endl;
    // }
    for (size_t i = 0; i < k_size && key_table.size() > 0; ++i) {
      auto rm_it = freq_table[st_freq - 1].first->back();
      key_table.erase(rm_it.key);
      evic_ids[i] = rm_it.key;
      // LOG(INFO) << "Evict " << rm_it.key;
      ++true_size;
      freq_table[st_freq - 1].first->pop_back();
      freq_table[st_freq - 1].second--;
      if (freq_table[st_freq - 1].second == 0) {
        ++st_freq;
        while (st_freq <= max_freq) {
          if (freq_table[st_freq - 1].second == 0) {
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

  void update(const K* batch_ids, size_t batch_size, bool use_locking = true) {
    mutex temp_mu;
    auto lock = BatchCache<K>::maybe_lock_cache(mu_, temp_mu, use_locking);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      // LOG(INFO) << "Insert " << id;
      auto it = key_table.find(id);
      if (it == key_table.end()) {
        freq_table[0].first->emplace_front(LFUNode(id, 1));
        freq_table[0].second++;
        key_table[id] = freq_table[0].first->begin();
        min_freq = 1;
        max_freq = std::max(max_freq, min_freq);
        BatchCache<K>::size_data_[0].num_miss_++;
      } else {
        typename std::list<LFUNode>::iterator node = it->second;
        size_t freq = node->freq;
        freq_table[freq - 1].first->erase(node);
        freq_table[freq - 1].second--;
        if (freq_table[freq - 1].second == 0) {
          if (min_freq == freq) min_freq += 1;
        }
        if (freq == freq_table.size()) {
          freq_table.emplace_back(
              std::pair<std::list<LFUNode>*, int64>(new std::list<LFUNode>, 0));
        }
        max_freq = std::max(max_freq, freq + 1);
        freq_table[freq].first->emplace_front(LFUNode(id, freq + 1));
        freq_table[freq].second++;
        key_table[id] = freq_table[freq].first->begin();
        BatchCache<K>::size_data_[0].num_hit_++;
      }
    }
  }

  void update(const K* batch_ids, const size_t batch_size,
              const int64* batch_versions, const int64* batch_freqs,
              bool use_locking = true) override {
    mutex temp_mu;
    auto lock = BatchCache<K>::maybe_lock_cache(mu_, temp_mu, use_locking);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      // LOG(INFO) << "Insert " << id;
      auto it = key_table.find(id);
      size_t freq = batch_freqs[i];
      if (it == key_table.end()) {
        if (freq < min_freq) {
          min_freq = freq;
        }

        if (freq > max_freq) {
          max_freq = freq;
          int64 prev_size = freq_table.size();
          if (max_freq > prev_size) {
            freq_table.resize(
                max_freq, std::pair<std::list<LFUNode>*, int64>(nullptr, 0));
            for (int64 j = prev_size; j < max_freq; j++) {
              freq_table[j].first = new std::list<LFUNode>;
            }
          }
        }
        freq_table[freq - 1].first->emplace_front(LFUNode(id, freq));
        freq_table[freq - 1].second++;
        key_table[id] = freq_table[freq - 1].first->begin();
        BatchCache<K>::size_data_[0].num_miss_++;
      } else {
        typename std::list<LFUNode>::iterator node = it->second;
        size_t last_freq = node->freq;
        size_t curr_freq = last_freq + freq;
        freq_table[last_freq - 1].first->erase(node);
        freq_table[last_freq - 1].second--;

        if (curr_freq > max_freq) {
          max_freq = curr_freq;
          freq_table.resize(max_freq, std::pair<std::list<LFUNode>*, int64>(
                                          new std::list<LFUNode>, 0));
        }

        if (freq_table[last_freq - 1].second == 0) {
          if (min_freq == last_freq) {
            update_min_freq();
          }
        }

        freq_table[curr_freq - 1].first->emplace_front(LFUNode(id, curr_freq));
        freq_table[curr_freq - 1].second++;
        key_table[id] = freq_table[curr_freq - 1].first->begin();
        BatchCache<K>::size_data_[0].num_hit_++;
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
  void reset_min_and_max_freq() {
    min_freq = std::numeric_limits<size_t>::max();
    max_freq = 0;
  }

  void update_min_freq() {
    size_t i;
    for (i = min_freq + 1; i <= max_freq; i++) {
      if (freq_table[i - 1].second != 0) {
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
    for (i = max_freq - 1; i >= min_freq; i--) {
      if (freq_table[i - 1].second != 0) {
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
  size_t capacity_;
  std::vector<std::pair<std::list<LFUNode>*, int64>> freq_table;
  std::unordered_map<K, typename std::list<LFUNode>::iterator> key_table;
  std::unordered_map<K, PrefetchLFUNode<K>*> prefetch_id_table;
  mutex mu_;
};

}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
