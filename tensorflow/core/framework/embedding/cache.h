#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
#include <iostream>
#include <map>
#include <unordered_map>
#include <set>
#include <list>
#include <limits>
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace embedding {

template <class K>
class BatchCache {
 public:
  BatchCache() {}
  virtual ~BatchCache() {}
  void add_to_rank(const Tensor& t) {
    add_to_rank((K*)t.data(), t.NumElements());
  }
  virtual size_t get_evic_ids(K* evic_ids, size_t k_size) = 0;
  virtual void add_to_rank(const K* batch_ids, size_t batch_size) = 0;
  virtual void add_to_rank(const K* batch_ids, size_t batch_size,
                           const int64* batch_versions,
                           const int64* batch_freqs) = 0;
  virtual size_t size() = 0;
  virtual void reset_status() {
     num_hit = 0;
     num_miss = 0;
  }
  std::string DebugString() {
    float hit_rate = 0.0;
    if (num_hit > 0 || num_miss > 0) {
      hit_rate = num_hit * 100.0 / (num_hit + num_miss);
    }
    return strings::StrCat("HitRate = " , hit_rate,
                          " %, visit_count = ", num_hit + num_miss,
                           ", hit_count = ", num_hit);
  }

 protected:
  int64 num_hit;
  int64 num_miss;
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
    BatchCache<K>::num_hit = 0;
    BatchCache<K>::num_miss = 0;
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

  void add_to_rank(const K* batch_ids, size_t batch_size) {
    mutex_lock l(mu_);
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
        BatchCache<K>::num_hit++;
      } else {
        LRUNode *newNode = new LRUNode(id);
        head->next->pre = newNode;
        newNode->next = head->next;
        head->next = newNode;
        newNode->pre = head;
        mp[id] = newNode;
        BatchCache<K>::num_miss++;
      }
    }
  }

  void add_to_rank(const K* batch_ids, size_t batch_size,
                    const int64* batch_version,
                    const int64* batch_freqs) {
    //TODO: add to rank accroding to the version of ids
    add_to_rank(batch_ids, batch_size);
  }
 private:
  class LRUNode {
   public:
     K id;
     LRUNode *pre, *next;
     LRUNode(K id) : id(id), pre(nullptr), next(nullptr) {}
  };
  LRUNode *head, *tail;
  std::map<K, LRUNode *> mp;
  mutex mu_;
};

template <class K>
class OriginLFUCache : public BatchCache<K> {
 public:
  OriginLFUCache() {
    min_freq = std::numeric_limits<size_t>::max();
    max_freq = 0;
    freq_table.emplace_back(std::pair<std::list<LFUNode>*, int64>(
      new std::list<LFUNode>, 0));
    BatchCache<K>::num_hit = 0;
    BatchCache<K>::num_miss = 0;
  }

  size_t size() {
    mutex_lock l(mu_);
    return key_table.size();
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
            break;
          }
        }
      }
    }
    return true_size;
  }

  void add_to_rank(const K *batch_ids, size_t batch_size) {
    mutex_lock l(mu_);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it = key_table.find(id);
      if (it == key_table.end()) {
        freq_table[0].first->emplace_front(LFUNode(id, 1));
        freq_table[0].second++;
        key_table[id] = freq_table[0].first->begin();
        min_freq = 1;
        max_freq = std::max(max_freq, min_freq);
        BatchCache<K>::num_miss++;
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
        BatchCache<K>::num_hit++;
      }
    }
  }

  void add_to_rank(const K *batch_ids, const size_t batch_size,
                   const int64* batch_versions,
                   const int64* batch_freqs) {
    mutex_lock l(mu_);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it = key_table.find(id);
      size_t freq =  batch_freqs[i];
      if (it == key_table.end()) {
        if (freq < min_freq) {
          min_freq = freq;
        }

        if (freq >= max_freq) {
          max_freq = freq;
          freq_table.resize(max_freq, std::pair<std::list<LFUNode>*, int64>(
           new std::list<LFUNode>, 0));
        }
        freq_table[freq-1].first->emplace_front(LFUNode(id, freq));
        freq_table[freq-1].second++;
        key_table[id] = freq_table[freq-1].first->begin();
        BatchCache<K>::num_miss++;
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
            for (size_t j = last_freq + 1; j < max_freq; j++) {
              if(freq_table[j-1].second != 0) {
                min_freq = j;
              }
            }
          }
        }
       
        freq_table[curr_freq-1].first->emplace_front(LFUNode(id, curr_freq));
        freq_table[curr_freq-1].second++;
        key_table[id] = freq_table[curr_freq-1].first->begin();
        BatchCache<K>::num_hit++;
      }
    }
  }


 private:
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
  mutex mu_;
};


template <class K>
class LFUNode {
 public:
  K key;
  size_t freq;
  LFUNode(K key, unsigned now) : key(key), freq(0) {}
  size_t GetIndex() { return freq; }
  size_t UpdateAndReturnIndex(unsigned now, bool lru_mode) { return ++freq; }
};

template <class K>
class AgingNode {
 public:
  static const uint8_t INIT_CNT = 5;
  static const uint8_t MIN_CNT = 1;
  static const uint8_t MAX_CNT = 255;
  static const unsigned INCR_FACTOR = 7;
  static const unsigned DECR_FACTOR = 10;
  // 32640 = 1 + 2 + ... + 255
  static const unsigned UNIT_STEP = 32640 * INCR_FACTOR;
  static const unsigned IGNORE_STEP = 0;

 public:
  K key;
  uint8_t count;
  uint8_t index;
  unsigned last;
  AgingNode(K key, unsigned now)
      : key(key), count(INIT_CNT), index(INIT_CNT), last(now) {}
  AgingNode(typename std::list<AgingNode>::iterator that)
      : key(that->key),
        count(that->count),
        index(that->index),
        last(that->last) {}
  void DecrByProb(unsigned period) {
    size_t ret1 = rand();
    size_t ret2 = RAND_MAX;
    ret1 *= UNIT_STEP * DECR_FACTOR;
    ret2 *= (period - IGNORE_STEP);
    if (count > 0 && ret1 < ret2) count--;
  }
  void DecrByPeriod(unsigned period) {
    unsigned decrease_value = period / UNIT_STEP;
    if (decrease_value + INIT_CNT > count)
      count = INIT_CNT;
    else
      count -= decrease_value;
  }
  void Decrease(unsigned now) {
    unsigned period = now >= last
                          ? now - last
                          : std::numeric_limits<size_t>::max() - last + now;
    DecrByPeriod(period);
  }
  void IncrByProb() {
    size_t ret = rand();
    ret *= (count - INIT_CNT) * INCR_FACTOR;
    if (count < 255 && ret < RAND_MAX) count++;
  }
  uint8_t UpdateAndReturnIndex(unsigned now, bool lru_mode) {
    Decrease(now);
    IncrByProb();
    if (lru_mode)
      index = MAX_CNT;
    else
      index = count;
    return index;
  }
  size_t GetIndex() { return index; }
};

template <class K, class Node>
class BaseLFUCache : public BatchCache<K> {
 public:
  using map_iter =
      typename std::unordered_map<K,
                                  typename std::list<Node>::iterator>::iterator;
  BaseLFUCache() {
    min_freq = std::numeric_limits<size_t>::max();
    max_freq = 0;
    freq_table.emplace_back(
        std::pair<std::list<Node>*, int64>(new std::list<Node>, 0));
    BatchCache<K>::num_hit = 0;
    BatchCache<K>::num_miss = 0;
  }

  size_t size() {
    mutex_lock l(mu_);
    return key_table.size();
  }

  size_t get_evic_ids(K* evic_ids, size_t k_size) {
    mutex_lock l(mu_);
    size_t true_size = 0;
    size_t st_freq = min_freq;
    for (size_t i = 0; i < k_size && key_table.size() > 0; ++i) {
      auto rm_it = freq_table[st_freq].first->back();
      key_table.erase(rm_it.key);
      evic_ids[i] = rm_it.key;
      ++true_size;
      freq_table[st_freq].first->pop_back();
      freq_table[st_freq].second--;
      if (freq_table[st_freq].second == 0) {
        ++st_freq;
        while (st_freq <= max_freq) {
          if (freq_table[st_freq].second == 0) {
            ++st_freq;
          } else {
            break;
          }
        }
      }
    }
    return true_size;
  }

  void AddNode(K id, unsigned now) {
    Node node(id, now);
    size_t index = node.GetIndex();
    freq_table[index].first->emplace_front(node);
    freq_table[index].second++;
    key_table[id] = freq_table[index].first->begin();
    min_freq = std::min(min_freq, index);
    max_freq = std::max(max_freq, index);
  }

  void UpdateNode(K id, map_iter it, unsigned now, bool lru_mode) {
    Node node = (Node)(*(it->second));
    size_t index = node.GetIndex();
    freq_table[index].first->erase(it->second);
    freq_table[index].second--;
    if (freq_table[index].second == 0) {
      if (min_freq == index) min_freq += 1;
    }
    index = node.UpdateAndReturnIndex(now, lru_mode);
    if (index == freq_table.size()) {
      freq_table.emplace_back(
          std::pair<std::list<Node>*, int64>(new std::list<Node>, 0));
    }
    max_freq = std::max(max_freq, index);
    min_freq = std::min(min_freq, index);
    freq_table[index].first->emplace_front(node);
    freq_table[index].second++;
    key_table[id] = freq_table[index].first->begin();
  }

  void add_to_rank(const K* batch_ids, size_t batch_size) {
    mutex_lock l(mu_);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto it = key_table.find(id);
      if (it == key_table.end()) {
        AddNode(id, 0);
        BatchCache<K>::num_miss++;
      } else {
        UpdateNode(id, it, 0, false);
        BatchCache<K>::num_hit++;
      }
    }
  }

  void add_to_rank(const K* batch_ids, size_t batch_size,
                   const int64* batch_version, const int64* batch_freqs) {
    // TODO: add to rank accroding to the version of ids
    add_to_rank(batch_ids, batch_size);
  }

 protected:
  size_t min_freq;
  size_t max_freq;
  std::vector<std::pair<std::list<Node>*, int64>> freq_table;
  std::unordered_map<K, typename std::list<Node>::iterator> key_table;
  mutex mu_;
};

template <class K>
class LFUCache : public BaseLFUCache<K, LFUNode<K>> {
 public:
  LFUCache() : BaseLFUCache<K, LFUNode<K>>() {}
};

template <class K>
class AgingLFUCache : public BaseLFUCache<K, AgingNode<K>> {
 public:
  AgingLFUCache() : BaseLFUCache<K, AgingNode<K>>() { 
    global_step = 0;
    for (size_t i = 0; i <= AgingNode<K>::INIT_CNT; ++i) {
      this->freq_table.emplace_back(std::pair<std::list<AgingNode<K>>*, int64>(
          new std::list<AgingNode<K>>, 0));
    }
  }

  void add_to_rank(const K* batch_ids, size_t batch_size) {
    mutex_lock l(this->mu_);
    for (size_t i = 0; i < batch_size; ++i) {
      if (global_step == std::numeric_limits<size_t>::max()) global_step = 0;
      global_step++;
      K id = batch_ids[i];
      auto it = this->key_table.find(id);
      if (it == this->key_table.end()) {
        this->AddNode(id, global_step);
        BatchCache<K>::num_miss++;
      } else {
        this->UpdateNode(id, it, global_step, false);
        BatchCache<K>::num_hit++;
      }
    }
  }

  void add_to_rank(const K* batch_ids, size_t batch_size,
                   const int64* batch_version, const int64* batch_freqs) {
    // TODO: add to rank accroding to the version of ids
    add_to_rank(batch_ids, batch_size);
  }

 protected:
  size_t global_step;
};

} // embedding
} // tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
