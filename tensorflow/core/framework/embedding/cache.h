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
class LFUCache : public BatchCache<K> {
 public:
  LFUCache() {
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
class AutoLRFUCache : public BatchCache<K> {
 public:
  AutoLFUCache() {
    global_step = 0;
    key_table.clear();
    freq_table.clear();
    state = F0;
    mode = FREQ_MODE;
    span_last_check = 0;
    counter_switch = 0;
    prev_hit_rate = -100;
    counter_replacement = 0;
    num_replacement = 1;
    HitSpan = 1000;
  }

  void rebuild() {
    if (mode == STEP_MODE) return;
    std::unordered_map<K, BaseNode*> new_table;
    for (auto it = key_table.begin(); it != key_table.end(); it++) {
      BaseNode* node = (BaseNode*)(*(it->second));
      node->updateLFU(global_step, mode);
      new_table[node->get_index()] = node;
    }
    freq_table.clear();
    key_table.clear();
    for (auto it = new_table.begin(); it != new_table.end(); it++) {
      BaseNode* node = (BaseNode*)(it->second);
      freq_table[node->get_index()].emplace_front(node);
    }
    for (auto list = freq_table.begin(); list != freq_table.end(); list++)
      for (auto it = list->second.begin(); it != list->second.end(); it++)
        key_table[(*it)->key] = it;
  }

  int get_hit_rate_len_100000(int len) {
    int total = hit_recent.size();
    if (total <= 0) return 0.0;
    int hit = 0;
    for (auto it = hit_recent.begin(); it != hit_recent.end(); it++)
      hit += (len-- > 0) && (*it);
    if (hit == 0) return 0.0;
    return int(hit * 100000 / total);
  }

  int get_hit_rate100000() {
    int total = hit_recent.size();
    if (total <= 0) return 0.0;
    int hit = 0;
    for (auto it = hit_recent.begin(); it != hit_recent.end(); it++) hit += *it;
    if (hit == 0) return 0.0;
    return int(hit * 100000 / total);
  }

  void mode_switch() {
    if (mode == STEP_MODE) {
      mode = FREQ_MODE;
      rebuild();  // 根据保存的频次信息，重新构建链表
    } else {
      mode = STEP_MODE;
    }
  }

  void auto_switch() {
    if (state == S3 && counter_replacement < HitSpan)
      return;
    else if (counter_replacement <
             num_replacement *
                 (SingleCache<K, V>::capacity() + HitSpan))  // TODO:
      return;

    counter_replacement = 0;
    int curr_hit_rate = get_hit_rate100000();
    if (state == F0) {
      if (prev_hit_rate >= 0 &&
          curr_hit_rate <
              prev_hit_rate * 0.85) {  // (prev_hit_rate - curr_hit_rate) /
                                       // prev_hit_rate > 0.15
        mode_switch();
        state = S1;
      }
    } else if (state == S1) {
      if (curr_hit_rate <= 1.15 * prev_hit_rate) {  //没有显著提高
        mode_switch();
        state = F0;
      } else {
        state = S2;
      }
    } else if (state == S2) {
      mode_switch();
      state = S3;
    } else if (state == S3) {
      curr_hit_rate = get_hit_rate_len_100000(HitSpan);
      if (curr_hit_rate > prev_hit_rate) {
        state = F0;
        num_replacement = 1;
      } else {
        mode_switch();
        state = S2;
        num_replacement = num_replacement * 2;
      }
    }
    if (state != S3) prev_hit_rate = curr_hit_rate;
  }

  void add_to_rank(const K* batch_ids, size_t batch_size) {
    mutex_lock l(mu_);
    for (size_t i = 0; i < batch_size; ++i) {
      K id = batch_ids[i];
      auto_switch();
      global_step++;
      if (hit_recent.size() > 6000) {
        hit_recent.pop_back();
      }
      auto it = key_table.find(id);
      if (it != key_table.end()) {
        hit_recent.push_front(true);
        BaseNode* node = (BaseNode*)(*(it->second));
        short index = node->get_index();
        freq_table[index].erase(it->second);
        index = node->updateLFU(global_step, mode);
        freq_table[index].emplace_front(node);
        key_table[id] = freq_table[index].begin();
      } else {
        hit_recent.push_front(false);
        counter_replacement++;
        BaseNode* node = new BaseNode(id, global_step);
        short index = node->updateLFU(global_step, mode);
        auto& freq_ls = freq_table[index];
        freq_ls.emplace_front(node);
        key_table[id] = freq_ls.begin();
      }
    }
  }

  size_t get_evic_ids(K* evic_ids, size_t k_size) {
    mutex_lock l(mu_);
    size_t true_size = 0;
    short min_freq = BaseNode::MinCount;
    for (size_t i = 0; i < k_size && key_table.size() > 0; ++i) {
      for (; min_freq <= BaseNode::MaxCount; min_freq++)
        if (freq_table[min_freq].size() > 0) break;

      auto& freq_ls = freq_table[min_freq];
      auto rm_it = freq_ls.back();

      freq_ls.pop_back();
      K evic_id = rm_it->key;
      key_table.erase(evic_id);
      delete rm_it;

      evic_ids[i] = evic_id;
      ++true_size;
    }
    return true_size;
  }

  size_t size() {
    mutex_lock l(mu_);
    return key_table.size();
  }

 private:
  enum Mode { STEP_MODE, FREQ_MODE };
  enum State { F0, S1, S2, S3 };
  class BaseNode {
   public:
    enum { PrivilegeStep = 255, MinCount = 0, MaxCount = 255 };
    K key;
    short count;
    short index;
    unsigned last;
    BaseNode(K key, unsigned step)
        : key(key), count(LFU_INIT_VAL), index(LFU_INIT_VAL), last(step) {}
    BaseNode(BaseNode* that)
        : key(that->key),
          count(that->count),
          index(that->index),
          last(that->last) {}

    short decrFuncLinear(unsigned long period, short count) {
      unsigned long num = period * MaxCount / TIMES_TO_MAX;
      if (num > 0) count = std::max(0, int(count - num));
      return count;
    }

    short decrFuncLog(unsigned long period, short count) {
      if (count > MinCount &&
          rand() * TIMES_TO_MAX < RAND_MAX * (period - PrivilegeStep))
        count--;  //   rand()/RAND_MAX < (period-PrivilegeStep)/timem
      return count;
    }
    unsigned long LFUTimeElapsed(unsigned long now) {
      if (now >= last) return now - last;
      return 16777215 - (last - now);
    }
    short LFULogIncr(short count) {
      if (count < MaxCount &&
          rand() * (count - LFU_INIT_VAL) * LFU_LOG_FACTOR < RAND_MAX)
        count++;  // rand()/RAND_MAX < 1/( (count - LFU_INIT_VAL) *
                  // LFU_LOG_FACTOR )
      return count;
    }
    short LFUDecrAndReturn(unsigned long step) {
      unsigned long period = this->LFUTimeElapsed(step);
      return this->decrFuncLog(period, count);
    }
    short updateLFU(unsigned long step, Mode mode) {
      step = MOD_STEP(step);
      this->count = this->LFUDecrAndReturn(step);  //文档结果没有这个
      this->count = this->LFULogIncr(
          this->count);  // 按概率自增1，概率约为 1/(LFU_LOG_FACTOR * counter)
      this->last = step;
      this->index = (mode == FREQ_MODE ? this->count : MaxCount);
      return this->index;
    }
    inline short get_index() { return this->index; }
  };

  time_t global_step;
  std::unordered_map<K, typename std::list<BaseNode*>::iterator> key_table;
  std::unordered_map<short, typename std::list<BaseNode*>> freq_table;
  std::deque<bool> hit_recent;
  short HitSpan;  // 每span次查看
  size_t span_last_check;
  size_t counter_switch;
  size_t counter_replacement;
  size_t counter_visit;
  int num_replacement;
  int prev_hit_rate;
  Mode mode;
  State state;
};

} // embedding
} // tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_H_
