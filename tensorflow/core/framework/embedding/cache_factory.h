/* Copyright 2022 The DeepRec Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
======================================================================*/
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/block_lock_lru_cache.h"
#include "tensorflow/core/framework/embedding/block_lock_lfu_cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"

namespace tensorflow {
namespace embedding {
class CacheFactory {
 public:
  template<typename K>
  static BatchCache<K>* Create(CacheStrategy cache_strategy, std::string name, int64 capacity, unsigned num_threads) {
    switch (cache_strategy) {
      case CacheStrategy::LRU:
        LOG(INFO) << " Use Strategy::LRU in multi-tier EmbeddingVariable "
                << name;
        return new LRUCache<K>(capacity);
      case CacheStrategy::B4LRU:
        LOG(INFO) << " Use Strategy::B4LRU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLRUCache<K>(name, capacity, 4, num_threads);
      case CacheStrategy::B8LRU:
        LOG(INFO) << " Use Strategy::B8LRU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLRUCache<K>(name, capacity, 8, num_threads);
      case CacheStrategy::B16LRU:
        LOG(INFO) << " Use Strategy::B16LRU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLRUCache<K>(name, capacity, 16, num_threads);
      case CacheStrategy::B32LRU:
        LOG(INFO) << " Use Strategy::B32LRU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLRUCache<K>(name, capacity, 32, num_threads);
      case CacheStrategy::B48LRU:
        LOG(INFO) << " Use Strategy::B48LRU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLRUCache<K>(name, capacity, 48, num_threads);
      case CacheStrategy::B64LRU:
        LOG(INFO) << " Use Strategy::B64LRU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLRUCache<K>(name, capacity, 64, num_threads);
      case CacheStrategy::LFU:
        LOG(INFO) << " Use Strategy::LFU in multi-tier EmbeddingVariable "
                << name;
        return new LFUCache<K>(capacity);
      case CacheStrategy::B4LFU:
        LOG(INFO) << " Use Strategy::B4LFU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLFUCache<K>(name, capacity, 4, num_threads);
      case CacheStrategy::B8LFU:
        LOG(INFO) << " Use Strategy::B8LFU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLFUCache<K>(name, capacity, 8, num_threads);
      case CacheStrategy::B16LFU:
        LOG(INFO) << " Use Strategy::B16LFU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLFUCache<K>(name, capacity, 16, num_threads);
      case CacheStrategy::B32LFU:
        LOG(INFO) << " Use Strategy::B32LFU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLFUCache<K>(name, capacity, 32, num_threads);
      case CacheStrategy::B48LFU:
        LOG(INFO) << " Use Strategy::B48LFU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLFUCache<K>(name, capacity, 48, num_threads);
      case CacheStrategy::B64LFU:
        LOG(INFO) << " Use Strategy::B64LFU in multi-tier EmbeddingVariable "
                << name;
        return new BlockLockLFUCache<K>(name, capacity, 64, num_threads);
      case CacheStrategy::ShardedLRU:
        LOG(INFO) << " Use Strategy::ShardedLRU in multi-tier EmbeddingVariable "
                << name;
        return new ShardedLRUCache<K>(name, capacity, num_threads);
      default:
        LOG(INFO) << " Invalid Cache strategy, \
                       use LFU in multi-tier EmbeddingVariable "
                << name;
        return new LFUCache<K>(capacity);
    }
  }
};
} // embedding
} // tensorflow

#endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_
