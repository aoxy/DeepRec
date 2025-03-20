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

#include <string>

#include "tensorflow/core/framework/embedding/cache.h"
#include "tensorflow/core/framework/embedding/block_lock_lru_cache.h"
#include "tensorflow/core/framework/embedding/block_lock_lfu_cache.h"
#include "tensorflow/core/framework/embedding/config.pb.h"
#include "tensorflow/core/framework/embedding/cache_manager.h"
#include "tensorflow/core/framework/embedding/cache_profiler.h"
#include "tensorflow/core/framework/embedding/profiled_cache.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {
template<typename K, typename C>
class Creator {
  public:
    [[noreturn]] static C* Create(const std::string& name, int64 capacity, int num_threads, int way) {
      LOG(FATAL) << "Not implemented: " << typeid(Create).name();
    }
};

template<typename K>
class Creator<K, LRUCache<K>> {
  public:
    static LRUCache<K>* Create(const std::string& name, int64 capacity, int num_threads, int way) {
      return new LRUCache<K>(name);
    }
};

template<typename K>
class Creator<K, LFUCache<K>> {
  public:
    static LFUCache<K>* Create(const std::string& name, int64 capacity, int num_threads, int way) {
      return new LFUCache<K>(name);
    }
};

template<typename K>
class Creator<K, BlockLockLRUCache<K>> {
  public:
    static BlockLockLRUCache<K>* Create(const std::string& name, int64 capacity, int num_threads, int way) {
      return new BlockLockLRUCache<K>(name, capacity, way, num_threads);
    }
};

template<typename K>
class Creator<K, BlockLockLFUCache<K>> {
  public:
    static BlockLockLFUCache<K>* Create(const std::string& name, int64 capacity, int num_threads, int way) {
      return new BlockLockLFUCache<K>(name, capacity, way, num_threads);
    }
};


class CacheFactory {
  public:
     template<typename K>
     static BatchCache<K>* Create(CacheStrategy cache_strategy, ProfilingStrategy profiling_strategy, std::string name, int64 capacity, int num_threads, TunableCache *tunable_cache = nullptr) {
       switch (cache_strategy) {
          case CacheStrategy::LRU:
            LOG(INFO) << " Use Storage::LRU in multi-tier EmbeddingVariable " << name;
            return CreateCache<K, LRUCache<K>>(profiling_strategy, name, capacity, num_threads, 0, tunable_cache);
          case CacheStrategy::LFU:
            LOG(INFO) << " Use Storage::LFU in multi-tier EmbeddingVariable " << name;
            return CreateCache<K, LFUCache<K>>(profiling_strategy, name, capacity, num_threads, 0, tunable_cache);
          case CacheStrategy::B8LRU:
            LOG(INFO) << " Use Strategy::B8LRU in multi-tier EmbeddingVariable "  << name;
            return CreateCache<K, BlockLockLRUCache<K>>(profiling_strategy, name, capacity, num_threads, 8, tunable_cache);
          case CacheStrategy::B8LFU:
            LOG(INFO) << " Use Strategy::B8LFU in multi-tier EmbeddingVariable " << name;
            return CreateCache<K, BlockLockLFUCache<K>>(profiling_strategy, name, capacity, num_threads, 8, tunable_cache);
          case CacheStrategy::B16LRU:
            LOG(INFO) << " Use Strategy::B16LRU in multi-tier EmbeddingVariable " << name;
            return CreateCache<K, BlockLockLRUCache<K>>(profiling_strategy, name, capacity, num_threads, 16, tunable_cache);
          case CacheStrategy::B16LFU:
            LOG(INFO) << " Use Strategy::B16LFU in multi-tier EmbeddingVariable " << name;
            return CreateCache<K, BlockLockLFUCache<K>>(profiling_strategy, name, capacity, num_threads, 16, tunable_cache);
          case CacheStrategy::B32LRU:
            LOG(INFO) << " Use Strategy::B32LRU in multi-tier EmbeddingVariable " << name;
            return CreateCache<K, BlockLockLRUCache<K>>(profiling_strategy, name, capacity, num_threads, 32, tunable_cache);
          case CacheStrategy::B32LFU:
            LOG(INFO) << " Use Strategy::B32LFU in multi-tier EmbeddingVariable " << name;
            //  return CreateCache<K, BlockLockLFUCache<K>>(profiling_strategy, name, capacity, num_threads, 32, tunable_cache);
            return new BlockLockLFUCache<K>(name, capacity, 32, num_threads);
          case CacheStrategy::B48LRU:
            LOG(INFO) << " Use Strategy::B48LRU in multi-tier EmbeddingVariable " << name;
            return CreateCache<K, BlockLockLRUCache<K>>(profiling_strategy, name, capacity, num_threads, 48, tunable_cache);
          case CacheStrategy::B48LFU:
            LOG(INFO) << " Use Strategy::B48LFU in multi-tier EmbeddingVariable " << name;
            return CreateCache<K, BlockLockLFUCache<K>>(profiling_strategy, name, capacity, num_threads, 48, tunable_cache);
          default:
            LOG(FATAL) << " Invalid Cache strategy";
            return nullptr;
       }
   }
 
   template<typename K, typename Base>
   static Base* CreateCache(const ProfilingStrategy profiling_strategy, const std::string& name, int64 capacity, int num_threads, int way, TunableCache *tunable_cache = nullptr) {
     switch (profiling_strategy) {
        case ProfilingStrategy::NONE:
          LOG(INFO) << " Use ProfilingStrategy::NONE in multi-tier EmbeddingVariable";
          return Creator<K, Base>::Create(name, capacity, num_threads, way);
        case ProfilingStrategy::AET: {
          LOG(INFO) << " Use ProfilingStrategy::AET in multi-tier EmbeddingVariable";
          size_t bucket_size;
          size_t max_reuse_dist;
          uint64_t sampling_interval;
          ReadInt64FromEnvVar("CACHE_PROFILER_BUCKET_SIZE", 10, reinterpret_cast<int64 *>(&bucket_size));
          ReadInt64FromEnvVar("CACHE_PROFILER_MAX_REUSE_DIST", 100000, reinterpret_cast<int64 *>(&max_reuse_dist));
          ReadInt64FromEnvVar("CACHE_PROFILER_SAMPLING_INTERVAL", 1, reinterpret_cast<int64 *>(&sampling_interval));
          ProfiledCacheProxy<K, Base> *proxy_cache = new ProfiledCacheProxy<K, Base>(name, bucket_size, max_reuse_dist, sampling_interval, capacity, num_threads, way, tunable_cache);
          if (tunable_cache != nullptr) {
            CacheManager::GetInstance().RegisterCache(*proxy_cache->GetProfiler());
          }
          return proxy_cache;
        }
     }
   }
 };

 } // embedding
 } // tensorflow
 
 #endif // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_CACHE_FACTORY_H_
