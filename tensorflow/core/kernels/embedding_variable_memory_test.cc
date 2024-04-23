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
#include "tensorflow/core/kernels/embedding_variable_test.h"

namespace tensorflow {
namespace embedding {
float PerfMemory(Tensor& default_value,
                const std::vector<int64>& id_list,
                int value_size, int64 default_value_dim,
                int64 filter_freq = 0, int64 steps_to_live = 0,
                int64 record_freq = false, bool multi_tier = false) {
  EmbeddingVar<int64,float>* ev;
  if (multi_tier) {
    ev = CreateMultiTierEmbeddingVar(value_size, default_value, default_value_dim,
                            filter_freq, steps_to_live, -1.0,
                            CacheStrategy::B8LFU, 8,
                            embedding::StorageType::DRAM_SSDHASH,
                            {100*1024*1024, 1024, 1024, 1024},
                            record_freq);
  } else {
    ev = CreateEmbeddingVar(value_size, default_value, default_value_dim,
                            filter_freq, steps_to_live, -1.0,
                            embedding::StorageType::DRAM,
                            {1024, 1024, 1024, 1024},
                            record_freq);
  }
  void* value_ptr = nullptr;
  bool is_filter = false;
  double start_mem, end_mem;
  start_mem = getResident() * getpagesize();
  for (int i = 0; i < id_list.size(); i++) {
    ev->LookupOrCreateKey(id_list[i], &value_ptr, &is_filter, false);
    if (is_filter)
      ev->flat(value_ptr);
  }
  end_mem = getResident() * getpagesize();
  double used_mb = (end_mem - start_mem) / 1000000;
  LOG(INFO) << "[TestMemory] Use Memory: " << used_mb << "MB";
  // if (multi_tier) {
  //   static_cast<embedding::MultiTierStorage<int64, float>*>(ev->storage())->DeleteFromEvictionManager();
  //   // ev->Unref();
  // }
  return used_mb;
}

TEST(EmbeddingVariableMemoryTest, TestMemory) {
  int value_size = 32;
  int64 default_value_dim = 4096;
  int filter_freq = 2;
  Tensor default_value(
      DT_FLOAT, TensorShape({default_value_dim, value_size}));
  auto default_value_matrix = default_value.matrix<float>();
	for (int i = 0; i < default_value_dim; i++) {
		for (int j = 0 ; j < value_size; j++) {
			default_value_matrix(i, j) = i * value_size + j;
		}
	}

  int num_of_ids = 1000000;
  std::vector<int64> id_list(num_of_ids);
  for (int i = 0; i < num_of_ids; i++) {
    id_list[i] = i;
  }
  float used_mb = PerfMemory(default_value, id_list,
                             value_size, default_value_dim);
  float theoritical_mb =
      50 + num_of_ids * (value_size * sizeof(float)) / 1000000;
  EXPECT_TRUE((used_mb > theoritical_mb * 0.99) &&
              (used_mb < theoritical_mb * 1.01));

  for (int i = 0; i < num_of_ids / 2; i++) {
    id_list.emplace_back(i);
  }
  used_mb = PerfMemory(default_value, id_list, value_size,
                       default_value_dim, filter_freq);
  theoritical_mb =
      50 + num_of_ids * (8 + value_size * sizeof(float) / 2
                         + 4/*memory for ids_list*/) / 1000000;
  EXPECT_TRUE((used_mb > theoritical_mb * 0.99) &&
              (used_mb < theoritical_mb * 1.02));
}

TEST(EmbeddingVariableMemoryTest, TestMemoryDRAMSSDHASH) {
  setenv("TF_CACHE_RECORD_HITRATE", "true", 1);
  int value_size = 32;
  int64 default_value_dim = 4096;
  int filter_freq = 2;
  Tensor default_value(
      DT_FLOAT, TensorShape({default_value_dim, value_size}));
  auto default_value_matrix = default_value.matrix<float>();
	for (int i = 0; i < default_value_dim; i++) {
		for (int j = 0 ; j < value_size; j++) {
			default_value_matrix(i, j) = i * value_size + j;
		}
	}

  int num_of_ids = 1000000;
  std::vector<int64> id_list(num_of_ids);
  for (int i = 0; i < num_of_ids; i++) {
    id_list[i] = i;
  }
  float used_mb = PerfMemory(default_value, id_list, value_size,
                              default_value_dim, 0, 0, false,
                              /*multi_tier=*/true);
  float theoritical_mb =
      50 + num_of_ids * (value_size * sizeof(float)) / 1000000;
  // EXPECT_TRUE((used_mb > theoritical_mb * 0.99) &&
  LOG(INFO) << "[TestMemory] Theoritical Memory: " << theoritical_mb << "MB";

  for (int i = 0; i < num_of_ids / 2; i++) {
    id_list.emplace_back(i);
  }
  used_mb = PerfMemory(default_value, id_list, value_size,
                        default_value_dim, 0, 0, false,
                        /*multi_tier=*/true);
  theoritical_mb =
      50 + num_of_ids * (8 + value_size * sizeof(float) / 2
                         + 4/*memory for ids_list*/) / 1000000;
  // EXPECT_TRUE((used_mb > theoritical_mb * 0.99) &&
  //             (used_mb < theoritical_mb * 1.02));
  LOG(INFO) << "[TestMemory] Theoritical Memory: " << theoritical_mb << "MB";
  unsetenv("TF_CACHE_RECORD_HITRATE");
}
} //namespace embedding
} //namespace tensorflow
