/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_
#define TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_

#include "tensorflow/core/framework/allocator.h"
#include "tensorflow/core/framework/bounds_check.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/embedding_var.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/kernels/save_restore_tensor.h"
#include "tensorflow/core/lib/random/philox_random.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/lib/random/random_distributions.h"
#include "tensorflow/core/util/tensor_bundle/tensor_bundle.h"

#include "tensorflow/core/framework/hashmap.h"

namespace tensorflow {
namespace {
  const int kSavedPartitionNum = 1000;
}

template<class K, class T>
class EVKeyDumpIterator: public  DumpIterator<K, T> {
 public:
  EVKeyDumpIterator(HashMap<T, K>*& hash_map, std::vector<T>& key_list):hash_map_(hash_map), key_list_(key_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < key_list_.size();
  }

  T Next() {
    return key_list_[keys_idx_++];
  }

 private:
  int64 keys_idx_;
  HashMap<T, K>* hash_map_;
  std::vector<T>& key_list_;
};

template<class K, class T>
class EVValueDumpIterator: public  DumpIterator<K, T> {
 public:
  EVValueDumpIterator(HashMap<K, T>*& hash_map, std::vector<T* >& valueptr_list):hash_map_(hash_map), valueptr_list_(valueptr_list) {
    keys_idx_ = 0;
    col_idx_ = 0;
  }

  bool HasNext() const {
    if (keys_idx_ < valueptr_list_.size()) {
      if (keys_idx_ < valueptr_list_.size() - 1)
        return true;
      else
        return col_idx_ < hash_map_->ValueLen();
    } else
      return false;
  }

  T Next() {
    if (col_idx_ >= hash_map_->ValueLen()) {
      keys_idx_++;
      col_idx_ = 0;
    }
    Eigen::array<Eigen::DenseIndex, 1> dims({hash_map_->ValueLen()});
    typename TTypes<T>::Flat value_flat = typename TTypes<T>::Flat(valueptr_list_[keys_idx_], dims);
    return value_flat(col_idx_++);
  }

 private:
  HashMap<K, T>* hash_map_;
  std::vector<T* >& valueptr_list_;
  int64 keys_idx_;
  int64 col_idx_;
};


template<class K, class T>
class EVVersionDumpIterator: public  DumpIterator<K, T> {
 public:
  EVVersionDumpIterator(HashMap<T, K>*& hash_map, std::vector<int64 >& version_list):hash_map_(hash_map), version_list_(version_list) {
    keys_idx_ = 0;
  }

  bool HasNext() const {
    return keys_idx_ < version_list_.size();
  }

  T Next() {
    return version_list_[keys_idx_++];
  }

 private:
  HashMap<T, K>* hash_map_;
  std::vector<int64>& version_list_;
  int64 keys_idx_;
};

template <class K, class V>
Status GetInputHashMap(OpKernelContext* ctx, int input,
                       HashMap<K, V>** hashmap) {
  EmbeddingVar<K, V>* var = nullptr;
  if (LookupResource(ctx, HandleFromInput(ctx, input), &var).ok()) {
    *hashmap = var->hashmap();
    return Status::OK();
  } else {
    return errors::Internal("Invalid versioned variable reference.");
  }
}

template <class K, class V>
Status DumpEmbeddingValues(EmbeddingVar<K, V>* ev, const string& tensor_key, BundleWriter* writer, Tensor* part_offset_tensor) {
  std::vector<K> tot_key_list;
  std::vector<V* > tot_valueptr_list;
  std::vector<int64> tot_version_list;
  HashMap<K, V>* hash_map = ev->hashmap();
  int64 total_size = hash_map->GetSnapshot(&tot_key_list, &tot_valueptr_list, &tot_version_list);
  LOG(INFO) << "EV:" << tensor_key << ", save size:" << total_size;
  
  std::vector<std::vector<K> > key_list_parts;
  std::vector<std::vector<V* > > valueptr_list_parts;
  std::vector<std::vector<int64 > > version_list_parts;

  std::vector<K> partitioned_tot_key_list;
  std::vector<V* > partitioned_tot_valueptr_list;
  std::vector<int64> partitioned_tot_version_list;

  key_list_parts.resize(kSavedPartitionNum);
  valueptr_list_parts.resize(kSavedPartitionNum);
  version_list_parts.resize(kSavedPartitionNum);
  //partitioned_tot_key_list.resize(tot_key_list.size());
  //partitioned_tot_valueptr_list.resize(tot_valueptr_list.size());

  // save the ev with kSavedPartitionNum piece of tensor so that we can dynamically load ev with changed partition number
  for (size_t i = 0; i < tot_key_list.size(); i++) {
    for (int partid = 0; partid < kSavedPartitionNum; partid++) {
      if (tot_key_list[i] % kSavedPartitionNum == partid) {
        // std::cout << "key:" << tot_key_list[i] << ", partid:" << partid << std::endl;
        key_list_parts[partid].push_back(tot_key_list[i]);
        valueptr_list_parts[partid].push_back(tot_valueptr_list[i]);
        version_list_parts[partid].push_back(tot_version_list[i]);
        break;
      }
    }
  }
  // LOG(INFO) << "EV:" << tensor_key << ", key_list_parts:" << key_list_parts.size();
  
  auto part_offset_flat = part_offset_tensor->flat<int32>();
  part_offset_flat(0) = 0;
  int ptsize = 0;
  for (int partid = 0; partid < kSavedPartitionNum; partid++) {
    std::vector<K>& key_list = key_list_parts[partid];
    std::vector<V* >& valueptr_list = valueptr_list_parts[partid];
    std::vector<int64>& version_list = version_list_parts[partid];

    ptsize += key_list.size(); 
    for (int inpid = 0; inpid < key_list.size(); inpid++) {
      partitioned_tot_key_list.push_back(key_list[inpid]);
      partitioned_tot_valueptr_list.push_back(valueptr_list[inpid]);
      partitioned_tot_version_list.push_back(version_list[inpid]);
    } 

    part_offset_flat(partid + 1) = part_offset_flat(partid) + key_list.size();
  }
  writer->Add(tensor_key + "-partition_offset", *part_offset_tensor);


  LOG(INFO) << "EV before partition:" << tensor_key << ", keysize:" <<  tot_key_list.size() << ", valueptr size:" << tot_valueptr_list.size();
  LOG(INFO) << "EV after partition:" << tensor_key << ", ptsize:" << ptsize << " ,keysize:" <<  partitioned_tot_key_list.size() << ", valueptr size:" << partitioned_tot_valueptr_list.size();

  size_t bytes_limit = 8 << 20;
  char* dump_buffer = (char*)malloc(sizeof(char) * bytes_limit);
  Status st;
  EVKeyDumpIterator<V, K> ev_key_dump_iter(hash_map, partitioned_tot_key_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-keys", writer, dump_buffer, bytes_limit, &ev_key_dump_iter, TensorShape({partitioned_tot_key_list.size()}));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVValueDumpIterator<K, V> ev_value_dump_iter(hash_map, partitioned_tot_valueptr_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-values", writer, dump_buffer, bytes_limit, &ev_value_dump_iter, TensorShape({partitioned_tot_key_list.size(), hash_map->ValueLen()}));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

  EVVersionDumpIterator<V, K> ev_version_dump_iter(hash_map, partitioned_tot_version_list);
  st = SaveTensorWithFixedBuffer(tensor_key + "-versions", writer, dump_buffer, bytes_limit, &ev_version_dump_iter, TensorShape({partitioned_tot_key_list.size()}));
  if (!st.ok()) {
    free(dump_buffer);
    return st;
  }

    
    
  free(dump_buffer);

  return Status::OK();
}

template<typename K, typename V>
Status DynamicRestoreValue(HashMap<K, V>* hashmap, BundleReader* reader, std::string name_string, int orig_partnum,
       int64 partition_id = 0, int64 partition_num = 1) {
  string part_str = "part_";
  string curr_partid_str = std::to_string(partition_id);
  for (int i = 0; i < orig_partnum; i++) {
    string part_id = std::to_string(i);
    string pre_subname = name_string.substr(0, name_string.find("part_"));
    string post_subname = name_string.substr(name_string.find("part_") + part_str.size() + curr_partid_str.size());
    string tensor_name = pre_subname + part_str + part_id + post_subname;
    
    string tensor_key = tensor_name + "-keys";
    string tensor_value = tensor_name + "-values";
    string tensor_version = tensor_name + "-versions";
    TensorShape key_shape, value_shape, version_shape;      
    Status st = reader->LookupTensorShape(tensor_key, &key_shape);
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupTensorShape(tensor_value, &value_shape);
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupTensorShape(tensor_version, &version_shape);
    if (!st.ok()) {
      return st;
    }
    reader->LookupHeader(tensor_key, sizeof(K) * key_shape.dim_size(0));
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupHeader(tensor_value, sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
    if (!st.ok()) {
      return st;
    }
    st = reader->LookupHeader(tensor_version, sizeof(int64) * version_shape.dim_size(0));
    if (!st.ok()) {
      return st;
    }
    size_t buffer_size = 8 << 20;
    RestoreBuffer restore_buff;
    restore_buff.key_buffer = new char[buffer_size];
    restore_buff.value_buffer = new char[buffer_size];
    restore_buff.version_buffer = new char[buffer_size];
 
    size_t key_bytes_read = 0, value_bytes_read = 0, version_bytes_read = 0;
    int64 tot_key_num = key_shape.dim_size(0);
    size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);

    while(tot_key_num > 0) {  
      size_t read_key_num = std::min(std::min(buffer_size / sizeof(K), buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
      read_key_num = std::min((int64)read_key_num, tot_key_num);
      reader->LookupSegment(tensor_key, read_key_num * sizeof(K), restore_buff.key_buffer, key_bytes_read);
      reader->LookupSegment(tensor_value, read_key_num * value_unit_bytes, restore_buff.value_buffer, value_bytes_read);
      reader->LookupSegment(tensor_version, read_key_num * sizeof(int64), restore_buff.version_buffer, version_bytes_read);

      if (key_bytes_read > 0) {
        read_key_num = key_bytes_read / sizeof(K);
        VLOG(2) << "repartition, read_key_num:" << read_key_num;
        st = hashmap->ImportV2(restore_buff, read_key_num, kSavedPartitionNum, partition_id, partition_num);  
        if (!st.ok()) {
          return st;
        }
        tot_key_num -= read_key_num;
      }
    }
  }
  return Status::OK();
}


template<typename K, typename V>
Status RestoreValue(HashMap<K, V>* hashmap, BundleReader* reader, std::string tensor_key, std::string tensor_value, std::string tensor_version) {
  TensorShape key_shape, value_shape, version_shape;
  reader->LookupTensorShape(tensor_key, &key_shape);
  reader->LookupTensorShape(tensor_value, &value_shape);
  reader->LookupTensorShape(tensor_version, &version_shape);
  Status st;
  st = reader->LookupHeader(tensor_key, sizeof(K) * key_shape.dim_size(0));
  if (!st.ok())
    return st;
  st = reader->LookupHeader(tensor_value, sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
  if (!st.ok())
    return st;
  st = reader->LookupHeader(tensor_version, sizeof(int64) * version_shape.dim_size(0));
  if (!st.ok())
    return st;
  size_t buffer_size = 8 << 20;
  RestoreBuffer restore_buff;
  restore_buff.key_buffer = new char[buffer_size];
  restore_buff.value_buffer = new char[buffer_size];
  restore_buff.version_buffer = new char[buffer_size];
  size_t key_bytes_read = 0, value_bytes_read = 0, version_bytes_read = 0;
   
  int64 tot_key_num = key_shape.dim_size(0);
  size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);
  std::string key_str = "|";
  while(tot_key_num > 0) {  
    size_t read_key_num = std::min(std::min(buffer_size / sizeof(K), buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
    read_key_num = std::min((int64)read_key_num, tot_key_num);
    reader->LookupSegment(tensor_key, read_key_num * sizeof(K), restore_buff.key_buffer, key_bytes_read);
    reader->LookupSegment(tensor_value, read_key_num * value_unit_bytes, restore_buff.value_buffer, value_bytes_read);
    reader->LookupSegment(tensor_version, read_key_num * sizeof(int64), restore_buff.version_buffer, version_bytes_read);
    if (key_bytes_read > 0) {
      read_key_num = key_bytes_read / sizeof(K);
      VLOG(2) << "restore, read_key_num:" << read_key_num;
     
      st = hashmap->ImportV2(restore_buff, read_key_num, 1, 0, 1);
      if (!st.ok())
        return st;
       
      tot_key_num -= read_key_num;
    }
  }

  return Status::OK();
}

template<typename K, typename V>
Status EVRestoreDynamically(HashMap<K, V>* hashmap, std::string name_string, int partition_id, int partition_num, 
          OpKernelContext* context, BundleReader* reader, std::string part_offset_tensor_suffix,
          std::string key_suffix, std::string value_suffix, std::string version_suffix) {

    // first check whether there is partition
    string part_str = "part_";
    if (name_string.find(part_str) == std::string::npos) {
      // no partition    
      Status s = RestoreValue(hashmap, reader, name_string + key_suffix, name_string + value_suffix, name_string + version_suffix);
      if (!s.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << s.ToString();		    
      }
      return s;
    }

    // then check whether checkpoint is in old form
    bool is_oldform = false;
    string curr_partid_str = std::to_string(partition_id);

    {
      string part_id = std::to_string(0);
      string pre_subname = name_string.substr(0, name_string.find(part_str));
      string post_subname = name_string.substr(name_string.find(part_str) + part_str.size() + curr_partid_str.size());
      string tensor_name = pre_subname + part_str + part_id + post_subname;

      TensorShape part_offset_shape;
      DataType part_offset_type; 
      Status form_st = reader->LookupDtypeAndShape(tensor_name + part_offset_tensor_suffix, &part_offset_type, &part_offset_shape); 
      if (!form_st.ok()) {
        is_oldform = true;
      } 
    }
    
    if (is_oldform) {
       // first get original partition number
      int orig_partnum = 0;
      for (;  ; orig_partnum++) {
        string part_id = std::to_string(orig_partnum);
        string pre_subname = name_string.substr(0, name_string.find(part_str));
        string post_subname = name_string.substr(name_string.find(part_str) + part_str.size() + curr_partid_str.size());
        string tensor_name = pre_subname + part_str + part_id + post_subname;

        string tensor_key = tensor_name + key_suffix;
        TensorShape key_shape;
        Status st = reader->LookupTensorShape(tensor_key, &key_shape);
        if (!st.ok()) {
          break;
        }
      }
       
      LOG(INFO) <<  "old form, EV name:" << name_string << ", partition_id:" << partition_id 
            << ", old partition_num:" << orig_partnum << ", new partition num:" << partition_num;
      Status s = DynamicRestoreValue(hashmap, reader, name_string, orig_partnum,  partition_id, partition_num);
      if (!s.ok()) {
        LOG(FATAL) <<  "EV restoring fail:" << s.ToString();
      }
    } else {

       // first find out which sub parts we should load
      std::vector<int> loaded_parts;
      for (int i = 0; i < kSavedPartitionNum; i++) {
        if (i % partition_num == partition_id) {
          loaded_parts.push_back(i);
        }
      }

      // then  we use primary  partition number to compose with sub partition number

      LOG(INFO) << "new form:" << name_string << ", partition_id:" << partition_id << ", partition_num:" << partition_num;

      int orig_partnum = 0;
      size_t buffer_size = 8 << 20;
      RestoreBuffer restore_buff;
      restore_buff.key_buffer = new char[buffer_size];
      restore_buff.value_buffer = new char[buffer_size];
      restore_buff.version_buffer = new char[buffer_size];

      for (;  ; orig_partnum++) {
        string part_id = std::to_string(orig_partnum);
        string pre_subname = name_string.substr(0, name_string.find(part_str));
        string post_subname = name_string.substr(name_string.find(part_str) + part_str.size() + curr_partid_str.size());
        string tensor_name = pre_subname + part_str + part_id + post_subname;

        // first check whether is  old ckpt form 
        string tensor_key = tensor_name + key_suffix;
        string tensor_value = tensor_name + value_suffix;
        string tensor_version = tensor_name + version_suffix;
        TensorShape key_shape, value_shape, version_shape;      
        Status st = reader->LookupTensorShape(tensor_key, &key_shape);
        if (!st.ok()) {
          LOG(INFO) << "ev part " << tensor_key << " not exist, reach the end of restoring";
          break;
        }
        st = reader->LookupTensorShape(tensor_value, &value_shape);
        if (!st.ok()) {
          break;
        }
        st = reader->LookupTensorShape(tensor_version, &version_shape);
        if (!st.ok()) {
          break;
        }

        reader->LookupHeader(tensor_key, sizeof(K) * key_shape.dim_size(0));
        if (!st.ok()) {
          break;
        }
        st = reader->LookupHeader(tensor_value, sizeof(V) * value_shape.dim_size(0) * value_shape.dim_size(1));
        if (!st.ok()) {
          break;
        }
        st = reader->LookupHeader(tensor_version, sizeof(int64) * version_shape.dim_size(0));
        if (!st.ok()) {
          break;
        }
        TensorShape part_offset_shape;
        DataType part_offset_type; 
        string offset_tensor_name = tensor_name + part_offset_tensor_suffix;
        st = reader->LookupDtypeAndShape(offset_tensor_name, &part_offset_type, &part_offset_shape); 
        if (!st.ok()) {
          LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
        }
        
        Tensor part_offset_tensor;
        st = context->allocate_temp(part_offset_type, part_offset_shape, &part_offset_tensor);
        if (!st.ok()) {
          LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
        }
        st = reader->Lookup(offset_tensor_name, &part_offset_tensor);
        if (!st.ok()) {
          LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
        }
        auto part_offset_flat = part_offset_tensor.flat<int32>();


        for (size_t i = 0; i < loaded_parts.size(); i++) {

          int subpart_id = loaded_parts[i];
          int subpart_offset = part_offset_flat(subpart_id);

          size_t value_unit_bytes = sizeof(V) *  value_shape.dim_size(1);
          int64 tot_key_num = part_offset_flat(subpart_id + 1) - subpart_offset;
          int64 key_part_offset = subpart_offset * sizeof(K); 
          int64 value_part_offset = subpart_offset *  value_unit_bytes;
          int64 version_part_offset = subpart_offset * sizeof(int64);

          LOG(INFO) <<  "dynamically load ev : " << name_string <<  " ,subpartid:" << loaded_parts[i] << " ,subpart_offset:" << subpart_offset <<  ", partition_id:" << partition_id << ", partition_num:" << partition_num << " ,keynum:" << tot_key_num;

          int64 tot_key_bytes_read(0), tot_value_bytes_read(0), tot_version_bytes_read(0);
          size_t key_bytes_read = 0, value_bytes_read = 0, version_bytes_read = 0;
          while(tot_key_num > 0) {  
            size_t read_key_num = std::min(std::min(buffer_size / sizeof(K), buffer_size / value_unit_bytes), buffer_size / sizeof(int64));
            read_key_num = std::min((int64)read_key_num, tot_key_num);
            reader->LookupSegmentOffset(tensor_key, key_part_offset + tot_key_bytes_read, read_key_num * sizeof(K),  restore_buff.key_buffer, key_bytes_read);

            reader->LookupSegmentOffset(tensor_value, value_part_offset + tot_value_bytes_read, read_key_num * value_unit_bytes, restore_buff.value_buffer, value_bytes_read);

            reader->LookupSegmentOffset(tensor_version, version_part_offset + tot_version_bytes_read, read_key_num * sizeof(int64) , restore_buff.version_buffer, version_bytes_read);
            if (key_bytes_read > 0) {
              read_key_num = key_bytes_read / sizeof(K);
              VLOG(2) << "restore, read_key_num:" << read_key_num;
              st = hashmap->ImportV3(restore_buff, read_key_num, kSavedPartitionNum, partition_id, partition_num);
              if (!st.ok()) {
                LOG(FATAL) <<  "EV restoring fail:" << st.ToString();
              }
            }
            tot_key_num -= read_key_num;
            tot_key_bytes_read += key_bytes_read;
            tot_value_bytes_read += value_bytes_read;
            tot_version_bytes_read += version_bytes_read;
          }
        }
      }
    }
    return Status::OK();
  } 

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_KV_VARIABLE_OPS_H_