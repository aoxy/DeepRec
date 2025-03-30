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
=======================================================================*/

#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASH_KV_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASH_KV_H_

#include <map>
#include <vector>
#include <cstdlib>

#include "sparsehash/dense_hash_map_lockless"
#include "sparsehash/dense_hash_set_lockless"
#include "tensorflow/core/framework/embedding/ssd_record_descriptor.h"
#include "tensorflow/core/framework/embedding/emb_file_creator.h"
#include "tensorflow/core/framework/embedding/kv_interface.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {
typedef uint32 EmbPosition;

const uint32 kEmbPositionVersionMod  = 0b11111111111;
const uint32 kEmbPositionVersionMask = 0b11111111111000000000000000000000;
const uint32 kEmbPositionOffsetMask  = 0b00000000000111111111111111111100;
const uint32 kEmbPositionInvalidMask = 0b00000000000000000000000000000010;
const uint32 kEmbPositionFlushedMask = 0b00000000000000000000000000000001;
const uint32 kEmbPositionVersionOffset = 21;

static inline EmbPosition SetEmbPositionVersion(EmbPosition& ep, uint32 offset) {
  ep &= ~static_cast<uint32>(kEmbPositionVersionMask);
  ep |= offset << kEmbPositionVersionOffset;
  return ep;
}

static inline EmbPosition SetEmbPositionOffset(EmbPosition& ep, uint32 offset) {
  ep &= ~kEmbPositionOffsetMask;
  ep |= offset << 2;
  return ep;
}

static inline EmbPosition SetEmbPositionInvalid(EmbPosition& ep, bool invalid) {
  ep &= ~kEmbPositionInvalidMask;
  ep |= invalid << 1;
  return ep;
}

static inline EmbPosition SetEmbPositionFlushed(EmbPosition& ep, bool flushed) {
  if (flushed) {
    ep |= kEmbPositionFlushedMask;
  } else {
    ep &= ~kEmbPositionFlushedMask;
  }
  return ep;
}

static inline uint32 GetEmbPositionVersion(EmbPosition& ep) {
  return (ep & kEmbPositionVersionMask) >> kEmbPositionVersionOffset;
}

static inline uint32 GetEmbPositionOffset(EmbPosition& ep) {
  return (ep & kEmbPositionOffsetMask) >> 2;
}

static inline bool IsEmbPositionInvalid(EmbPosition& ep) {
  return ep & kEmbPositionInvalidMask;
}

static inline bool IsEmbPositionFlushed(EmbPosition& ep) {
  return ep & kEmbPositionFlushedMask;
}

static inline bool IsEmbPositionSameIgnoreStatus(EmbPosition& ep1, EmbPosition& ep2) {
  return (ep1 >> 2) == (ep2 >> 2);
}

static inline EmbPosition CreateEmbPosition(uint32 o, uint32 v, bool f) {
  EmbPosition ep = static_cast<uint32>(f);
  SetEmbPositionOffset(ep, o);
  SetEmbPositionVersion(ep, v);
  return ep;
}

static inline EmbPosition CreateEmbPosition() { return 0; }

static inline void PrintEmbPosition(EmbPosition& ep) {
  std::cout << "EmbPosition: " << ep
            << ", version = " << GetEmbPositionVersion(ep)
            << ", offset = " << GetEmbPositionOffset(ep)
            << ", invalid = " << IsEmbPositionInvalid(ep)
            << ", flushed = " << IsEmbPositionFlushed(ep) << std::endl;
}

template <class K>
class SSDIterator {
 public:
  SSDIterator(google::dense_hash_map_lockless<K, EmbPosition>* hash_map,
              const std::vector<EmbFile*>& emb_files, int64 value_len,
              char* write_buffer)
      : emb_files_(emb_files),
        curr_file_(0),
        curr_vec_(0),
        value_len_(value_len),
        write_buffer_(write_buffer) {
    for (auto it : *hash_map) {
      EmbPosition posi = it.second;
      uint32 version = GetEmbPositionVersion(posi);
      auto iter = file_map_.find(version);
      if (iter == file_map_.end()) {
        std::vector<std::pair<K, EmbPosition>> tmp;
        file_map_[version] = tmp;
        file_id_vec_.emplace_back(version);
      }
      file_map_[version].emplace_back(it);
    }
  }

  virtual ~SSDIterator() {}

  virtual bool Valid() {
    return !(curr_file_ == file_id_vec_.size());
  }

  virtual void SeekToFirst() {
    curr_file_ = 0;
    curr_vec_ = 0;
    if (file_id_vec_.size() > 0) {
      int64 f_id = file_id_vec_[curr_file_];
      emb_files_[f_id]->MapForRead();
    }
  }

  virtual void Next() {
    curr_vec_++;
    int64 f_id = file_id_vec_[curr_file_];
    if (curr_vec_ == file_map_[f_id].size()) {
      emb_files_[f_id]->UnmapForRead();
      curr_vec_ = 0;
      curr_file_++;
      if (curr_file_ < file_id_vec_.size())
        emb_files_[file_id_vec_[curr_file_]]->MapForRead();
    }
  }

  virtual K Key() {
    int64 f_id = file_id_vec_[curr_file_];
    return (file_map_[f_id])[curr_vec_].first;
  }

  virtual int64 FileId() {
    return file_id_vec_[curr_file_];
  }

  virtual int64 Offset() {
    int64 f_id = file_id_vec_[curr_file_];
    EmbPosition posi = (file_map_[f_id])[curr_vec_].second;
    return GetEmbPositionOffset(posi);
  }

 private:
  int64 value_len_;
  int64 curr_file_;
  int64 curr_vec_;
  char* write_buffer_;
  std::map<int64, std::vector<std::pair<K, EmbPosition>>> file_map_;
  std::vector<int64> file_id_vec_;
  std::vector<EmbFile*> emb_files_;
};

template <class K, class V>
class SSDHashKV : public KVInterface<K, V> {
 public:
  explicit SSDHashKV(const std::string& path,
                     FeatureDescriptor<V>* feat_desc)
  : feat_desc_(feat_desc) {
    path_ = io::JoinPath(
        path, "ssd_kv_" + std::to_string(Env::Default()->NowMicros()) + "_");
    hash_map_.max_load_factor(0.8);
    hash_map_.set_empty_key_and_value(EMPTY_KEY, 987654321);
    hash_map_.set_counternum(16);
    hash_map_.set_deleted_key(DELETED_KEY);
    evict_file_set_.max_load_factor(0.8);
    evict_file_set_.set_empty_key_and_value(EMPTY_KEY, -1);
    evict_file_set_.set_counternum(16);
    evict_file_set_.set_deleted_key(DELETED_KEY);

    std::string io_scheme = "mmap_and_madvise";
    TF_CHECK_OK(ReadStringFromEnvVar(
        "TF_SSDHASH_IO_SCHEME", "mmap_and_madvise", &io_scheme));
    emb_file_creator_ =  EmbFileCreatorFactory::Create(io_scheme);
    CreateFile(false);

    bool enable_compaction_ = true;
    TF_CHECK_OK(ReadBoolFromEnvVar("TF_ENABLE_SSDKV_COMPACTION", true,
                                   &enable_compaction_));
    compaction_fn_ = [this]() { };
    if(enable_compaction_) {
      compaction_fn_ = [this]() { Compaction(); };
    }
    check_buffer_fn_ = [this]() { CheckBuffer(); };
    save_kv_fn_ = [this](K key, const void* value_ptr,
                         bool is_compaction = false) {
      SaveKV(key, value_ptr, is_compaction);
    };
  }

  void Init() {
    val_len_ = feat_desc_->data_bytes();
    file_capacity_ = BUFFER_SIZE / val_len_;
    if (file_capacity_ > (1 << 19)) {
      LOG(FATAL) << "The 19-bit offset is not enough to save the Embedding of length " << val_len_ << " in the SSD KV.";
    }
    write_buffer_ = new char[BUFFER_SIZE];
    key_buffer_ = new K[file_capacity_];
    VLOG(0) << "[SSDKV] val_len_ = " << val_len_;
    VLOG(0) << "[SSDKV] file_capacity_ = " << file_capacity_;
  }

  void SetSsdRecordDescriptor(SsdRecordDescriptor<K>* ssd_rec_desc) {
    SaveBuffer();
    SSDIterator<K> ssd_iter(&hash_map_, emb_files_, val_len_, write_buffer_);
    for (ssd_iter.SeekToFirst(); ssd_iter.Valid(); ssd_iter.Next()) {
      ssd_rec_desc->key_list.emplace_back(ssd_iter.Key());
      ssd_rec_desc->key_file_id_list.emplace_back(ssd_iter.FileId());
      ssd_rec_desc->key_offset_list.emplace_back(ssd_iter.Offset());
    }
    ssd_rec_desc->file_prefix = path_;
    for (auto file: emb_files_) {
      if (file->IsDeleted() || evict_file_map_.count(file->Version()))
        continue;
      ssd_rec_desc->file_list.emplace_back(file->Version());
      ssd_rec_desc->invalid_record_count_list.emplace_back(
          file->InvalidCount());
      ssd_rec_desc->record_count_list.emplace_back(
          file->Count());
    }
  }

  ~SSDHashKV() override {
    SaveBuffer();
    for (auto it : emb_files_) {
      if (!it->IsDeleted()) {
        it->DeleteFile();
      }
      delete it;
    }
    delete[] write_buffer_;
    delete[] key_buffer_;
    VLOG(0) << "[SSDKV] current_version_ = " << current_version_;
    VLOG(0) << "[SSDKV] total_written_count_ = " << current_version_ * file_capacity_ + current_offset_;
    VLOG(0) << "[SSDKV] active_emb_count_ = " << active_emb_count_;
    VLOG(0) << "[SSDKV] compaction_count_ = " << compaction_count_;
    VLOG(0) << "[SSDKV] compaction_skip_count_ = " << compaction_skip_count_; // TODO: Only used in testing
    VLOG(0) << "[SSDKV] compaction_app_count_ = " << compaction_app_count_; // TODO: Only used in testing
    VLOG(0) << "[SSDKV] Write amplification factor = " << 1.0 * (current_version_ * file_capacity_ + current_offset_) / active_emb_count_;
    VLOG(0) << "[SSDKV] hash_map_.size_lockless() = " << hash_map_.size_lockless();
  }

  void EnsureUpdate(K key, EmbPosition target, int place=101) {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == EMPTY_KEY) {
      LOG(FATAL) << place << " - Unable to find Key: " << key << " in SSDHashKV.";
    } else if (iter.second != target) {
      LOG(FATAL) << place << " - Update Key: " << key << " Failed. " << iter.second << "->" << target;
    }
  }

  EmbPosition UpdatePositionEnsure(K key, EmbPosition target) {
    auto iter = hash_map_.insert_lockless(std::move(
        std::pair<K, EmbPosition>(key, target)));
    EmbPosition posi = (*(iter.first)).second;
    bool flag = false;
    do {
      flag = __sync_bool_compare_and_swap(
              &((*(iter.first)).second),
              (*(iter.first)).second,
              target);
    } while (!flag);
    return posi;
  }

  EmbPosition UpdatePositionStatus(K key, EmbPosition posi, std::function<EmbPosition(EmbPosition&, bool)> set_status) {
    EmbPosition target = posi;
    set_status(target, true);
    auto iter = hash_map_.insert_lockless(std::move(
        std::pair<K, EmbPosition>(key, target)));
    bool flag = __sync_bool_compare_and_swap(
                &((*(iter.first)).second),
                posi, target);
    if (!flag) {
      posi = (*(iter.first)).second;
      if (IsEmbPositionSameIgnoreStatus(target, posi)) {
        __sync_bool_compare_and_swap(
          &((*(iter.first)).second),
          posi, set_status(posi, true));
      }
    }
    return posi;
  }

  Status UpdateFlushStatus() {
    for (int i = 0; i < current_offset_; ++i) {
      auto iter = hash_map_.find_wait_free(key_buffer_[i]);
      if (iter.first == EMPTY_KEY) {
        return errors::NotFound("Unable to find Key: ",
            key_buffer_[i], " in SSDHashKV.");
      } else {
        EmbPosition posi = iter.second;
        UpdatePositionStatus(key_buffer_[i], posi, SetEmbPositionFlushed);
      }
    }
    current_offset_ = 0;
    return Status::OK();
  }

  Status Lookup(K key, void** value_ptr) override {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == EMPTY_KEY) {
      return errors::NotFound("Unable to find Key: ", key, " in SSDHashKV.");
    } else {
      void* val = feat_desc_->Allocate();
      EmbPosition posi = iter.second;
      int offset = GetEmbPositionOffset(posi) * val_len_;
      if (IsEmbPositionFlushed(posi)) {
        emb_files_[GetEmbPositionVersion(posi)]->Read((char*)val,
            val_len_, offset);
      } else {
        memcpy((char*)val, write_buffer_ + offset, val_len_);
      }
      *value_ptr = val;
      UpdatePositionStatus(key, posi, SetEmbPositionInvalid);
      return Status::OK();
    }
  }

  Status Contains(K key) override {
    auto iter = hash_map_.find_wait_free(key);
    if (iter.first == EMPTY_KEY) {
      return errors::NotFound("Unable to find Key: ", key, " in SSDHashKV.");
    } else {
      return Status::OK();
    }
  }

  Status Insert(K key, const void* value_ptr) override {
    return Status::OK();
  }

  Status BatchInsert(const std::vector<K>& keys,
                     const std::vector<void*>& value_ptrs) override {
    return BatchCommit(keys, value_ptrs);
  }

  Status BatchCommit(const std::vector<K>& keys,
                     const std::vector<void*>& value_ptrs) override {
    compaction_fn_();
    __sync_fetch_and_add(&active_emb_count_, keys.size());
    for (int i = 0; i < keys.size(); i++) {
      check_buffer_fn_();
      save_kv_fn_(keys[i], value_ptrs[i], false);
      delete value_ptrs[i];
    }
    return Status::OK();
  }

  Status Commit(K key, const void* value_ptr) override {
    compaction_fn_();
    __sync_fetch_and_add(&active_emb_count_, 1);
    check_buffer_fn_();
    save_kv_fn_(key, value_ptr, false);
    return Status::OK();
  }

  Status Remove(K key) override {
    if (hash_map_.erase_lockless(key)) {
      return Status::OK();
    } else {
      return errors::NotFound("Unable to find Key: ",
          key, " in SSDHashKV.");
    }
  }

  Status GetSnapshot(std::vector<K>* key_list,
                     std::vector<void*>* value_ptr_list) override {
    return Status::OK();
  }

  Status GetShardedSnapshot(
      std::vector<K>* key_list, std::vector<void*>* value_ptr_list,
      int partition_id, int partition_nums) override {
    return Status::OK();
  }

  Status GetSnapshot(
      std::vector<K>* key_list,
      std::vector<EmbFile*>* file_list) {
    int64 bucket_count;
    auto it = hash_map_.GetSnapshot();
    auto hash_map_dump = it.first;
    bucket_count = it.second;
    for (int64 j = 0; j < bucket_count; j++) {
      if (hash_map_dump[j].first != LocklessHashMap<K, V>::EMPTY_KEY_
          && hash_map_dump[j].first != LocklessHashMap<K, V>::DELETED_KEY_) {
        key_list->emplace_back(hash_map_dump[j].first);
        file_list->emplace_back(hash_map_dump[j].second);
      }
    }
    //Free the memory of snapshot allocated by hash map.
    free(hash_map_dump);
    return Status::OK();
  }

  void Import(K* key_list, int64* key_file_id_list,
              int64* key_offset_list, int64 num_of_keys,
              std::map<int64, int64>& file_id_map) {
    for (int i = 0; i < num_of_keys; i++) {
      int64 old_file_id = key_file_id_list[i];
      int64 new_file_id = file_id_map[old_file_id];
      EmbPosition ep = CreateEmbPosition(key_offset_list[i], new_file_id, true);
      hash_map_.insert_lockless(std::move(
        std::pair<K, EmbPosition>(key_list[i], ep)));
    }
  }

  void CopyEmbFilesFromCkpt(
      int64* file_list, int64* invalid_record_count_list,
      int64* record_count_list, int64 num_of_files,
      const std::string& old_file_prefix) {
    for (int64 i = 0; i < num_of_files; i++) {
      std::stringstream ss;
      ss << old_file_prefix << "/" << file_list[i] << ".emb";
      std::string old_file_path = ss.str();
      EmbFile* f = emb_files_[current_version_];
      f->LoadExistFile(old_file_path,
                       record_count_list[i],
                       invalid_record_count_list[i]);
      f->Reopen();
      active_emb_count_ += record_count_list[i];
      CreateFile();
    }
  }

  int64 Size() const override { return hash_map_.size_lockless(); }

  void FreeValuePtr(void* value_ptr) override {
    feat_desc_->Deallocate(value_ptr);
  }

 private:
  void WriteFile(size_t version, size_t curr_buffer_offset) {
    version = version & kEmbPositionVersionMod;
    emb_files_[version]->Reopen();
    emb_files_[version]->Write(write_buffer_, curr_buffer_offset);
    emb_files_[version]->Flush();
  }

  void CreateFile(bool need_increase = true) {
    if (need_increase) {
      ++current_version_;
    }
    uint32 version = current_version_ & kEmbPositionVersionMod;
    if (version < emb_files_.size()) {
      while (!emb_files_[version]->IsDeleted()) {
        VLOG(0) << "Embedding file still exists. version = " << version;
        ++current_version_;
        version = current_version_ & kEmbPositionVersionMod;
      }
      delete emb_files_[version];
      emb_files_[version] =
          emb_file_creator_->Create(path_, version, BUFFER_SIZE, val_len_);
    } else {
      EmbFile* f = emb_file_creator_->Create(path_, version, BUFFER_SIZE, val_len_);
      emb_files_.emplace_back(f);
    }
  }

  void CheckBuffer() {
    if (current_offset_ >= file_capacity_) {
      WriteFile(current_version_, current_offset_ * val_len_);
      TF_CHECK_OK(UpdateFlushStatus());
      CreateFile();
    }
  }

  void SaveBuffer() {
    if (current_offset_ > 0) {
      WriteFile(current_version_, current_offset_ * val_len_);
      TF_CHECK_OK(UpdateFlushStatus());
      CreateFile();
    }
  }

  void AppendToWriteBuffer(K key, const void* value_ptr) {
    memcpy(write_buffer_ + current_offset_ * val_len_,
        (char*)value_ptr, val_len_);
    key_buffer_[current_offset_] = key;
    ++current_offset_;
  }

  void SaveKV(K key, const void* value_ptr,
      bool is_compaction = false) {
    uint32 temp_current_version_ = current_version_ & kEmbPositionVersionMod;
    EmbPosition ep = CreateEmbPosition(current_offset_, temp_current_version_, false);
    AppendToWriteBuffer(key, value_ptr);
    EmbPosition old_posi = UpdatePositionEnsure(key, ep);
    emb_files_[temp_current_version_]->AddCount(1);
    if (!is_compaction && old_posi != ep) {
      uint32 old_version = GetEmbPositionVersion(old_posi);
      emb_files_[old_version]->AddInvalidCount(1);
      if (old_version != temp_current_version_ &&
          !emb_files_[old_version]->IsDeleted() &&
          emb_files_[old_version]->IsNeedToBeCompacted()) {
        evict_file_set_.insert_lockless(old_version);
      }
    }
  }

  void DeleteInvalidFiles() {
    for (auto it : evict_file_map_) {
      emb_files_[it.first]->DeleteFile();
    }
    evict_file_map_.clear();
  }

  void LookupValidItems() {
    for (auto it : hash_map_) {
      EmbPosition posi = it.second;
      auto iter = evict_file_map_.find(GetEmbPositionVersion(posi));
      if (iter != evict_file_map_.end()) {
        (*iter).second.emplace_back(it);
      }
    }
  }

  void InitializeEvictMap() {
    for (auto it : evict_file_set_) {
      if (!emb_files_[it]->IsDeleted()) {
        std::vector<std::pair<K, EmbPosition>> tmp;
        evict_file_map_[it] = tmp;
      }
    }
    for (auto it : evict_file_map_) {
      evict_file_set_.erase_lockless(it.first);
    }
    if (!evict_file_map_.empty()) {
      LookupValidItems();
    }
  }

  void MoveToNewFile() {
    if (evict_file_map_.empty()) {
      return;
    }
    ++compaction_count_;
    void* val = feat_desc_->Allocate();
    for (auto it : evict_file_map_) {
      EmbFile* file = emb_files_[it.first];
      active_emb_count_ -= file->InvalidCount();
      file->MapForRead();
      for (auto it_vec : it.second) {
        EmbPosition posi = it_vec.second;
        if (!IsEmbPositionInvalid(posi)) {
          file->ReadWithMemcpy((char*)val, val_len_,
            GetEmbPositionOffset(posi) * val_len_);
          CheckBuffer();
          SaveKV(it_vec.first, val, true);
          compaction_app_count_++;
        } else {
          compaction_skip_count_++;
        }
      }
      file->UnmapForRead();
    }
    feat_desc_->Deallocate(val);
  }

  void Compaction() {
    int64 hash_size = hash_map_.size_lockless();
    //These parameter that can be adjusted in the future
    if (hash_size * 3 / 2 < active_emb_count_ ||
        active_emb_count_ - hash_size > CAP_INVALID_ID) {
      // delete the evict_files
      DeleteInvalidFiles();
      // Initialize evict_file_map
      InitializeEvictMap();
      // read embeddings and write to new file
      MoveToNewFile();
    }
  }

  std::string DebugString() const {
    return strings::StrCat("map info size:", Size(),
                           ", map info bucket_count:",
                           hash_map_.load_factor(),
                           ",map info load_factor:",
                           hash_map_.load_factor(),
                           ", map info max_load_factor:",
                           hash_map_.max_load_factor(),
                           ", map info min_load_factor: ",
                           hash_map_.min_load_factor());
  }
 private:

 private:
  uint32 val_len_ = 0;
  uint32 current_version_ = 0;
  uint32 file_capacity_ = 0;
  uint32 current_offset_ = 0;
  uint32 compaction_count_ = 0;
  size_t active_emb_count_ = 0;
  size_t compaction_skip_count_ = 0;
  size_t compaction_app_count_ = 0;

  char* write_buffer_ = nullptr;
  K* key_buffer_ = nullptr;
  FeatureDescriptor<V>* feat_desc_;

  std::string path_;

  typedef google::dense_hash_map_lockless<K, EmbPosition> LockLessHashMap;
  LockLessHashMap hash_map_;

  static const int EMPTY_KEY;
  static const int DELETED_KEY;
  static const int CAP_INVALID_ID;
  static const uint64 BUFFER_SIZE;

  std::vector<EmbFile*> emb_files_;
  typedef google::dense_hash_set_lockless<K> LocklessHashSet;
  LocklessHashSet evict_file_set_;
  std::map<int64, std::vector<std::pair<K, EmbPosition>>> evict_file_map_;

  std::function<void()> compaction_fn_;
  std::function<void()> check_buffer_fn_;
  std::function<void(K, const void*, bool)> save_kv_fn_;
  EmbFileCreator* emb_file_creator_ = nullptr;
};
template <class K, class V>
const int SSDHashKV<K, V>::EMPTY_KEY = -1;
template <class K, class V>
const int SSDHashKV<K, V>::DELETED_KEY = -2;
template <class K, class V>
const int SSDHashKV<K, V>::CAP_INVALID_ID = 10000000;
template <class K, class V>
const uint64 SSDHashKV<K, V>::BUFFER_SIZE = 1 << 27;

}  // namespace embedding
}  // namespace tensorflow

#endif //TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_SSD_HASH_KV_H_
