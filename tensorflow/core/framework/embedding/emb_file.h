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
#ifndef TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMB_FILE_H_
#define TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMB_FILE_H_
#include <fcntl.h>
#include <malloc.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <map>
#include <string>

#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace embedding {
class EmbFile {
 public:
  EmbFile(const std::string& path,
          uint32 version,
          uint64 buffer_size,
          uint32 val_len)
      : version_(version),
        file_size_(buffer_size),
        val_len_(val_len),
        fd_(-1),
        count_(0),
        invalid_count_(0),
        is_deleted_(true) {
    std::stringstream ss;
    ss << std::setw(6) << std::setfill('0') << version << ".emb";
    filepath_ = path + ss.str();
  }

  virtual ~EmbFile() {}
  virtual void Reopen() = 0;
  virtual void Read(char* val, const size_t val_len,
      const size_t offset) = 0;

  virtual void DeleteFile() {
    if (fs_.is_open()) {
      fs_.close();
    }
    if (fd_ > 0) {
      close(fd_);
    }
    if (!is_deleted_) {
      is_deleted_ = true;
      std::remove(filepath_.c_str());
    }
  }

  void LoadExistFile(const std::string& old_file_path,
                     uint32 count,
                     uint32 invalid_count) {
    TF_CHECK_OK(Env::Default()->CopyFile(old_file_path, filepath_));
    TF_CHECK_OK(Env::Default()->GetFileSize(filepath_, &file_size_));
    count_ = count;
    invalid_count_ = invalid_count;
  }

  void Flush() {
    if (fs_.is_open()) {
      fs_.flush();
    }
  }

  void MapForRead() {
    file_addr_for_read_ =
        (char*)mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
  }

  void UnmapForRead() { munmap((void*)file_addr_for_read_, file_size_); }

  void ReadWithMemcpy(char* val, const size_t val_len,
      const size_t offset) {
    memcpy(val, file_addr_for_read_ + offset, val_len);
  }

  void Write(const char* data, const size_t data_len) {
    if (fs_.is_open()) {
      fs_.write(data, data_len);
      file_size_ = fs_.tellg();
      CHECK(!fs_.fail());
      posix_fadvise(fd_, 0, file_size_, POSIX_FADV_DONTNEED);
    } else {
      OpenFstream();
      fs_.write(data, data_len);
      file_size_ = fs_.tellg();
      CHECK(!fs_.fail());
      CloseFstream();
    }
  }

  uint32 Count() const { return count_; }

  void AddCount(uint32 n) { count_ += n; }

  uint32 InvalidCount() const { return invalid_count_; }

  void AddInvalidCount(uint32 n) { invalid_count_ += n; }

  void AddInvalidCountAtomic(uint32 n) {
    __sync_fetch_and_add(&invalid_count_, n);
  }

  uint32 Version() const { return version_; }

  bool IsDeleted() const { return is_deleted_; }

  bool IsNeedToBeCompacted() {
    return (count_ >= invalid_count_) && (count_ / 3 < invalid_count_);
  }

 protected:
  virtual void OpenFstream() {
    fs_.open(filepath_, std::ios::app | std::ios::out | std::ios::binary);
    if (!fs_.good()) {
      LOG(FATAL) << "The directory for file " << filepath_
                 << " may not exist. Please create it and try again.";
    }
    is_deleted_ = false;
  }

  virtual void CloseFstream() {
    if (fs_.is_open()) {
      fs_.close();
    }
  }

 private:
  uint32 version_;
  uint32 count_;
  uint32 invalid_count_;
  char* file_addr_for_read_;
  std::fstream fs_;

 protected:
  uint64 file_size_;
  uint32 val_len_;
  int fd_;
  bool is_deleted_;
  std::string filepath_;
};

class MmapMadviseEmbFile : public EmbFile {
 public:
  MmapMadviseEmbFile(const std::string& path,
                     uint32 ver,
                     uint64 buffer_size,
                     uint32 val_len)
      : EmbFile(path, ver, buffer_size, val_len) {}

  void Reopen() override {
    OpenFstream();
    // CloseFstream();
    EmbFile::fd_ = open(EmbFile::filepath_.c_str(), O_RDONLY);
    file_addr_ = (char*)mmap(nullptr, EmbFile::file_size_, PROT_READ,
                             MAP_PRIVATE, fd_, 0);
  }

  void DeleteFile() override {
    is_deleted_ = true;
    CloseFstream();
    munmap((void*)file_addr_, EmbFile::file_size_);
    close(EmbFile::fd_);
    std::remove(EmbFile::filepath_.c_str());
  }

  void Read(char* val, const size_t val_len,
            const size_t offset) override {
    memcpy(val, file_addr_ + offset, val_len);
    int err = madvise(file_addr_, EmbFile::file_size_, MADV_DONTNEED);
    if (err < 0) {
      LOG(FATAL) << "Failed to madvise the page, file_addr_: "
                 << (void*)file_addr_ << ", file_size: " << EmbFile::file_size_;
    }
  }

 private:
  char* file_addr_;
};

class MmapEmbFile : public EmbFile {
 public:
  MmapEmbFile(const std::string& path,
              uint32 ver,
              uint64 buffer_size,
              uint32 val_len)
      : EmbFile(path, ver, buffer_size, val_len) {}

  void Reopen() override {
    OpenFstream();
    // CloseFstream();
    EmbFile::fd_ = open(EmbFile::filepath_.c_str(), O_RDONLY);
    file_addr_ = (char*)mmap(nullptr, EmbFile::file_size_, PROT_READ,
                             MAP_PRIVATE, fd_, 0);
  }

  void DeleteFile() override {
    is_deleted_ = true;
    CloseFstream();
    munmap((void*)file_addr_, EmbFile::file_size_);
    close(EmbFile::fd_);
    std::remove(EmbFile::filepath_.c_str());
  }

  void Read(char* val, const size_t val_len,
            const size_t offset) override {
    memcpy(val, file_addr_ + offset, val_len);
  }

 private:
  char* file_addr_;
};

class DirectIoEmbFile : public EmbFile {
 public:
  DirectIoEmbFile(const std::string& path,
                  uint32 ver,
                  uint64 buffer_size,
                  uint32 val_len)
      : EmbFile(path, ver, buffer_size, val_len) {
    page_size_ = getpagesize();
    int pages_to_read = (val_len_ + page_size_ - 1) / page_size_ + 1;
    read_buffer_size_ = page_size_ * pages_to_read;
  }

  void Reopen() override {
    OpenFstream();
    // CloseFstream();
    EmbFile::fd_ = open(EmbFile::filepath_.c_str(), O_RDONLY | O_DIRECT);
    
  }

  void Read(char* val, const size_t val_len,
            const size_t offset) override {
    size_t page_size = getpagesize();
    int pages_to_read = val_len / page_size;
    if (val_len % page_size != 0) {
      pages_to_read += 1;
    }
    if (offset + val_len >= page_size * pages_to_read) {
      pages_to_read += 1;
    }
    int aligned_offset = offset - (offset % page_size);
    char* read_buffer = (char*)memalign(page_size, page_size * pages_to_read);

    int status = pread(EmbFile::fd_,
                       (void*)read_buffer,
                       page_size * pages_to_read,
                       aligned_offset);
    if (status < 0) {
      LOG(FATAL)<<"Failed to pread, read size: "
                <<page_size * pages_to_read
                <<", offset: "<<aligned_offset;
    }
    memcpy(val, read_buffer + (offset % page_size), val_len);
    free(read_buffer);
  }

 private:
  uint32 read_buffer_size_ = 0;
  size_t page_size_;
};

}  // namespace embedding
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_EMBEDDING_EMB_FILE_H_
