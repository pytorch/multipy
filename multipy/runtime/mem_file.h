// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <fcntl.h>
#include <multipy/runtime/Exception.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#include <cstdio>
#include <iostream>

namespace torch {
namespace deploy {

/// Memory maps a file into the address space read-only, and manages the
/// lifetime of the mapping. Here are a few use cases:
/// 1. Used in the loader to read in initial image, and to inspect
// ELF files for dependencies before callling dlopen.
///
/// 2. Used in unity to load the elf file.
struct MemFile {
  explicit MemFile(const char* filename_)
      : fd_(0), mem_(nullptr), n_bytes_(0), name_(filename_) {
    fd_ = open(filename_, O_RDONLY);
    MULTIPY_CHECK(
        fd_ != -1, "failed to open {}: {}" + filename_ + strerror(errno));
    // NOLINTNEXTLINE
    struct stat s;
    if (-1 == fstat(fd_, &s)) {
      close(fd_); // destructors don't run during exceptions
      MULTIPY_CHECK(
          false, "failed to stat {}: {}" + filename_ + strerror(errno));
    }
    n_bytes_ = s.st_size;
    mem_ = mmap(nullptr, n_bytes_, PROT_READ, MAP_SHARED, fd_, 0);
    if (MAP_FAILED == mem_) {
      close(fd_);
      MULTIPY_CHECK(
          false, "failed to mmap {}: {}" + filename_ + strerror(errno));
    }
  }
  MemFile(const MemFile&) = delete;
  MemFile& operator=(const MemFile&) = delete;
  [[nodiscard]] const char* data() const {
    return (const char*)mem_;
  }

  /// Returns whether or not the file descriptor
  /// of the underlying file is valid.
  int valid() {
    return fcntl(fd_, F_GETFD) != -1 || errno != EBADF;
  }
  ~MemFile() {
    if (mem_) {
      munmap((void*)mem_, n_bytes_);
    }
    if (fd_) {
      close(fd_);
    }
  }

  /// Returns the size of the underlying file defined by the `MemFile`
  size_t size() {
    return n_bytes_;
  }
  [[nodiscard]] int fd() const {
    return fd_;
  }

 private:
  int fd_;
  void* mem_;
  size_t n_bytes_;
  std::string name_;
};

} // namespace deploy
} // namespace torch
