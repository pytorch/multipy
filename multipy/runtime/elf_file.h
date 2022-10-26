// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <elf.h>
#include <multipy/runtime/Exception.h>
#include <multipy/runtime/interpreter/Optional.hpp>
#include <multipy/runtime/mem_file.h>
#include <memory>
#include <vector>

namespace torch {
namespace deploy {

/// A section of an ElfFile.
struct Section {
  Section() {}
  explicit Section(
      std::shared_ptr<MemFile> _memfile,
      const char* _name,
      const char* _start,
      size_t _len = 0)
      : memfile(_memfile), name(_name), start(_start), len(_len) {}

  std::shared_ptr<MemFile> memfile;
  const char* name{nullptr};
  const char* start{nullptr};
  size_t len{0};

  operator bool() const {
    return start != nullptr;
  }
};

// TODO: consolidate other ELF file related functions in loader.cpp to this file

/*
 * This class provie utilities to handle ELF file. Only support 64bit ELF file.
 */
class ElfFile {
 public:
  /// Constructs an Elffile with the corresponding `filename`
  explicit ElfFile(const char* filename);

  /// Finds and return a `Section` with the corresponding `name`.  If nothing is
  /// found, then a `multipy::nullopt` is returned.
  multipy::optional<Section> findSection(const char* name) const;

 private:
  Section toSection(Elf64_Shdr* shdr) {
    auto nameOff = shdr->sh_name;
    auto shOff = shdr->sh_offset;
    auto len = shdr->sh_size;
    const char* name = "";

    if (strtabSection_) {
      MULTIPY_CHECK(nameOff >= 0 && nameOff < strtabSection_.len);
      name = strtabSection_.start + nameOff;
    }
    const char* start = memFile_->data() + shOff;
    return Section{memFile_, name, start, len};
  }

  [[nodiscard]] const char* str(size_t off) const {
    MULTIPY_CHECK(off < strtabSection_.len, "String table index out of range");
    return strtabSection_.start + off;
  }
  void checkFormat() const;
  std::shared_ptr<MemFile> memFile_;
  Elf64_Ehdr* ehdr_;
  Elf64_Shdr* shdrList_;
  size_t numSections_;

  Section strtabSection_;
  std::vector<Section> sections_;
};

multipy::optional<Section> searchForSection(const char* name);

} // namespace deploy
} // namespace torch
