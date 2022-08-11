// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <dlfcn.h>
#include <link.h>
#include <fstream>

#include <c10/util/irange.h>
#include <multipy/runtime/Exception.h>
#include <multipy/runtime/elf_file.h>
#include <multipy/runtime/interpreter/Optional.hpp>

namespace torch {
namespace deploy {

ElfFile::ElfFile(const char* filename)
    : memFile_(std::make_shared<MemFile>(filename)) {
  const char* fileData = memFile_->data();
  ehdr_ = (Elf64_Ehdr*)fileData;
  checkFormat();

  numSections_ = ehdr_->e_shnum;
  shdrList_ = (Elf64_Shdr*)(fileData + ehdr_->e_shoff);

  auto strtabSecNo = ehdr_->e_shstrndx;
  MULTIPY_CHECK(
      strtabSecNo >= 0 && strtabSecNo < numSections_,
      "e_shstrndx out of range");

  strtabSection_ = toSection(&shdrList_[strtabSecNo]);

  sections_.reserve(numSections_);
  for (const auto i : c10::irange(numSections_)) {
    sections_.emplace_back(toSection(&shdrList_[i]));
  }
}

multipy::optional<Section> ElfFile::findSection(const char* name) const {
  MULTIPY_CHECK(name != nullptr, "Null name");
  multipy::optional<Section> found = multipy::nullopt;
  for (const auto& section : sections_) {
    if (strcmp(name, section.name) == 0) {
      found = section;
      break;
    }
  }

  return found;
}

void ElfFile::checkFormat() const {
  // check the magic numbers
  MULTIPY_CHECK(
      (ehdr_->e_ident[EI_MAG0] == ELFMAG0) &&
          (ehdr_->e_ident[EI_MAG1] == ELFMAG1) &&
          (ehdr_->e_ident[EI_MAG2] == ELFMAG2) &&
          (ehdr_->e_ident[EI_MAG3] == ELFMAG3),
      "Unexpected magic numbers");
  MULTIPY_CHECK(
      ehdr_->e_ident[EI_CLASS] == ELFCLASS64, "Only support 64bit ELF file");
}

namespace {
// from https://stackoverflow.com/a/12774387/4722305
// MIT license
inline bool exists(const char* name) {
  std::ifstream f(name);
  return f.good();
}
} // namespace

multipy::optional<Section> searchForSection(const char* name) {
  {
    std::string execPath;
    std::ifstream("/proc/self/cmdline") >> execPath;
    ElfFile elfFile(execPath.c_str());
    auto section = elfFile.findSection(name);
    if (section) {
      return section;
    }
  }

  struct context {
    const char* name;
    multipy::optional<Section> section{};
  };
  context ctx;
  ctx.name = name;

  dl_iterate_phdr(
      [](struct dl_phdr_info* info, size_t, void* data) {
        if (!exists(info->dlpi_name)) {
          return 0;
        }
        ElfFile elfFile(info->dlpi_name);
        auto localCtx = static_cast<context*>(data);
        localCtx->section = elfFile.findSection(localCtx->name);
        if (localCtx->section) {
          return 1;
        }
        return 0;
      },
      &ctx);
  return ctx.section;
}

} // namespace deploy
} // namespace torch
