// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <dlfcn.h>
#include <multipy/runtime/Exception.h>
#include <multipy/runtime/elf_file.h>
#include <multipy/runtime/embedded_file.h>
#include <torch/cuda.h>
#include <fstream>

namespace torch {
namespace deploy {

EmbeddedFile::EmbeddedFile(
    std::string name,
    const std::initializer_list<ExeSection>& sections,
    const std::initializer_list<InterpreterSymbol> symbols)
    : libraryName("/tmp/multipy_" + name + "XXXXXX") {
  int fd = mkstemp(&libraryName[0]);
  MULTIPY_INTERNAL_ASSERT(fd != -1, "failed to create temporary file");
  FILE* dst = fdopen(fd, "wb");
  MULTIPY_INTERNAL_ASSERT(dst);

  const char* payloadStart = nullptr;
  size_t size = 0;
  // payloadSection needs to be kept to ensure the source file is still mapped.
  multipy::optional<Section> payloadSection;
  for (const auto& s : sections) {
    payloadSection = searchForSection(s.sectionName);
    if (payloadSection != multipy::nullopt) {
      payloadStart = payloadSection->start;
      customLoader = s.customLoader;
      size = payloadSection->len;
      MULTIPY_CHECK(payloadSection.has_value(), "Missing the payload section");
      break;
    }
  }
  if (payloadStart == nullptr) {
    const char* libStart = nullptr;
    const char* libEnd = nullptr;
    for (const auto& s : symbols) {
      libStart = (const char*)dlsym(nullptr, s.startSym);
      if (libStart) {
        libEnd = (const char*)dlsym(nullptr, s.endSym);
        customLoader = s.customLoader;
        break;
      }
    }
    MULTIPY_CHECK(
        libStart != nullptr && libEnd != nullptr,
        "torch::deploy requires a build-time dependency on "
        "embedded_interpreter or embedded_interpreter_cuda, neither of which "
        "were found. name=" +
            name + " torch::cuda::is_available()=" +
            std::to_string(torch::cuda::is_available()));

    size = libEnd - libStart;
    payloadStart = libStart;
  }
  size_t written = fwrite(payloadStart, 1, size, dst);
  MULTIPY_INTERNAL_ASSERT(size == written, "expected written == size");

  fclose(dst);
}

EmbeddedFile::~EmbeddedFile() {
  unlink(libraryName.c_str());
}

} // namespace deploy
} // namespace torch
