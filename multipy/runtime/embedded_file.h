// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>

namespace torch {
namespace deploy {

// Represents an ExeSection of an EmbeddedFile.
struct ExeSection {
  const char *sectionName;
  bool customLoader;
};

// These are symbols used by the subinterpreters.
struct InterpreterSymbol {
  const char *startSym;
  const char *endSym;
  bool customLoader;
};

// Represents an EmbeddedFile which is a file which contains a binary for a
// subinterprerter.
struct EmbeddedFile {
  std::string libraryName{""};
  bool customLoader{false};

  EmbeddedFile(std::string name,
               const std::initializer_list<ExeSection> &sections,
               const std::initializer_list<InterpreterSymbol> symbols);

  ~EmbeddedFile();

  EmbeddedFile &operator=(const EmbeddedFile &) = delete;
};

} // namespace deploy
} // namespace torch
