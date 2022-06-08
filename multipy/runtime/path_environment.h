// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <string>
#include "environment.h"

namespace torch {
namespace deploy {

class PathEnvironment : public Environment {
 public:
  explicit PathEnvironment(std::string path) : path_(std::move(path)) {}
  void configureInterpreter(Interpreter* interp) override;

 private:
  std::string path_;
};

} // namespace deploy
} // namespace torch
