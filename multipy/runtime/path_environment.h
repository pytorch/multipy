// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <multipy/runtime/environment.h>
#include <string>

namespace torch {
namespace deploy {

/// An Environment which is defined by a specific path to python code (ie.
/// condas sitepackages)
class PathEnvironment : public Environment {
 public:
  /// Environment constructor which takes a file name for the
  /// directory for the python modules.
  explicit PathEnvironment(std::string path) : path_(std::move(path)) {}
  /// Adds the path defined in the `PathEnvironment` to `interp`
  void configureInterpreter(Interpreter* interp) override;

 private:
  std::string path_;
};

} // namespace deploy
} // namespace torch
