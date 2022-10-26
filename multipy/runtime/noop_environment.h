// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <multipy/runtime/environment.h>

namespace torch {
namespace deploy {

/// The local python Environment
class NoopEnvironment : public Environment {
 public:
  /// no-op function as this is the no-op environment :)
  void configureInterpreter(Interpreter* /* interp */) override {}
};

} // namespace deploy
} // namespace torch
