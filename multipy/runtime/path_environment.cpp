// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <multipy/runtime/deploy.h>
#include <multipy/runtime/path_environment.h>

namespace torch {
namespace deploy {

void PathEnvironment::configureInterpreter(Interpreter* interp) {
  auto I = interp->acquireSession();
  I.global("sys", "path").attr("append")({path_});
}

} // namespace deploy
} // namespace torch
