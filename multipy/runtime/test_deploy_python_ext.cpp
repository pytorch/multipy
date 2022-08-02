// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <multipy/runtime/deploy.h>
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cstdio>
#include <iostream>

bool run() {
  torch::deploy::InterpreterManager m(2);
  m.registerModuleSource("check_none", "check = id(None)\n");
  int64_t id0 = 0, id1 = 0;
  {
    auto I = m.allInstances()[0].acquireSession();
    id0 = I.global("check_none", "check").toIValue().toInt();
  }
  {
    auto I = m.allInstances()[1].acquireSession();
    id1 = I.global("check_none", "check").toIValue().toInt();
  }
  return id0 != id1;
}

PYBIND11_MODULE(test_deploy_python_ext, m) {
  m.def("run", run);
}
