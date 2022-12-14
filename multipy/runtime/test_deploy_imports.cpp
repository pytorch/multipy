// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <ATen/Parallel.h>
#include <gtest/gtest.h>
#include <libgen.h>
#include <cstring>

#include <c10/util/irange.h>
#include <multipy/runtime/deploy.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <future>
#include <iostream>
#include <string>

void test_import(std::vector<const char*> moduleNames) {
  // Test whether importing the python modules specified
  // in "moduleNames" work inside a deploy interpreter

  torch::deploy::InterpreterManager manager(1);
  auto I = manager.acquireOne();
  for (const char* moduleName : moduleNames) {
    I.global("builtins", "__import__")({moduleName});
  }
}

TEST(TorchpyTest, TestImportTorch) {
  test_import({"torch"});
}

TEST(TorchpyTest, TestImportFSDP) {
  test_import({"torch.distributed.fsdp"});
}

TEST(TorchpyTest, TestImportTorchGen) {
  test_import({"torchgen"});
}

TEST(TorchpyTest, TestImportMultiPy) {
  test_import({"multipy"});
}

TEST(TorchpyTest, TestImportSympy) {
  test_import({"mpmath", "sympy"});
}

int main(int argc, char* argv[]) {
  ::testing::InitGoogleTest(&argc, argv);
  char tempeh[256];
  getcwd(tempeh, 256);
  std::cout << "Current working directory: " << tempeh << std::endl;
  int rc = RUN_ALL_TESTS();
  char tmp[256];
  getcwd(tmp, 256);
  std::cout << "Current working directory: " << tmp << std::endl;
  return rc;
}
