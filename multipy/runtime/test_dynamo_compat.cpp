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
#include <libgen.h>
#include <multipy/runtime/deploy.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <future>
#include <iostream>
#include <string>

void compare_torchpy_jit(const char* model_filename, const char* jit_filename) {
  // Test

  torch::deploy::InterpreterManager m(2);
  torch::deploy::Package p = m.loadPackage(model_filename);
  auto model = p.loadPickle("model", "model.pkl");
  at::IValue eg;
  {
    auto I = p.acquireSession();
    eg = I.self.attr("load_pickle")({"model", "example.pkl"}).toIValue();
  }
  auto I = p.acquireSession();
  auto cModelObj = I.global("torch", "compile")(model.toObj(&I));
  auto cModel = m.createMovable(cModelObj, &I);
  at::Tensor output = cModel(eg.toTupleRef().elements()).toTensor();

  // Reference
  auto ref_model = torch::jit::load(jit_filename);
  at::Tensor ref_output =
      ref_model.forward(eg.toTupleRef().elements()).toTensor();

  ASSERT_TRUE(ref_output.allclose(output, 1e-03, 1e-05));
}

const char* simple = "multipy/runtime/example/generated/simple";
const char* simple_jit = "multipy/runtime/example/generated/simple_jit";

const char* path(const char* envname, const char* path) {
  const char* e = getenv(envname);
  return e ? e : path;
}

TEST(TorchpyTest, SimpleModel) {
  compare_torchpy_jit(path("SIMPLE", simple), path("SIMPLE_JIT", simple_jit));
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
