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

  at::Tensor output = model(eg.toTupleRef().elements()).toTensor();

  // Reference
  auto ref_model = torch::jit::load(jit_filename);
  at::Tensor ref_output =
      ref_model.forward(eg.toTupleRef().elements()).toTensor();
  ASSERT_TRUE(ref_output.allclose(output, 1e-03, 1e-05));
}
const char* resnet_path = "multipy/runtime/example/generated/resnet";
// const char* resnet_path = "multipy/runtime/example/generated/resnet_dynamo";
const char* resnet_jit_path = "multipy/runtime/example/generated/resnet_jit";

const char* path(const char* envname, const char* path) {
  const char* e = getenv(envname);
  return e ? e : path;
}

TEST(TorchpyTest, ResNetWithDynamo) {
  compare_torchpy_jit(
      path("RESNET", resnet_path),
      path("RESNET_JIT", resnet_jit_path));
}

TEST(TorchpyTest, ThreadedResnetModelWithDynamo) {
  size_t nthreads = 3;
  torch::deploy::InterpreterManager manager(nthreads);

  torch::deploy::Package p = manager.loadPackage(path("RESNET", resnet_path));
  auto model = p.loadPickle("model", "model.pkl");
  auto ref_model = torch::jit::load(path("RESNET_JIT", resnet_jit_path));

  auto input = torch::ones({10, 20});

  std::vector<at::Tensor> outputs;

  std::vector<std::future<at::Tensor>> futures;
  for (const auto i : c10::irange(nthreads)) {
    (void)i;
    futures.push_back(std::async(std::launch::async, [&model]() {
      auto input = torch::ones({10, 10, 10});
      for (const auto j : c10::irange(100)) {
        (void)j;
        model({input.alias()}).toTensor();
      }
      auto result = model({input.alias()}).toTensor();
      return result;
    }));
  }
  for (const auto i : c10::irange(nthreads)) {
    outputs.push_back(futures[i].get());
  }

  // Generate reference
  auto ref_output = ref_model.forward({input.alias()}).toTensor();

  // Compare all to reference
  for (const auto i : c10::irange(nthreads)) {
    ASSERT_TRUE(ref_output.equal(outputs[i]));
  }
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
