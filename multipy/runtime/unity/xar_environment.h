// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <multipy/runtime/deploy.h>
#include <multipy/runtime/environment.h>
#include <string>

namespace torch {
namespace deploy {

constexpr const char* DEFAULT_PYTHON_APP_DIR = "/tmp/torch_deploy_python_app";

class XarEnvironment : public Environment {
 public:
  explicit XarEnvironment(
      std::string exePath,
      std::string pythonAppDir = DEFAULT_PYTHON_APP_DIR);
  ~XarEnvironment() override;

 protected:
  void configureInterpreter(Interpreter* interp) override;

 private:
  void setupPythonApp();
  void preloadSharedLibraries();

  std::string exePath_;
  std::string pythonAppDir_;
  std::string pythonAppRoot_;
  bool alreadySetupPythonApp_ = false;
};

} // namespace deploy
} // namespace torch
