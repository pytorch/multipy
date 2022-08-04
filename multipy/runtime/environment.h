// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <multipy/runtime/Exception.h>
#include <multipy/runtime/elf_file.h>
#include <string>

namespace torch {
namespace deploy {

class Interpreter;

/*
 * An environment is the concept to decribe the circumstances in which a
 * torch::deploy interpreter runs. In can be an xar file embedded in the binary,
 * a filesystem path for the installed libraries etc.
 */
class Environment {
  std::vector<std::string> extraPythonPaths_;
  // all zipped python libraries will be written
  // under this directory
  std::string extraPythonLibrariesDir_;
  std::string getZippedArchive(
      const char* zipped_torch_name,
      const std::string& pythonAppDir) {
    std::string execPath;
    std::ifstream("/proc/self/cmdline") >> execPath;
    ElfFile elfFile(execPath.c_str());
    // load the zipped torch modules
    auto zippedTorchSection = elfFile.findSection(zipped_torch_name);
    MULTIPY_CHECK(
        zippedTorchSection.has_value(), "Missing the zipped torch section");
    const char* zippedTorchStart = zippedTorchSection->start;
    auto zippedTorchSize = zippedTorchSection->len;

    std::string zipArchive = pythonAppDir;
    auto zippedFile = fopen(zipArchive.c_str(), "wb");
    MULTIPY_CHECK(
        zippedFile != nullptr, "Fail to create file: ", strerror(errno));
    fwrite(zippedTorchStart, 1, zippedTorchSize, zippedFile);
    fclose(zippedFile);
    return zipArchive;
  }
  void setupZippedPythonModules(const std::string& pythonAppDir) {
#ifdef FBCODE_CAFFE2
    extraPythonPaths_.push_back(getZippedArchive(
        ".torch_python_modules",
        std::string(pythonAppDir) + "/torch_python_modules.zip"));
    extraPythonPaths_.push_back(getZippedArchive(
        ".multipy_python_modules",
        std::string(pythonAppDir) + "/multipy_python_modules.zip"));

#endif
    extraPythonLibrariesDir_ = pythonAppDir;
  }

 public:
  explicit Environment() {
    char tempDirName[] = "/tmp/torch_deploy_zipXXXXXX";
    char* tempDirectory = mkdtemp(tempDirName);
    setupZippedPythonModules(tempDirectory);
  }
  explicit Environment(const std::string& pythonAppDir) {
    setupZippedPythonModules(pythonAppDir);
  }
  virtual ~Environment() {
    auto rmCmd = "rm -rf " + extraPythonLibrariesDir_;
    system(rmCmd.c_str());
  }
  virtual void configureInterpreter(Interpreter* interp) = 0;
  virtual const std::vector<std::string>& getExtraPythonPaths() {
    return extraPythonPaths_;
  }
};

} // namespace deploy
} // namespace torch
