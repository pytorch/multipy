// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <multipy/runtime/loader.h>
#include <iostream>
#include <sstream>
#include <vector>

using torch::deploy::CustomLibrary;
using torch::deploy::CustomLibraryPtr;
using torch::deploy::SystemLibrary;

// NOLINTNEXTLINE
std::vector<CustomLibraryPtr> loaded_files_;
std::vector<CustomLibraryPtr> search_files_;
// NOLINTNEXTLINE
static void* deploy_self = nullptr;

void loadSearchFile(const char* pathname) {
  const char* args[] = {"deploy"};
  search_files_.emplace_back(CustomLibrary::create(pathname, 1, args));
  CustomLibrary& lib = *search_files_.back();
  lib.add_search_library(SystemLibrary::create(deploy_self));
  lib.add_search_library(SystemLibrary::create());
  for (auto f : search_files_) {
    if (f.get() == &lib) {
      continue;
    }
    lib.add_search_library(f);
  }
  lib.load();
}

extern "C" {

__attribute__((visibility("default"))) void deploy_set_self(void* v) {
  deploy_self = v;
}

typedef void (*dl_funcptr)();
extern "C" dl_funcptr _PyImport_FindSharedFuncptr(
    const char* prefix,
    const char* shortname,
    const char* pathname,
    FILE* fp) {
  const char* args[] = {"deploy"};
  // XXX: we have to manually flush loaded_files_ (see deploy_flush_python_libs)
  // when the manager unloads. Otherwise some libraries can live longer than
  // they are needed, and the process of unloading them might use functionality
  // that itself gets unloaded.
  loaded_files_.emplace_back(CustomLibrary::create(pathname, 1, args));
  CustomLibrary& lib = *loaded_files_.back();
  assert(deploy_self);
  lib.add_search_library(SystemLibrary::create(deploy_self));
  lib.add_search_library(SystemLibrary::create());
  for (auto f : search_files_) {
    lib.add_search_library(f);
  }
  lib.load();
  std::stringstream ss;
  ss << prefix << "_" << shortname;
  auto r = (dl_funcptr)lib.sym(ss.str().c_str()).value();
  assert(r);
  return r;
}
__attribute__((visibility("default"))) void deploy_flush_python_libs() {
  loaded_files_.clear();
  search_files_.clear();
}
}
