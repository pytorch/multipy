// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <dlfcn.h>
#include <libgen.h>
#include <multipy/runtime/Exception.h>
#include <multipy/runtime/deploy.h>
#include <unistd.h>
#include <functional>

#include <multipy/runtime/interpreter/Optional.hpp>

// these symbols are generated by cmake, using ld -r -b binary
// libtorch_deployinterpreter.so which takes the contents of the so and embeds
// it into a symbol that is then linked into libtorch_deploy.so. This enables us
// to simply copy the contents of this symbol to disk and dlopen it to create an
// instance of python.

namespace torch {
namespace deploy {

const std::initializer_list<ExeSection> pythonInterpreterSections = {
    {".torch_deploy_payload.interpreter_all", true},
    {".torch_deploy_payload.interpreter_cuda", false},
    {".torch_deploy_payload.interpreter_cpu", false},
    {".torch_deploy_payload.interpreter_hip", false},
};

const std::initializer_list<InterpreterSymbol> pythonInterpreterSymbols = {
    {"_binary_libtorch_deployinterpreter_all_so_start",
     "_binary_libtorch_deployinterpreter_all_so_end",
     true},
    {"_binary_libtorch_deployinterpreter_cuda_so_start",
     "_binary_libtorch_deployinterpreter_cuda_so_end",
     false},
    {"_binary_libtorch_deployinterpreter_cpu_so_start",
     "_binary_libtorch_deployinterpreter_cpu_so_end",
     false},
    {"_binary_libtorch_deployinterpreter_hip_so_start",
     "_binary_libtorch_deployinterpreter_hip_so_end",
     false},
};
const std::initializer_list<ExeSection> multipyTorchSections = {
    {".torch_deploy_payload.multipy_torch", false},
};
const std::initializer_list<InterpreterSymbol> multipyTorchSymbols = {};

InterpreterManager::InterpreterManager(
    size_t nInterp,
    std::shared_ptr<Environment> env)
    : resources_(nInterp) {
  // disable GIL deadlock detection if it's not set already
  setenv("TORCH_DISABLE_DEADLOCK_DETECTION", "1", /*overwrite*/ 0);
  // disable prims/torch.Library support
  setenv("PYTORCH_DISABLE_LIBRARY", "1", /*overwrite*/ 0);

  for (const auto i : c10::irange(nInterp)) {
#ifdef FBCODE_CAFFE2
    instances_.emplace_back(this, env);
#else
    instances_.emplace_back(env);
#endif
    auto I = instances_.back().acquireSession();
    // make torch.version.interp be the interpreter id
    // can be used for balancing work across GPUs
    I.global("torch", "version").attr("__setattr__")({"interp", int(i)});
    instances_.back().pImpl_->setFindModule(
        [this](const std::string& name) -> multipy::optional<std::string> {
          auto it = registeredModuleSource_.find(name);
          if (it != registeredModuleSource_.end()) {
            return it->second;
          } else {
            return multipy::nullopt;
          }
        });
  }

  // Pre-registered modules.
  // Since torch::deploy::Obj.toIValue cannot infer empty list, we hack it to
  // return None for empty list.
  // TODO(jwtan): Make the discovery of these modules easier.
  registerModuleSource(
      "GetArgumentNamesModule",
      "from inspect import signature\n"
      "from typing import Callable, Optional\n"
      "def getArgumentNames(function: Callable) -> Optional[list]:\n"
      "    names = list(signature(function).parameters.keys())\n"
      "    if len(names) == 0:\n"
      "        return None\n"
      "    return names\n");
}

Package InterpreterManager::loadPackage(const std::string& uri) {
  return Package(uri, this);
}

Package InterpreterManager::loadPackage(
    std::shared_ptr<caffe2::serialize::ReadAdapterInterface> reader) {
  return Package(reader, this);
}

Obj InterpreterSession::fromMovable(const ReplicatedObj& obj) {
  return impl_->unpickleOrGet(obj.pImpl_->objectId_, obj.pImpl_->data_);
}

InterpreterSession ReplicatedObj::acquireSession(
    const Interpreter* onThisInterpreter) const {
  MULTIPY_CHECK(
      (pImpl_->manager_ || onThisInterpreter),
      "ReplicatedObjImpl needs an interpreter or needs to be associated with an InterpreterManager (with ReplicatedObj::attachInterpreterManager) to create an InterpreterSession.");
  InterpreterSession I = onThisInterpreter ? onThisInterpreter->acquireSession()
                                           : pImpl_->manager_->acquireOne();
  I.self = I.fromMovable(*this);
  return I;
}

void ReplicatedObj::attachInterpreterManager(InterpreterManager* manager) {
  MULTIPY_CHECK(
      !pImpl_->manager_,
      "ReplicatedObjImpl must not be associated with an interpreter manager to attach it to one.");
  pImpl_->manager_ = manager;
}

bool InterpreterSession::attachDeconstructorCallback(
    std::function<void(void)> func) {
  if (deconstruction_callback_) {
    return false;
  }
  deconstruction_callback_ = func;
  return true;
}

// NOLINTNEXTLINE(bugprone-exception-escape)
InterpreterSession::~InterpreterSession() {
  if (deconstruction_callback_ != NULL) {
    deconstruction_callback_();
  }
}

void ReplicatedObjImpl::unload(const Interpreter* onThisInterpreter) {
  if (!onThisInterpreter) {
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    MULTIPY_CHECK(
        manager_,
        "ReplicatedObjImpl must be created from an InterpreterManager in order to unload without an interpreter");
    for (auto& interp : manager_->allInstances()) {
      unload(&interp);
    }
    return;
  }

  InterpreterSession I = onThisInterpreter->acquireSession();
  I.impl_->unload(objectId_);
}

// NOLINTNEXTLINE(bugprone-exception-escape)
ReplicatedObjImpl::~ReplicatedObjImpl() {
  unload(nullptr);
}

void ReplicatedObj::unload(const Interpreter* onThisInterpreter) {
  pImpl_->unload(onThisInterpreter);
}

ReplicatedObj InterpreterSession::createMovable(Obj obj) {
  MULTIPY_CHECK(
      manager_,
      "Can only create a movable object when the session was created "
      "from an interpreter that is part of a InterpreterManager");

  MULTIPY_CHECK(
      impl_->isOwner(obj),
      "Cannot create movable from an object that lives in different session");

// Fully deprecate after moving over internal users to new API, currently here
// to keep bc with old API.
#ifdef FBCODE_CAFFE2
  if (manager_) {
    return manager_->createMovable(obj, this);
  }
#endif

  auto pickled = impl_->pickle(self, obj);
  return ReplicatedObj(
      std::make_shared<ReplicatedObjImpl>(nextObjectId_++, std::move(pickled)));
}

ReplicatedObj InterpreterManager::createMovable(
    Obj obj,
    InterpreterSession* I) {
  MULTIPY_CHECK(
      I->isOwner(obj),
      "Cannot create movable from an object that lives in different session");
  PickledObject pickled = I->pickleObj(obj);
  return ReplicatedObj(std::make_shared<ReplicatedObjImpl>(
      I->nextObjectId_++, std::move(pickled), this));
}

PickledObject InterpreterSession::pickleObj(Obj obj) {
  MULTIPY_CHECK(
      impl_->isOwner(obj),
      "Cannot pickle an object that lives in different session");
  return impl_->pickle(self, obj);
}

using dlopen_t = void* (*)(const char*, int);

// ASAN overrides dlopen and errors when it sees the RTLD_DEEPBIND flags because
// it thinks that the library being loaded will not link against its overrides
// for things like malloc/free. However, our specially crafted library doesn't
// have any DT_NEEDED entries -- all undefined symbols will be resolved from the
// process's link map. So it is actually safe to use RTLD_DEEPBIND with ASAN. We
// have to get around its check though, so we do it by finding the real dlopen
// function.
static dlopen_t find_real_dlopen() {
  void* libc = dlopen("libdl.so.2", RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  // libdl is gone on some newer systems.
  if (!libc) {
    // libc.so won't open with dlopen because it's a linker script.
    libc = dlopen("libc.so.6", RTLD_NOLOAD | RTLD_LAZY | RTLD_LOCAL);
  }
  TORCH_INTERNAL_ASSERT(libc);
  auto dlopen_ = (dlopen_t)dlsym(libc, "dlopen");
  TORCH_INTERNAL_ASSERT(dlopen_);
  return dlopen_;
}

Interpreter::Interpreter(
    InterpreterManager* manager,
    std::shared_ptr<Environment> env)
    : handle_(nullptr),
      manager_(manager),
      env_(env),
      interpreterFile_(
          "interpreter",
          pythonInterpreterSections,
          pythonInterpreterSymbols) {
  setUpInterpreter();
}
Interpreter::Interpreter(std::shared_ptr<Environment> env)
    : handle_(nullptr),
      manager_(nullptr),
      env_(env),
      interpreterFile_(
          "interpreter",
          pythonInterpreterSections,
          pythonInterpreterSymbols) {
  setUpInterpreter();
}

void Interpreter::setUpInterpreter() {
  int flags = RTLD_LOCAL | RTLD_LAZY;
  if (interpreterFile_.customLoader) {
    flags |= RTLD_DEEPBIND;
  }

#ifdef FBCODE_CAFFE2
  static dlopen_t dlopen_ = find_real_dlopen();
  handle_ = dlopen_(interpreterFile_.libraryName.c_str(), flags);
#else
  handle_ = dlopen(interpreterFile_.libraryName.c_str(), flags);
#endif

  if (!handle_) {
    throw std::runtime_error(dlerror());
  }

  if (interpreterFile_.customLoader) {
    // when using the custom loader we need to link python symbols against
    // the right version of the symbols for the interpreter which an be looked
    // up from the handle_ to this shared library. here we register the handle
    // with the code that does custom loading of python extensions.
    auto deploySetSelfPtr = (void (*)(void*))dlsym(handle_, "deploy_set_self");
    AT_ASSERT(deploySetSelfPtr);
    deploySetSelfPtr(handle_);
  }

  std::vector<std::string> pluginPaths;
#ifndef FBCODE_CAFFE2
  torchPluginFile_.emplace(
      "multipy_torch", multipyTorchSections, multipyTorchSymbols);
  pluginPaths.emplace_back(torchPluginFile_->libraryName);
#endif

  auto extraPythonPaths = env_->getExtraPythonPaths();
  void* newInterpreterImpl = dlsym(handle_, "newInterpreterImpl");
  AT_ASSERT(newInterpreterImpl);
  pImpl_ = std::unique_ptr<InterpreterImpl>(
      ((InterpreterImpl *
        (*)(const std::vector<std::string>&, const std::vector<std::string>&))
           newInterpreterImpl)(extraPythonPaths, pluginPaths));
  env_->configureInterpreter(this);
}

Interpreter::~Interpreter() {
  if (handle_) {
    // ensure python uninitialization runs before we dlclose the library
    pImpl_.reset();
    if (interpreterFile_.customLoader) {
      auto deploy_flush_python_libs =
          (void (*)())dlsym(handle_, "deploy_flush_python_libs");
      deploy_flush_python_libs();
    }
    dlclose(handle_);
  }
}

int LoadBalancer::acquire() {
  thread_local int last = 0;
  size_t minusers = SIZE_MAX;
  int minIdx = 0;
  for (size_t i = 0; i < n_; ++i, ++last) {
    if (last >= static_cast<int>(n_)) {
      last = 0;
    }
    uint64_t prev = 0;
    bool acquired = __atomic_compare_exchange_n(
        &uses_[8 * last],
        &prev,
        1ULL,
        false,
        __ATOMIC_SEQ_CST,
        __ATOMIC_SEQ_CST);
    if (acquired) {
      // fast path, we found an interpreter with no users
      return last;
    }
    // slow path, we don't want to use this interpreter because it is being
    // used by someone else.

    if (prev < minusers) {
      minusers = prev;
      minIdx = last;
    }
  }
  // we failed to find a completely free interpreter. heuristically use the
  // one with the least number of user (note that this may have changed since
  // then, so this is only a heuristic).
  __atomic_fetch_add(&uses_[8 * minIdx], 1ULL, __ATOMIC_SEQ_CST);
  return minIdx;
}

void LoadBalancer::free(int where) {
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-magic-numbers)
  __atomic_fetch_sub(&uses_[8 * where], 1ULL, __ATOMIC_SEQ_CST);
}

void PythonMethodWrapper::setArgumentNames(
    std::vector<std::string>& argumentNamesOut) const {
  auto session = model_.acquireSession();
  auto method = session.self.attr(methodName_.c_str());
  auto iArgumentNames =
      session.global("GetArgumentNamesModule", "getArgumentNames")({method})
          .toIValue();
  if (iArgumentNames.isNone()) {
    return;
  }

  TORCH_INTERNAL_ASSERT(iArgumentNames.isList());
  auto argumentNames = iArgumentNames.toListRef();

  argumentNamesOut.reserve(argumentNames.size());
  for (auto& argumentName : argumentNames) {
    TORCH_INTERNAL_ASSERT(argumentName.isString());
    argumentNamesOut.push_back(argumentName.toStringRef());
  }
}

} // namespace deploy
} // namespace torch
