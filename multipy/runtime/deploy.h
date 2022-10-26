// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <c10/util/irange.h>
#include <multipy/runtime/embedded_file.h>
#include <multipy/runtime/interpreter/interpreter_impl.h>
#include <multipy/runtime/noop_environment.h>
#include <torch/csrc/api/include/torch/imethod.h>
#include <torch/csrc/jit/serialization/import.h>

#include <multipy/runtime/interpreter/Optional.hpp>
#include <cassert>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace torch {
namespace deploy {

struct ReplicatedObj;
struct InterpreterManager;
struct LoadBalancer;

struct TORCH_API InterpreterSession {
  friend struct LoadBalancer;

  explicit InterpreterSession(InterpreterSessionImpl* impl) noexcept
      : impl_(impl), manager_(nullptr) {}
  InterpreterSession(
      InterpreterSessionImpl* impl,
      InterpreterManager* manager) noexcept
      : impl_(impl), manager_(manager) {}

  /// Returns true if `obj` belongs to this `InterpreterSession`
  bool isOwner(Obj obj) {
    return impl_->isOwner(obj);
  }
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  Obj self; // when retrieved from a PythonMovable this will be set.
  InterpreterSession(InterpreterSession&&) noexcept = default;
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~InterpreterSession();

  /// `global` imports a python object from the specified module.
  /// Specifically `global` is analogous to "from `module` import `name`"
  /// in python.
  Obj global(const char* module, const char* name) {
    return impl_->global(module, name);
  }

  /// Creates an `Obj` from an `IValue`.
  Obj fromIValue(at::IValue ivalue) {
    return impl_->fromIValue(std::move(ivalue));
  }
  /// Use `ReplicatedObj InterpreterManager::createMovable(Obj obj,
  /// InterpreterSession* I)' instead. We will have no backwards compatibility
  /// guarentees for this function.
  ReplicatedObj createMovable(Obj obj);

  /// Converts a `ReplicatedObj` to an `Obj` on this InterpreterSession.
  Obj fromMovable(const ReplicatedObj& obj);

 protected:
  bool attachDeconstructorCallback(std::function<void()> func);

 private:
  friend struct ReplicatedObj;
  friend struct Package;
  friend struct InterpreterManager;
  friend struct ReplicatedObjImpl;
  inline static size_t nextObjectId_ = 0;
  std::unique_ptr<InterpreterSessionImpl> impl_;
  InterpreterManager* manager_; /// if created from one
  std::function<void()> deconstruction_callback_ = nullptr;
  PickledObject pickleObj(Obj obj);
};

/// An `Interpreter` represents an invidual subinterpreter created by
/// `torch::deploy`. It allows for the creation of `InterpreterSession` objects
/// which allow users to interact with python objects.
class TORCH_API Interpreter {
 private:
  void* handle_;
  std::unique_ptr<InterpreterImpl> pImpl_;
  InterpreterManager* manager_; /// optional if managed by one
  std::shared_ptr<Environment> env_;

  EmbeddedFile interpreterFile_;
  multipy::optional<EmbeddedFile> torchPluginFile_;

 public:
  /// Creates an Interpreter which is managed by `manager` and using the
  /// environment `env`
  Interpreter(InterpreterManager* manager, std::shared_ptr<Environment> env);

  /// Creates an Interpreter manager using environment `env` which is not tied
  /// to an Interpreter Manager.
  explicit Interpreter(std::shared_ptr<Environment> env)
      : Interpreter(nullptr, env) {}

  /// Gets a new `InterpreterSession` from this Interpreter.
  InterpreterSession acquireSession() const {
    if (manager_) {
      return InterpreterSession(pImpl_->acquireSession(), manager_);
    } else {
      return InterpreterSession(pImpl_->acquireSession());
    }
  }

  ~Interpreter();
  Interpreter(Interpreter&& rhs) noexcept
      : handle_(rhs.handle_),
        pImpl_(std::move(rhs.pImpl_)),
        manager_(rhs.manager_),
        interpreterFile_(std::move(rhs.interpreterFile_)),
        torchPluginFile_(std::move(rhs.torchPluginFile_)) {
    rhs.handle_ = nullptr;
  }

  Interpreter(const Interpreter&) = delete;
  Interpreter& operator=(const Interpreter&) = delete;
  Interpreter& operator=(Interpreter&&) = delete;
  friend struct InterpreterManager;
};

struct Package;

/// The default LoadBalancer for torch::deploy which handles allocating and
/// freeing subinterpreters.
struct TORCH_API LoadBalancer {
  /// Creates a Loadbalancer which handles `n` interpreters.
  explicit LoadBalancer(size_t n)
      : uses_(new uint64_t[8 * n]), allocated_(n), n_(n) {
    /// 8*... to avoid false sharing of atomics on the same cache line
    memset(uses_.get(), 0, 8 * n_ * sizeof(uint64_t));
  }

  /// Changes the amount of subinterpreters which is handled by the load
  /// balancer.
  void setResourceLimit(size_t n) {
    MULTIPY_INTERNAL_ASSERT(n <= allocated_);
    n_ = n;
  }

  /// Allocates an subinterpreter, and return its ID which is used to free it.
  int acquire();

  /// Frees the subinterpreter with ID `where`. This ID is returned by
  /// `LoadBalancer::acquire()`
  void free(int where);

 private:
  // NOLINTNEXTLINE(modernize-avoid-c-arrays,cppcoreguidelines-avoid-c-arrays)
  std::unique_ptr<uint64_t[]>
      uses_; /// the approximate count of the number of users of interpreter
  size_t allocated_;
  size_t n_;
};

/// An `InterpreterManager` handles the interaction of multiple subinterpreters
/// such as allocating subinterpreters, or load balancing the subinterpreters.
struct TORCH_API InterpreterManager {
  /// constructor for `InterpreterManager` which takes the number of
  /// interpreters (usually correlates to number of cores on your cpu), and a
  /// pointer to an `Environment`. The default uses the local python env.
  explicit InterpreterManager(
      size_t nInterp = 2,
      std::shared_ptr<Environment> env = std::make_shared<NoopEnvironment>());

  /// Returns a free interpreter or an arbitrary interpreter if there are
  /// none free. To ensure data safety it's best to match the number of
  /// calling threads to the size of the interpreter pool to avoid
  /// sharing an interpreter.
  InterpreterSession acquireOne() {
    int where = resources_.acquire();
    InterpreterSession I = instances_[where].acquireSession();
    I.attachDeconstructorCallback(
        [this, where]() -> void { resources_.free(where); });
    return I;
  }

  /// use to make sure something gets run on all interpreters, such as loading
  /// or unloading a model eagerly
  at::ArrayRef<Interpreter> allInstances() {
    return instances_;
  }

  /// debugging tool to control the size of the loadBalancer
  /// and change the number of interpreters on the fly
  void debugLimitInterpreters(size_t N) {
    AT_ASSERT(N <= instances_.size());
    resources_.setResourceLimit(N);
  }

  /// loads a package from a file with name `uri`
  Package loadPackage(const std::string& uri);

  /// loads a package from a `PyTorchStreamReader` or any class other which uses
  /// `ReadAdapterInterface`
  Package loadPackage(
      std::shared_ptr<caffe2::serialize::ReadAdapterInterface> reader);

  /// convience function for loading some python source code as a module across
  /// all interpreters. this can be used for writing tests of deploy that need
  /// to execute python code, or for small amounts of application logic that are
  /// best written in Python. For larger amounts of code, prefer creating and
  /// loading them as packages.
  void registerModuleSource(std::string name, std::string src) {
    registeredModuleSource_[std::move(name)] = std::move(src);
  }

  /// Util function for debugging which outputs the number of registered
  /// modules.
  size_t countRegisteredModuleSources() {
    return registeredModuleSource_.size();
  }

  /// Converts `obj` from on `InterpreterSession` I into a  `ReplicatedObj`.
  ReplicatedObj createMovable(Obj obj, InterpreterSession* I);
  InterpreterManager(const InterpreterManager&) = delete;
  InterpreterManager& operator=(const InterpreterManager&) = delete;
  InterpreterManager& operator=(InterpreterManager&&) = delete;

 private:
  friend struct Package;
  friend struct InterpreterSession;
  friend struct InterpreterSessionImpl;
  std::vector<Interpreter> instances_;
  LoadBalancer resources_;
  std::unordered_map<std::string, std::string> registeredModuleSource_;
};

struct TORCH_API ReplicatedObjImpl {
  ReplicatedObjImpl(
      size_t object_id,
      // NOLINTNEXTLINE(modernize-pass-by-value)
      PickledObject data,
      InterpreterManager* manager)
      : objectId_(object_id), data_(data), manager_(manager) {}
  // NOLINTNEXTLINE(bugprone-exception-escape)
  ~ReplicatedObjImpl();
  void unload(const Interpreter* onThisInterpreter);
  int64_t objectId_;
  PickledObject data_;
  InterpreterManager* manager_;
};

/// ReplicatedObj represents a python object that can be used on multiple
/// interpreters. Calling methods on this will pick an arbitrary interpreter
/// to run on, transfer it there if not already and run the method. A
/// replicated object can be converted to an interpreter specific `Obj`
/// using `InterpreterSession::fromMovable(ReplicatedObj)`
struct TORCH_API ReplicatedObj {
  ReplicatedObj() : pImpl_(nullptr) {}

  /// Creates a new InterpreterSession on onThisInterpreter if specified else
  /// uses an arbitrary one from InteprreterManager.
  InterpreterSession acquireSession(
      const Interpreter* onThisInterpreter = nullptr) const;
  at::IValue operator()(at::ArrayRef<at::IValue> args) const {
    auto I = acquireSession();
    return I.self(args).toIValue();
  }

  /// Invokes the Python function or class on an arbitrary interpreter with
  /// arguments given by the tuple args and named arguments given by the
  /// dictionary kwargs (equivalent to python's `__call__`).
  [[nodiscard]] at::IValue callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) const {
    auto I = acquireSession();
    return I.self.callKwargs(std::move(args), std::move(kwargs)).toIValue();
  }

  /// Invokes the Python function or class on an arbitrary  interpreter.with
  /// named arguments given by the dictionary kwargs (equivalent to python's
  /// `__call__`).
  [[nodiscard]] at::IValue callKwargs(
      std::unordered_map<std::string, c10::IValue> kwargs) const {
    auto I = acquireSession();
    return I.self.callKwargs(std::move(kwargs)).toIValue();
  }

  /// Returns true if `ReplicatedObj` has attribute with name `attr` and false
  /// otherwise. This is done on an arbitrary `InterpreterSession` which belongs
  /// to the `ReplicatedObj`'s manager.
  [[nodiscard]] bool hasattr(const char* attr) const {
    auto I = acquireSession();
    return I.self.hasattr(attr);
  }

  /// Deletes `ReplicatedObj` from onThisInterpreter, if onThisInterpreter is
  /// `nullptr`, unload is called on all interpreters belonging to the
  /// ReplicatedObject's InterpreterManager
  void unload(const Interpreter* onThisInterpreter = nullptr);

  /// Converts `ReplicatedObj` to `Obj` on `InterpreterSession` `I`
  Obj toObj(InterpreterSession* I);

 private:
  ReplicatedObj(std::shared_ptr<ReplicatedObjImpl> pImpl)
      : pImpl_(std::move(pImpl)) {}
  std::shared_ptr<ReplicatedObjImpl> pImpl_;
  friend struct Package;
  friend struct InterpreterSession;
  friend struct InterpreterManager;
};

/// PythonMethodWrapper is a more specific instance of a
/// ReplicatedObj which represents a python method, and
/// is therefore callable and has argument names accessible.
class PythonMethodWrapper : public torch::IMethod {
 public:
  // TODO(whc) make bound method pickleable, then directly construct from that

  PythonMethodWrapper(
      torch::deploy::ReplicatedObj model,
      std::string methodName)
      : model_(std::move(model)), methodName_(std::move(methodName)) {}

  /// return the name of the python method.
  const std::string& name() const override {
    return methodName_;
  }

  /// overrides the `()` operater to call the underlying python method.
  c10::IValue operator()(
      std::vector<c10::IValue> args,
      const IValueMap& kwargs = IValueMap()) const override {
    // TODO(whc) ideally, pickle the method itself as replicatedobj, to skip
    // this lookup each time
    auto modelSession = model_.acquireSession();
    auto method = modelSession.self.attr(methodName_.c_str());
    return method.callKwargs(args, kwargs).toIValue();
  }

 private:
  void setArgumentNames(std::vector<std::string>&) const override;

  torch::deploy::ReplicatedObj model_;
  std::string methodName_;
};

/// Package is a wrapper around `torch.package` which allows loading a
/// PyTorch model and its dependencies from a package.
struct TORCH_API Package {
  /// shorthand for getting the object as a pickle resource in the package
  ReplicatedObj loadPickle(const std::string& module, const std::string& file) {
    auto I = acquireSession();
    auto loaded = I.self.attr("load_pickle")({module, file});
    return createMovable(loaded, &I);
  }

#ifdef FBCODE_CAFFE2
  std::string loadText(const std::string& packageName, const std::string& key) {
    auto I = acquireSession();
    return I.self.attr("load_text")({packageName, key})
        .toIValue()
        .toStringRef();
  }

  // Example usage:
  //  in python:
  //    with PackageExporter(output) as pe:
  //        pe.save_binary("extra_files", "greeting", b'hello')
  //  in cpp:
  //    std::string decodedBinary = package->loadBinary("extra_files",
  //    "greeting").toStringRef();
  //    std::cout << decodedBinary; --> outputs "hello"
  std::string loadBinary(
      const std::string& packageName,
      const std::string& key) {
    auto I = acquireSession();
    return I.self.attr("load_binary")({packageName, key})
        .toIValue()
        .toStringRef();
  }
#endif

  /// Allocates an `InterpreterSession` and load the appropriate torch.package
  /// with it.
  InterpreterSession acquireSession() {
    auto I = manager_->acquireOne();
    I.self =
        I.impl_->createOrGetPackageImporterFromContainerFile(containerFile_);
    return I;
  }

  /// Converts an `Obj` from `InterpreterSession` `I` into a `ReplicatedObj`.
  ReplicatedObj createMovable(Obj obj, InterpreterSession* I) {
    return manager_->createMovable(obj, I);
  }

 private:
  Package(
      const std::string& uri,
      InterpreterManager*
          pm) // or really any of the constructors to our zip file format
      : manager_(pm),
        containerFile_(
            std::make_shared<caffe2::serialize::PyTorchStreamReader>(uri)) {}
  Package(
      std::shared_ptr<caffe2::serialize::ReadAdapterInterface> reader,
      InterpreterManager*
          pm) // or really any of the constructors to our zip file format
      : manager_(pm),
        containerFile_(
            std::make_shared<caffe2::serialize::PyTorchStreamReader>(reader)) {}
  friend struct ReplicatedObj;
  friend struct InterpreterManager;
  InterpreterManager* manager_;

  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> containerFile_;
};

} // namespace deploy
} // namespace torch

namespace multipy {
namespace runtime = torch::deploy;
}
