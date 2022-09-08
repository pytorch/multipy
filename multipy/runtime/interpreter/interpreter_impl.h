// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>
#include <multipy/runtime/interpreter/Optional.hpp>
#include <unordered_map>
#include <set>


/* Torch Deploy intentionally embeds multiple copies of c++ libraries
   providing python bindings necessary for torch::deploy users in the same
   process space in order to provide a multi-python environment.  As a result,
   any exception types defined by these duplicated libraries can't be safely
   caught or handled outside of the originating dynamic library (.so).

   In practice this means that you must either
   catch these exceptions inside the torch::deploy API boundary or risk crashing
   the client application.

   It is safe to throw exception types that are defined once in
   the context of the client application, such as std::runtime_error,
   which isn't duplicated in torch::deploy interpreters.

   ==> Use TORCH_DEPLOY_TRY, _SAFE_CATCH_RETHROW around _ALL_ torch::deploy APIs

   For more information, see
    https://gcc.gnu.org/wiki/Visibility (section on c++ exceptions)
    or https://stackoverflow.com/a/14364055
    or
   https://stackoverflow.com/questions/14268736/symbol-visibility-exceptions-runtime-error
    note- this may be only a serious problem on versions of gcc prior to 4.0,
   but still seems worth sealing off.

*/
#define TORCH_DEPLOY_TRY try {
#define TORCH_DEPLOY_SAFE_CATCH_RETHROW                                     \
  }                                                                         \
  catch (std::exception & err) {                                            \
    throw std::runtime_error(                                               \
        std::string(                                                        \
            "Exception Caught inside torch::deploy embedded library: \n") + \
        err.what());                                                        \
  }                                                                         \
  catch (...) {                                                             \
    throw std::runtime_error(std::string(                                   \
        "Unknown Exception Caught inside torch::deploy embedded library")); \
  }
namespace torch {
namespace deploy {

struct InterpreterSessionImpl;

struct PickledObject {
  std::string data_;
  std::vector<at::Storage> storages_;
  // types for the storages, required to
  // reconstruct correct Python storages
  std::vector<at::ScalarType> types_;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> containerFile_;
};

struct InterpreterObj {
  friend struct InterpreterSessionImpl;
  friend struct Obj;

  private:
    virtual at::IValue toIValue() const;
    virtual InterpreterObj call(at::ArrayRef<InterpreterObj> args);
    virtual InterpreterObj call(at::ArrayRef<c10::IValue> args);
    virtual InterpreterObj callKwargs(
        std::vector<at::IValue> args,
        std::unordered_map<std::string, c10::IValue> kwargs);
    virtual InterpreterObj callKwargs(std::unordered_map<std::string, c10::IValue> kwargs);
    virtual bool hasattr(const char* attr);
    virtual InterpreterObj attr(const char* attr);
};

struct Obj {
  friend struct InterpreterSessionImpl;
  friend struct InterpreterObj;
  Obj(InterpreterObj* baseObj)
      : baseObj_(baseObj){}

  at::IValue toIValue() const;
  Obj operator()(at::ArrayRef<Obj> args);
  Obj operator()(at::ArrayRef<at::IValue> args);
  Obj callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs);
  Obj callKwargs(std::unordered_map<std::string, c10::IValue> kwargs);
  bool hasattr(const char* attr);
  Obj attr(const char* attr);
  InterpreterObj* baseObj_;
};

struct InterpreterSessionImpl {
  friend struct Package;
  friend struct ReplicatedObj;
  friend struct Obj;
  friend struct InterpreterObj;
  friend struct InterpreterSession;
  friend struct ReplicatedObjImpl;

  virtual ~InterpreterSessionImpl() = default;

 private:
  virtual Obj global(const char* module, const char* name) = 0;
  virtual Obj fromIValue(at::IValue value) = 0;
  virtual Obj createOrGetPackageImporterFromContainerFile(
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          containerFile_) = 0;
  virtual PickledObject pickle(Obj container, Obj obj) = 0;
  virtual Obj unpickleOrGet(int64_t id, const PickledObject& obj) = 0;

  virtual at::IValue toIValue(Obj obj) const = 0;

  virtual Obj call(Obj obj, at::ArrayRef<Obj> args) = 0;
  virtual Obj call(Obj obj, at::ArrayRef<at::IValue> args) = 0;
  virtual void unload(int64_t id);
  virtual Obj callKwargs(
      Obj obj,
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual Obj callKwargs(
      Obj obj,
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual Obj attr(Obj obj, const char* attr) = 0;
  virtual bool hasattr(Obj obj, const char* attr) = 0;
  virtual bool isOwner(Obj obj) = 0;
};

struct InterpreterImpl {
  virtual InterpreterSessionImpl* acquireSession() = 0;
  virtual void setFindModule(
      std::function<multipy::optional<std::string>(const std::string&)>
          find_module) = 0;
  virtual ~InterpreterImpl() = default; // this will uninitialize python
};


// inline definitions for Objs are necessary to avoid introducing a
// source file that would need to exist it both the libinterpreter.so and then
// the libtorchpy library.
inline at::IValue Obj::toIValue() const {
  return baseObj_->toIValue();
}

inline Obj Obj::operator()(at::ArrayRef<Obj> args) {
  std::vector<InterpreterObj> iArgs;
  for (size_t i = 0, N = args.size(); i != N; ++i) {
    iArgs.emplace_back(args[i].baseObj_);
  }
  InterpreterObj iObj =  baseObj_->call(iArgs);
  return Obj(&iObj);
}

inline Obj Obj::operator()(at::ArrayRef<at::IValue> args) {
  InterpreterObj iObj = baseObj_->call(args);
  return Obj(&iObj);
}

inline Obj Obj::callKwargs(
    std::vector<at::IValue> args,
    std::unordered_map<std::string, c10::IValue> kwargs) {
  InterpreterObj iObj =  baseObj_->callKwargs(std::move(args), std::move(kwargs));
  return Obj(&iObj);
}
inline Obj Obj::callKwargs(
    std::unordered_map<std::string, c10::IValue> kwargs) {
  InterpreterObj iObj =  baseObj_->callKwargs(std::move(kwargs));
  return Obj(&iObj);
}
inline bool Obj::hasattr(const char* attr) {
  return baseObj_->hasattr(attr);
}

inline Obj Obj::attr(const char* attr) {
  InterpreterObj iObj =  baseObj_->attr(attr);
  return Obj(&iObj);
}

} // namespace deploy
} // namespace torch
