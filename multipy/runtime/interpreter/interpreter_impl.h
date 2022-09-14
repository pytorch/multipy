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

namespace torch {
namespace deploy {
struct Obj;
struct InterpreterSessionImpl;
struct InterpreterSession;

struct PickledObject {
  std::string data_;
  std::vector<at::Storage> storages_;
  // types for the storages, required to
  // reconstruct correct Python storages
  std::vector<at::ScalarType> types_;
  std::shared_ptr<caffe2::serialize::PyTorchStreamReader> containerFile_;
};

struct InterpreterObj {
  friend struct Obj;
  friend struct ReplicatedObjImpl;
  public:
    InterpreterObj() = default;
    InterpreterObj(const InterpreterObj& obj) = default;
    InterpreterObj(InterpreterObj&& obj) = default;
    // virtual ~InterpreterObj() = default;
  private:

    virtual at::IValue toIValue() const = 0;
    virtual Obj call(at::ArrayRef<Obj> args) = 0;
    virtual Obj call(at::ArrayRef<at::IValue> args) = 0;
    virtual Obj callKwargs(
        std::vector<at::IValue> args,
        std::unordered_map<std::string, c10::IValue> kwargs) = 0;
    virtual Obj callKwargs(std::unordered_map<std::string, c10::IValue> kwargs) = 0;
    virtual bool hasattr(const char* attr) = 0;
    virtual Obj attr(const char* attr) = 0;
};

struct Obj {
  friend struct InterpreterObj;
  Obj(InterpreterObj* baseObj)
      : baseObj_(baseObj){}
  Obj() : baseObj_(nullptr) {}

  at::IValue toIValue() const;
  Obj operator()(at::ArrayRef<Obj> args);
  Obj operator()(at::ArrayRef<at::IValue> args);
  Obj callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, at::IValue> kwargs);
  Obj callKwargs(std::unordered_map<std::string, at::IValue> kwargs);
  bool hasattr(const char* attr);
  Obj attr(const char* attr);
  InterpreterObj* getBaseObj();
private:
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
  virtual void unload(int64_t id) = 0;
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
  return baseObj_->call(args);
}

inline Obj Obj::operator()(at::ArrayRef<at::IValue> args) {
  return baseObj_->call(args);
}

inline Obj Obj::callKwargs(
    std::vector<at::IValue> args,
    std::unordered_map<std::string, at::IValue> kwargs) {
  return baseObj_->callKwargs(std::move(args), std::move(kwargs));
}
inline Obj Obj::callKwargs(
    std::unordered_map<std::string, at::IValue> kwargs) {
  return  baseObj_->callKwargs(std::move(kwargs));
}
inline bool Obj::hasattr(const char* attr) {
  return baseObj_->hasattr(attr);
}

inline Obj Obj::attr(const char* attr) {
  return baseObj_->attr(attr);
}

inline InterpreterObj* Obj::getBaseObj() {
  return baseObj_;
}

} // namespace deploy
} // namespace torch
