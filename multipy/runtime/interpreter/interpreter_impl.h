// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once
// multi-python abstract code
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>

#include <multipy/runtime/interpreter/Optional.hpp>

namespace torch {
namespace deploy {

struct InterpreterSessionImpl;
struct Obj;

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

 protected:
  InterpreterSessionImpl* interaction_;

 public:
  InterpreterObj() : interaction_(nullptr){};
  InterpreterObj(InterpreterSessionImpl* interaction)
      : interaction_(interaction){};
  InterpreterObj(const InterpreterObj& obj) = delete;
  InterpreterObj& operator=(const InterpreterObj& obj) = delete;
  InterpreterObj(InterpreterObj&& obj) = default;
  InterpreterObj& operator=(InterpreterObj&& obj) = default;
  virtual ~InterpreterObj() = default;

 private:
  virtual at::IValue toIValue() const = 0;
  virtual Obj call(at::ArrayRef<Obj> args) = 0;
  virtual Obj call(at::ArrayRef<at::IValue> args) = 0;
  virtual Obj callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual Obj callKwargs(
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual bool hasattr(const char* attr) = 0;
  virtual Obj attr(const char* attr) = 0;
};

// this is a wrapper class that refers to a PyObject* instance in a particular
// interpreter. We can't use normal PyObject or pybind11 objects here
// because these objects get used in a user application which will not directly
// link against libpython. Instead all interaction with the Python state in each
// interpreter is done via this wrapper class, and methods on
// InterpreterSession.
struct Obj {
  friend struct InterpreterSessionImpl;
  friend struct InterpreterObj;
  explicit Obj(std::shared_ptr<InterpreterObj> baseObj)
      : baseObj_(baseObj), isDefault_(false) {}
  Obj() : baseObj_(nullptr), isDefault_(true) {}
  explicit Obj(InterpreterSessionImpl* interaction,
      std::shared_ptr<InterpreterObj> baseObj)
      : baseObj_(baseObj), isDefault_(false) {}

  at::IValue toIValue() const;
  Obj operator()(at::ArrayRef<Obj> args);
  Obj operator()(at::ArrayRef<at::IValue> args);
  Obj callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs);
  Obj callKwargs(std::unordered_map<std::string, c10::IValue> kwargs);
  bool hasattr(const char* attr);
  Obj attr(const char* attr);
  InterpreterSessionImpl* getInteraction() {
    if (!baseObj_) {
      return nullptr;
    }
    return baseObj_->interaction_;
  }
  std::shared_ptr<InterpreterObj> baseObj_;

 private:
  bool isDefault_;
};

struct InterpreterSessionImpl {
  friend struct Package;
  friend struct ReplicatedObj;
  friend struct Obj;
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
  virtual void unload(int64_t id) = 0;

  virtual at::IValue toIValue(Obj obj) const = 0;

  virtual Obj call(Obj obj, at::ArrayRef<Obj> args) = 0;
  virtual Obj call(Obj obj, at::ArrayRef<at::IValue> args) = 0;
  virtual Obj callKwargs(
      Obj obj,
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual Obj callKwargs(
      Obj obj,
      std::unordered_map<std::string, c10::IValue> kwargs) = 0;
  virtual Obj attr(Obj obj, const char* attr) = 0;
  virtual bool hasattr(Obj obj, const char* attr) = 0;

 protected:
  int64_t isDefault(Obj obj) const {
    return obj.isDefault_;
  }
  bool isOwner(Obj obj) const {
    return this == obj.getInteraction();
  }
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
    std::unordered_map<std::string, c10::IValue> kwargs) {
  return baseObj_->callKwargs(std::move(args), std::move(kwargs));
}
inline Obj Obj::callKwargs(
    std::unordered_map<std::string, c10::IValue> kwargs) {
  return baseObj_->callKwargs(std::move(kwargs));
}
inline bool Obj::hasattr(const char* attr) {
  return baseObj_->hasattr(attr);
}

inline Obj Obj::attr(const char* attr) {
  return baseObj_->attr(attr);
}

} // namespace deploy
} // namespace torch
