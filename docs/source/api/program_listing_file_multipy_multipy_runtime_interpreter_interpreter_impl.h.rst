:github_url: https://github.com/pytorch/multipy


.. _program_listing_file_multipy_multipy_runtime_interpreter_interpreter_impl.h:

Program Listing for File interpreter_impl.h
===========================================

|exhale_lsh| :ref:`Return to documentation for file <file_multipy_multipy_runtime_interpreter_interpreter_impl.h>` (``multipy/multipy/runtime/interpreter/interpreter_impl.h``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

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
   
   #include <multipy/runtime/Exception.h>
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
     friend struct InterpreterSessionImpl;
   
    protected:
     InterpreterSessionImpl* owningSession_;
   
    public:
     InterpreterObj() : owningSession_(nullptr) {}
     explicit InterpreterObj(InterpreterSessionImpl* owningSession)
         : owningSession_(owningSession) {}
     InterpreterObj(const InterpreterObj& obj) = delete;
     InterpreterObj& operator=(const InterpreterObj& obj) = delete;
     InterpreterObj(InterpreterObj&& obj) = default;
     InterpreterObj& operator=(InterpreterObj&& obj) = default;
     virtual ~InterpreterObj() = default;
   
    private:
     virtual at::IValue toIValue() const = 0;
     virtual Obj call(at::ArrayRef<std::shared_ptr<InterpreterObj>> args) = 0;
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
   // link against libpython. Instead all owningSession with the Python state in
   // each interpreter is done via this wrapper class, and methods on
   // InterpreterSession.
   struct Obj {
     friend struct InterpreterSessionImpl;
     friend struct InterpreterObj;
     explicit Obj(std::shared_ptr<InterpreterObj> baseObj)
         : isDefault_(false), baseObj_(baseObj) {}
     Obj() : isDefault_(true), baseObj_(nullptr) {}
   
     at::IValue toIValue() const;
     Obj operator()(at::ArrayRef<Obj> args);
     Obj operator()(at::ArrayRef<at::IValue> args);
     Obj callKwargs(
         std::vector<at::IValue> args,
         std::unordered_map<std::string, c10::IValue> kwargs);
     Obj callKwargs(std::unordered_map<std::string, c10::IValue> kwargs);
     bool hasattr(const char* attr);
     Obj attr(const char* attr);
   
    private:
     bool isDefault_;
     std::shared_ptr<InterpreterObj> baseObj_;
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
     std::shared_ptr<InterpreterObj> getBaseObj(Obj obj) const {
       return obj.baseObj_;
     }
     bool isOwner(Obj obj) const {
       return this == obj.baseObj_->owningSession_;
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
     std::vector<std::shared_ptr<torch::deploy::InterpreterObj>> copy;
     for (Obj arg : args) {
       copy.push_back(arg.baseObj_);
     }
     return baseObj_->call(copy);
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
