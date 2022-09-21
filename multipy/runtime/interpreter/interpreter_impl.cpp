// Copyright (c) Meta Platforms, Inc. and affiliates.
// All rights reserved.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <dlfcn.h>
#include <multipy/runtime/interpreter/interpreter_impl.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <fmt/format.h>
#include <multipy/runtime/Exception.h>
#include <multipy/runtime/interpreter/builtin_registry.h>
#include <multipy/runtime/interpreter/import_find_sharedfuncptr.h>
#include <multipy/runtime/interpreter/plugin_registry.h>
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/autograd/generated/variable_factories.h>
#include <torch/csrc/jit/frontend/tracer.h>

#include <cassert>
#include <cstdio>
#include <iostream>
#include <map>
#include <memory>
#include <thread>

namespace py = pybind11;
using namespace py::literals;

// TODO this should come from cmake
#define DEBUG 1

#if (DEBUG == 1)
#define PYOBJ_ASSERT(obj) \
  if (NULL == obj) {      \
    PyErr_Print();        \
  }                       \
  assert(NULL != obj);
#elif (DEBUG == 0)
#define PYOBJ_ASSERT(obj) assert(NULL != obj);
#endif

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

   ==> Use MULTIPY_SAFE_RETHROW around _ALL_ torch::deploy APIs

   For more information, see
    https://gcc.gnu.org/wiki/Visibility (section on c++ exceptions)
    or https://stackoverflow.com/a/14364055
    or
   https://stackoverflow.com/questions/14268736/symbol-visibility-exceptions-runtime-error
    note- this may be only a serious problem on versions of gcc prior to 4.0,
   but still seems worth sealing off.

*/
#define MULTIPY_SAFE_RETHROW \
  return MultiPySafeRethrow(__FILE__, __LINE__) + [&]()
namespace {
class MultiPySafeRethrow {
 public:
  MultiPySafeRethrow(const char* file, int line) : file_(file), line_(line) {}

  // disable move and assignment
  MultiPySafeRethrow(const MultiPySafeRethrow&) = delete;
  MultiPySafeRethrow(MultiPySafeRethrow&&) = delete;
  MultiPySafeRethrow& operator=(const MultiPySafeRethrow&) = delete;
  MultiPySafeRethrow& operator=(MultiPySafeRethrow&&) = delete;

  template <typename FunctionType>
  auto operator+(FunctionType&& fn) const {
    try {
      return fn();
    } catch (py::error_already_set& err) {
      if (err.matches(PyExc_SystemExit)) {
        auto code = err.value().attr("code").cast<int>();
        std::exit(code);
      }
      throw std::runtime_error(
          std::string(file_) + ":" + std::to_string(line_) +
          ": Exception Caught inside torch::deploy embedded library: \n" +
          err.what());
    } catch (std::exception& err) {
      throw std::runtime_error(
          std::string(file_) + ":" + std::to_string(line_) +
          ": Exception Caught inside torch::deploy embedded library: \n" +
          err.what());
    } catch (...) {
      throw std::runtime_error(
          std::string(file_) + ":" + std::to_string(line_) +
          ": Unknown Exception Caught inside torch::deploy embedded library");
    }
  }

 private:
  const char* file_;
  const int line_;
};

std::vector<::torch::jit::StackEntry> noPythonCallstack() {
  return std::vector<::torch::jit::StackEntry>();
}

} // namespace

const char* start = R"PYTHON(
import _ssl # must come before _hashlib otherwise ssl's locks will be set to a Python that might no longer exist...
import sys
import importlib.abc
import linecache
from zipfile import ZipFile

# Disable Python library registration since it's not compatible with multipy.
sys.modules["torch._meta_registrations"] = object

class RegisterModuleImporter(importlib.abc.InspectLoader):
    def __init__(self, find_module_source):
        self.find_module_source = find_module_source

    def create_module(self, spec):
        return None

    def get_source(self, name):
        return self.find_module_source(name)

    def exec_module(self, module):
        filename = f"_deploy_internal.{module.__name__}"
        linecache.lazycache(filename, module.__dict__)
        code = compile(self.get_source(module.__name__), filename, "exec", dont_inherit=True)
        exec(code, module.__dict__)

    def find_spec(self, fullname, path, target=None):
        r = self.find_module_source(fullname)
        if r is not None:
            return importlib.util.spec_from_loader(fullname, self)
        return None

# print("exec_prefix:", sys.base_exec_prefix)
# print("_base_executable:", sys._base_executable)
# print("base_prefix:", sys.base_prefix)
# print("exec_prefix:", sys.exec_prefix)
# print("executable:", sys.executable)
# print("path:", sys.path)
# print("prefix:", sys.prefix)
# print("modules:", sys.modules)

import torch # has to be done serially otherwise things will segfault
import multipy.utils
try:
  import torch.version # for some reason torch doesn't import this and cuda fails?
except ModuleNotFoundError:
  # fbcode built doesn't have version.py, workaround by faking its info...
  from types import ModuleType
  _v = torch.version = sys.modules['torch.version'] = ModuleType('torch.version')
  _v.__version__ = '1.8.0a0+fake'
  _v.debug = False
  _v.cuda = '10.1'
  _v.git_version = 'fake'
  _v.hip = None


if torch.cuda.is_available():
  torch.zeros(1).cuda() # force cuda init...
import warnings
warnings.simplefilter("ignore")
)PYTHON";

extern "C" __attribute__((__weak__)) PyObject* PyInit_tensorrt(void);
extern "C"
    __attribute__((__weak__)) struct _frozen _PyImport_FrozenModules_tensorrt[];

using torch::deploy::BuiltinRegistry;
// TODO(shunting) move this to the tensorrt code
REGISTER_TORCH_DEPLOY_BUILTIN(
    tensorrt,
    _PyImport_FrozenModules_tensorrt,
    "tensorrt.tensorrt",
    PyInit_tensorrt);

static py::object global_impl(const char* module, const char* name) {
  return py::module::import(module).attr(name);
}

using at::IValue;
using torch::deploy::Obj;
using torch::deploy::PickledObject;

// Ensure GIL is held while this object is live,
// note: we are not use py::gil_scoped_acquire here because
// InitLockAcquire used below has to temporarily release the GIL
// within this scope to ensure locking order.  Having the source
// for these objects together makes it easier to see what is happening.
struct ScopedAcquire {
  ScopedAcquire() {
    gstate = PyGILState_Ensure();
  }
  ~ScopedAcquire() {
    PyGILState_Release(gstate);
  }
  PyGILState_STATE gstate;
};

struct InitLockAcquire {
  InitLockAcquire(std::mutex& init_lock) : init_lock_(init_lock) {
    // to avoid deadlock, we need to ensure a consistent lock order:
    // init_lock -> GIL. Otherwise, the GIL can be released by the python
    // interpreter during initalization tasks, and then re-acquired. If another
    // thread grabs the GIL to do non-initialization tasks, then it might start
    // initializing (GIL -> init_lock). To avoid this, release the GIL before
    // trying to get the init_lock and then reacquire it afterward.
    // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
    PyThreadState* _save;
    _save = PyEval_SaveThread();
    init_lock.lock();
    PyEval_RestoreThread(_save);
  }
  ~InitLockAcquire() {
    init_lock_.unlock();
  }

 private:
  std::mutex& init_lock_;
};

bool file_exists(const std::string& path) {
  struct stat buf;
  return (stat(path.c_str(), &buf) == 0);
}

struct __attribute__((visibility("hidden"))) ConcreteInterpreterObj
    : public torch::deploy::InterpreterObj {
  friend struct Obj;

  explicit ConcreteInterpreterObj(
      py::object pyObject,
      torch::deploy::InterpreterSessionImpl* interaction = nullptr)
      : torch::deploy::InterpreterObj(interaction), pyObject_(pyObject) {}
  explicit ConcreteInterpreterObj(
      at::IValue value,
      torch::deploy::InterpreterSessionImpl* interaction = nullptr)
      : torch::deploy::InterpreterObj(interaction),
        pyObject_(multipy::toPyObject(value)) {}
  explicit ConcreteInterpreterObj(Obj obj)
      : torch::deploy::InterpreterObj(obj.getInteraction()) {
    std::shared_ptr<ConcreteInterpreterObj> cObj =
        std::dynamic_pointer_cast<ConcreteInterpreterObj>(obj.baseObj_);
    pyObject_ = cObj->pyObject_;
  }
  ConcreteInterpreterObj() : pyObject_() {}
  ConcreteInterpreterObj(const ConcreteInterpreterObj& obj) = delete;
  ConcreteInterpreterObj& operator=(const ConcreteInterpreterObj& obj) = delete;
  ConcreteInterpreterObj(ConcreteInterpreterObj&& obj) = default;
  ConcreteInterpreterObj& operator=(ConcreteInterpreterObj&& obj) = default;

  py::handle getPyObject() const {
    MULTIPY_CHECK(pyObject_, "pyObject has already been freed");
    const py::handle h = pyObject_;
    return h;
  }

  at::IValue toIValue() const override {
    MULTIPY_SAFE_RETHROW {
      py::handle pyObj = getPyObject();
      return multipy::toTypeInferredIValue(pyObj);
    };
  }

  py::object call(py::handle args, py::handle kwargs = nullptr) {
    MULTIPY_SAFE_RETHROW {
      PyObject* result =
          PyObject_Call(getPyObject().ptr(), args.ptr(), kwargs.ptr());
      if (!result) {
        throw py::error_already_set();
      }
      return py::reinterpret_steal<py::object>(result);
    };
  }

  torch::deploy::Obj call(at::ArrayRef<Obj> args) override {
    MULTIPY_SAFE_RETHROW {
      py::tuple m_args(args.size());
      for (size_t i = 0, N = args.size(); i != N; ++i) {
        Obj obj = args[i];
        std::shared_ptr<ConcreteInterpreterObj> iObj =
            std::dynamic_pointer_cast<ConcreteInterpreterObj>(obj.baseObj_);
        m_args[i] =
            ((std::shared_ptr<ConcreteInterpreterObj>)iObj)->getPyObject();
      }
      py::object pyObj = call(m_args);
      std::shared_ptr<ConcreteInterpreterObj> cObj(
          new ConcreteInterpreterObj(pyObj, interaction_));
      return Obj(cObj);
    };
  }

  torch::deploy::Obj call(at::ArrayRef<at::IValue> args) override {
    MULTIPY_SAFE_RETHROW {
      py::tuple m_args(args.size());
      for (size_t i = 0, N = args.size(); i != N; ++i) {
        m_args[i] = multipy::toPyObject(args[i]);
      }
      py::object pyObj = call(m_args);
      std::shared_ptr<ConcreteInterpreterObj> cObj(
          new ConcreteInterpreterObj(pyObj, interaction_));
      return Obj(cObj);
    };
  }

  torch::deploy::Obj callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) override {
    MULTIPY_SAFE_RETHROW {
      py::tuple py_args(args.size());
      for (size_t i = 0, N = args.size(); i != N; ++i) {
        py_args[i] = multipy::toPyObject(args[i]);
      }

      py::dict py_kwargs;
      for (auto kv : kwargs) {
        py_kwargs[py::cast(std::get<0>(kv))] =
            multipy::toPyObject(std::get<1>(kv));
      }
      py::object pyObj = call(py_args, py_kwargs);
      std::shared_ptr<ConcreteInterpreterObj> cObj(
          new ConcreteInterpreterObj(pyObj, interaction_));
      return Obj(cObj);
    };
  }

  torch::deploy::Obj callKwargs(
      std::unordered_map<std::string, c10::IValue> kwargs) override {
    return callKwargs({}, kwargs);
  }

  bool hasattr(const char* attribute) override {
    MULTIPY_SAFE_RETHROW {
      return py::hasattr(getPyObject(), attribute);
    };
  }

  torch::deploy::Obj attr(const char* attribute) override {
    MULTIPY_SAFE_RETHROW {
      bool a = hasattr(attribute);
      py::object pyObj = getPyObject().attr(attribute);
      std::shared_ptr<ConcreteInterpreterObj> cObj(
          new ConcreteInterpreterObj(pyObj, interaction_));
      return Obj(cObj);
    };
  }

  void unload() {
    MULTIPY_SAFE_RETHROW {
      MULTIPY_CHECK(pyObject_, "pyObject has already been freed");
      // free(pyObject_);
      // pyObject_ = nullptr;
    };
  }

  py::object pyObject_;
};

extern "C" __attribute__((visibility("default"))) void
ConcreteInterpreterImplConstructorCommon(
    const std::vector<std::string>& extra_python_paths,
    const std::vector<std::string>& plugin_paths) {
  BuiltinRegistry::runPreInitialization();

#ifndef LEGACY_PYTHON_PRE_3_8
  PyPreConfig preconfig;
  PyPreConfig_InitIsolatedConfig(&preconfig);
  PyStatus status = Py_PreInitialize(&preconfig);
  TORCH_INTERNAL_ASSERT(!PyStatus_Exception(status))

  PyConfig config;

#ifdef FBCODE_CAFFE2
  PyConfig_InitIsolatedConfig(&config);

  // Completely blank out the path configuration. This ensures we have
  // complete control of how our embedded Python searches for modules, and we
  // will never consult the external filesystem. See:
  // https://docs.python.org/3/c-api/init_config.html#path-configuration
  config.site_import = 0;
  status = PyConfig_SetString(&config, &config.base_exec_prefix, L"");
  status =
      PyConfig_SetString(&config, &config.base_executable, L"torch_deploy");
  status = PyConfig_SetString(&config, &config.base_prefix, L"");
  status = PyConfig_SetString(&config, &config.exec_prefix, L"");
  status = PyConfig_SetString(&config, &config.executable, L"torch_deploy");
  status = PyConfig_SetString(&config, &config.prefix, L"");
  config.module_search_paths_set = 1;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-c-arrays,modernize-avoid-c-arrays)
  wchar_t* module_search_paths[0] = {};
  status = PyConfig_SetWideStringList(
      &config, &config.module_search_paths, 0, module_search_paths);
#else
  // dynamic linking path
  PyConfig_InitPythonConfig(&config);

#endif

  status = Py_InitializeFromConfig(&config);
  PyConfig_Clear(&config);
  TORCH_INTERNAL_ASSERT(!PyStatus_Exception(status))

#else
  Py_InitializeEx(1);
  TORCH_INTERNAL_ASSERT(Py_IsInitialized);
#endif

#ifdef FBCODE_CAFFE2
  auto sys_path = global_impl("sys", "path");
  for (const auto& entry : extra_python_paths) {
    sys_path.attr("insert")(0, entry);
  }
#endif

  if (plugin_paths.size() > 0) {
    auto sys_path = global_impl("sys", "path").cast<std::vector<std::string>>();
    std::string libtorch_python_path;
    for (auto path : sys_path) {
      auto file = path + "/torch/lib/libtorch_python.so";
      if (file_exists(file)) {
        libtorch_python_path = file;
        break;
      }
    }
    loadSearchFile(libtorch_python_path.c_str());
    for (auto path : plugin_paths) {
      loadSearchFile(path.c_str());
    }
  }

  BuiltinRegistry::runPostInitialization();
}

struct __attribute__((visibility("hidden"))) ConcreteInterpreterImpl
    : public torch::deploy::InterpreterImpl {
  explicit ConcreteInterpreterImpl(
      py::object saveStorageArg,
      py::object loadStorageArg,
      py::object getPackageArg,
      py::dict objectsArg)
      : saveStorage(saveStorageArg),
        loadStorage(loadStorageArg),
        getPackage(getPackageArg),
        objects(objectsArg) {}

  ~ConcreteInterpreterImpl() override {
    PyGILState_Ensure();
    // make sure pybind11 doesn't try to decref after we have destroyed python
    // note: this leads the referneces to these objects, but we are about to
    // deinit python anyway so it doesn't matter
    objects.release();
    saveStorage.release();
    loadStorage.release();
    getPackage.release();
    if (Py_FinalizeEx() != 0) {
      exit(1); // can't use TORCH_INTERNAL_ASSERT because we are in a
               // non-throwing destructor.
    }
  }

  void setFindModule(
      std::function<multipy::optional<std::string>(const std::string&)>
          find_module) override {
    std::function<py::object(const std::string&)> wrapped_find_module =
        [=](const std::string& name) -> py::object {
      auto r = find_module(name);
      return r ? py::cast(*r) : py::none();
    };
    py::object register_module_importer =
        py::module::import("__main__")
            .attr("RegisterModuleImporter")(wrapped_find_module);
    py::module::import("sys")
        .attr("meta_path")
        .attr("append")(register_module_importer);
  }

  torch::deploy::InterpreterSessionImpl* acquireSession() override;
  py::object saveStorage;
  py::object loadStorage;
  py::object getPackage;
  py::dict objects;
  std::mutex init_lock_;
};

struct __attribute__((visibility("hidden"))) ConcreteInterpreterSessionImpl
    : public torch::deploy::InterpreterSessionImpl {
  explicit ConcreteInterpreterSessionImpl(ConcreteInterpreterImpl* interp)
      : defaultObj_(Py_None), interp_(interp) {}
  Obj global(const char* module, const char* name) override {
    MULTIPY_SAFE_RETHROW {
      return wrap(global_impl(module, name));
    };
  }

  Obj fromIValue(IValue value) override {
    MULTIPY_SAFE_RETHROW {
      return wrap(multipy::toPyObject(value));
    };
  }

  Obj createOrGetPackageImporterFromContainerFile(
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          containerFile_) override {
    MULTIPY_SAFE_RETHROW {
      InitLockAcquire guard(interp_->init_lock_);
      py::object pet =
          (py::object)py::module_::import("torch._C").attr("PyTorchFileReader");
      return wrap(interp_->getPackage(containerFile_));
    };
  }

  PickledObject pickle(Obj container, Obj obj) override {
    MULTIPY_SAFE_RETHROW {
      py::tuple result = interp_->saveStorage(unwrap(container), unwrap(obj));
      py::bytes bytes = py::cast<py::bytes>(result[0]);
      py::list storages = py::cast<py::list>(result[1]);
      py::list dtypes = py::cast<py::list>(result[2]);
      auto container_file =
          py::cast<std::shared_ptr<caffe2::serialize::PyTorchStreamReader>>(
              result[3]);

      std::vector<at::Storage> storages_c;
      std::vector<at::ScalarType> dtypes_c;

      for (size_t i = 0, N = storages.size(); i < N; ++i) {
        storages_c.push_back(multipy::createStorage(storages[i].ptr()));
        dtypes_c.push_back(
            reinterpret_cast<THPDtype*>(dtypes[i].ptr())->scalar_type);
      }
      return PickledObject{
          bytes,
          std::move(storages_c),
          std::move(dtypes_c),
          std::move(container_file)};
    };
  }

  // meant to be used with replicated objects
  Obj unpickleOrGet(int64_t id, const PickledObject& obj) override {
    MULTIPY_SAFE_RETHROW {
      py::dict objects = interp_->objects;
      py::object id_p = py::cast(id);
      if (objects.contains(id_p)) {
        return wrap(objects[id_p]);
      }

      InitLockAcquire guard(interp_->init_lock_);
      // re-check if something else loaded this before we acquired the
      // init_lock_
      if (objects.contains(id_p)) {
        return wrap(objects[id_p]);
      }

      py::tuple storages(obj.storages_.size());
      for (size_t i = 0, N = obj.storages_.size(); i < N; ++i) {
        py::object new_storage = py::reinterpret_steal<py::object>(
            multipy::createPyObject(obj.storages_[i]));
        storages[i] = std::move(new_storage);
      }
      py::tuple dtypes(obj.types_.size());
      for (size_t i = 0, N = obj.types_.size(); i < N; ++i) {
        auto dtype = (PyObject*)multipy::getTHPDtype(obj.types_[i]);
        Py_INCREF(dtype);
        dtypes[i] = dtype;
      }
      py::object result = interp_->loadStorage(
          id, obj.containerFile_, py::bytes(obj.data_), storages, dtypes);
      return wrap(result);
    };
  }

  void unload(int64_t id) override {
    MULTIPY_SAFE_RETHROW {
      py::dict objects = interp_->objects;
      py::object id_p = py::cast(id);
      if (objects.contains(id_p)) {
        objects.attr("__delitem__")(id_p);
      }
    };
  }

  IValue toIValue(Obj obj) const override {
    MULTIPY_SAFE_RETHROW {
      std::shared_ptr<ConcreteInterpreterObj> cObj = unwrapConcreteObj(obj);
      return cObj->toIValue();
      // return multipy::toTypeInferredIValue(unwrap(obj));
    };
  }

  Obj call(Obj obj, at::ArrayRef<Obj> args) override {
    MULTIPY_SAFE_RETHROW {
      return unwrapConcreteObj(obj)->call(args);
    };
  }

  Obj call(Obj obj, at::ArrayRef<at::IValue> args) override {
    MULTIPY_SAFE_RETHROW {
      return unwrapConcreteObj(obj)->call(args);
    };
  }

  Obj callKwargs(
      Obj obj,
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) override {
    MULTIPY_SAFE_RETHROW {
      return unwrapConcreteObj(obj)->callKwargs(args, kwargs);
    };
  }

  Obj callKwargs(Obj obj, std::unordered_map<std::string, c10::IValue> kwargs)
      override {
    return callKwargs(obj, {}, kwargs);
  }

  bool hasattr(Obj obj, const char* attr) override {
    MULTIPY_SAFE_RETHROW {
      return unwrapConcreteObj(obj)->hasattr(attr);
      // return py::hasattr(unwrap(obj), attr);
    };
  }

  Obj attr(Obj obj, const char* attr) override {
    MULTIPY_SAFE_RETHROW {
      return unwrapConcreteObj(obj)->attr(attr);
      // return wrap(unwrap(obj).attr(attr));
    };
  }

  static py::object
  call(py::handle object, py::handle args, py::handle kwargs = nullptr) {
    MULTIPY_SAFE_RETHROW {
      PyObject* result = PyObject_Call(object.ptr(), args.ptr(), kwargs.ptr());
      if (!result) {
        throw py::error_already_set();
      }
      return py::reinterpret_steal<py::object>(result);
    };
  }

  std::shared_ptr<ConcreteInterpreterObj> unwrapConcreteObj(Obj obj) const {
    if (isDefault(obj)) {
      std::shared_ptr<ConcreteInterpreterObj> pConcreteObj(
          new ConcreteInterpreterObj(
              py::reinterpret_borrow<py::object>(defaultObj_)));
      return pConcreteObj;
    }
    std::shared_ptr<ConcreteInterpreterObj> cObj =
        std::dynamic_pointer_cast<ConcreteInterpreterObj>(obj.baseObj_);
    return cObj;
  }

  py::handle unwrap(Obj obj) const {
    // create a breakpoint here and check if h1 and h2 are the same.
    if (isDefault(obj)) {
      return defaultObj_;
    }
    std::shared_ptr<ConcreteInterpreterObj> cObj =
        std::dynamic_pointer_cast<ConcreteInterpreterObj>(obj.baseObj_);
    py::handle h = cObj->getPyObject();
    return h;
  }

  Obj wrap(py::object obj) {
    if (!defaultObj_) {
      defaultObj_ = obj;
    }
    std::shared_ptr<torch::deploy::InterpreterObj> pConcreteObj = std::make_shared<ConcreteInterpreterObj>(std::move(obj), this);
    return Obj(pConcreteObj);
  }

  py::handle defaultObj_;
  ConcreteInterpreterImpl* interp_;
  ScopedAcquire acquire_;
};

torch::deploy::InterpreterSessionImpl*
ConcreteInterpreterImpl::acquireSession() {
  return new ConcreteInterpreterSessionImpl(this);
}

extern "C" __attribute__((visibility("default")))
torch::deploy::InterpreterImpl*
newInterpreterImpl(
    const std::vector<std::string>& extra_python_paths,
    const std::vector<std::string>& plugin_paths) {
  ConcreteInterpreterImplConstructorCommon(extra_python_paths, plugin_paths);

  int r = PyRun_SimpleString(start);
  TORCH_INTERNAL_ASSERT(r == 0);

  // disable python callstack for jit tracer
  ::torch::jit::tracer::setPythonCallstack(&noPythonCallstack);

  py::object saveStorage =
      global_impl("multipy.utils._deploy", "_save_storages");
  py::object loadStorage =
      global_impl("multipy.utils._deploy", "_load_storages");
  py::object getPackage = global_impl("multipy.utils._deploy", "_get_package");
  py::dict objects = global_impl("multipy.utils._deploy", "_deploy_objects");

  PyEval_SaveThread();

  return new ConcreteInterpreterImpl(
      saveStorage, loadStorage, getPackage, objects);
}
