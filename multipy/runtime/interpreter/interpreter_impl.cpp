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

sys.executable = 'torch_deploy'

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

using c10::IValue;
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
  ConcreteInterpreterObj(py::object* pyObject)
      : pyObject_(pyObject) {}
  ConcreteInterpreterObj(Obj obj){
    ConcreteInterpreterObj* iObj = (ConcreteInterpreterObj*) (obj.getBaseObj());
    py::object pyObj = iObj->getPyObject();
    pyObject_ = &pyObj;
  }
  ConcreteInterpreterObj()
      : pyObject_(nullptr) {}
  ConcreteInterpreterObj(const ConcreteInterpreterObj& obj) = default;
  ConcreteInterpreterObj(ConcreteInterpreterObj&& obj) = default;
  ConcreteInterpreterObj(ConcreteInterpreterObj& obj) = default;

  py::object getPyObject() const {
    MULTIPY_CHECK(pyObject_, "pyObject has already been freed");
    return *pyObject_;
  }

  c10::IValue toIValue() const override {
    TORCH_DEPLOY_TRY
    py::handle pyObj = getPyObject();
    return multipy::toTypeInferredIValue(pyObj);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  py::object call(py::handle args, py::handle kwargs = nullptr) {
    TORCH_DEPLOY_TRY
    PyObject* result = PyObject_Call((*getPyObject()).ptr(), args.ptr(), kwargs.ptr());
    if (!result) {
      throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(result);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  torch::deploy::InterpreterObj* call(std::vector<Obj> args) override {
    TORCH_DEPLOY_TRY
    py::tuple m_args(args.size());
    for (size_t i = 0, N = args.size(); i != N; ++i) {
      InterpreterObj* iObj = args[i].getBaseObj();
      m_args[i] = ((ConcreteInterpreterObj*)iObj)->getPyObject();
    }
    py::object pyObj = call(m_args);
    ConcreteInterpreterObj iObj = ConcreteInterpreterObj(&pyObj);
    return &iObj;
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

torch::deploy::InterpreterObj* call(std::vector<c10::IValue> args) override {
      TORCH_DEPLOY_TRY
      py::tuple m_args(args.size());
      for (size_t i = 0, N = args.size(); i != N; ++i) {
        m_args[i] = multipy::toPyObject(args[i]);
      }
      py::object pyObj = call(m_args);
      ConcreteInterpreterObj iObj = ConcreteInterpreterObj(&pyObj);
      return &iObj;
      TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

torch::deploy::InterpreterObj* callKwargs(
      std::vector<c10::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) override {

    TORCH_DEPLOY_TRY
    py::tuple py_args(args.size());
    for (size_t i = 0, N = args.size(); i != N; ++i) {
      py_args[i] = multipy::toPyObject(args[i]);
    }

    py::dict py_kwargs;
    for (auto kv : kwargs) {
      py_kwargs[py::cast(std::get<0>(kv))] =
          multipy::toPyObject(std::get<1>(kv));
    }
    py::object pyObj =call(py_args, py_kwargs);
    ConcreteInterpreterObj iObj = ConcreteInterpreterObj(&pyObj);
    return &iObj;
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

torch::deploy::InterpreterObj* callKwargs(std::unordered_map<std::string, c10::IValue> kwargs) override
      {
    TORCH_DEPLOY_TRY
    std::vector<c10::IValue> args;
    return callKwargs(args, kwargs);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

bool hasattr(const char* attr) override {
  TORCH_DEPLOY_TRY
  return py::hasattr(getPyObject(), attr);
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

torch::deploy::InterpreterObj* attr(const char* attr) override {
  TORCH_DEPLOY_TRY
  py::object pyObj = getPyObject().attr(attr);
  ConcreteInterpreterObj iObj = ConcreteInterpreterObj(&pyObj);
  return &iObj;
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

void unload() {
  TORCH_DEPLOY_TRY
  MULTIPY_CHECK(pyObject_, "pyObject has already been freed");
  free(pyObject_);
  pyObject_ = nullptr;
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

~ConcreteInterpreterObj(){
  unload();
}
  py::object* pyObject_;
};


struct __attribute__((visibility("hidden"))) ConcreteInterpreterImpl
    : public torch::deploy::InterpreterImpl {
  explicit ConcreteInterpreterImpl(
      const std::vector<std::string>& extra_python_paths,
      const std::vector<std::string>& plugin_paths) {
    BuiltinRegistry::runPreInitialization();
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
#ifdef FBCODE_CAFFE2
    auto sys_path = global_impl("sys", "path");
    for (const auto& entry : extra_python_paths) {
      sys_path.attr("insert")(0, entry);
    }
#endif

    if (plugin_paths.size() > 0) {
      auto sys_path =
          global_impl("sys", "path").cast<std::vector<std::string>>();
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

    int r = PyRun_SimpleString(start);
    TORCH_INTERNAL_ASSERT(r == 0);

    // we cache these so we don't have to repeat the conversion of strings into
    // Python and hash table lookups to get to these object
    saveStorage = global_impl("multipy.utils._deploy", "_save_storages");
    loadStorage = global_impl("multipy.utils._deploy", "_load_storages");
    getPackage = global_impl("multipy.utils._deploy", "_get_package");
    objects = global_impl("multipy.utils._deploy", "_deploy_objects");
    // Release the GIL that PyInitialize acquires
    PyEval_SaveThread();
  }

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
  ConcreteInterpreterSessionImpl(ConcreteInterpreterImpl* interp)
      : interp_(interp) {}
  Obj createObj(py::object* pyObj){
    ConcreteInterpreterObj concreteObj = ConcreteInterpreterObj(pyObj);
    Obj obj = Obj(&concreteObj);
    createdObjs_.insert(pyObj);
    return obj;
  }
  Obj global(const char* module, const char* name) override {
    py::object globalObj = global_impl(module, name);
    return createObj(&globalObj);
  }

  Obj fromIValue(IValue value) override {
    py::object pyObj = multipy::toPyObject(value);
    return createObj(&pyObj);
  }
  Obj createOrGetPackageImporterFromContainerFile(
      const std::shared_ptr<caffe2::serialize::PyTorchStreamReader>&
          containerFile_) override {
    InitLockAcquire guard(interp_->init_lock_);
    py::object pet =
        (py::object)py::module_::import("torch._C").attr("PyTorchFileReader");
    py::object pyObj =  interp_->getPackage(containerFile_);
    return createObj(&pyObj);
  }
  // optoin 1) for this we stil have to enforce a hashtable relationship between objects and py::objects in the interpretersession.
  // option 2) another option is to embedded saveStorages into objects which are created from this interpreter
  // we can track if something is which is not too difficult
  // Regardless we probably have to do an isOwnerCheck :(
  PickledObject pickle(Obj container, Obj obj) override {
    ConcreteInterpreterObj* containerIObj = (ConcreteInterpreterObj*) (container.getBaseObj());
    ConcreteInterpreterObj* iObj = (ConcreteInterpreterObj*) obj.getBaseObj();
    py::object containerPyObject = containerIObj->getPyObject();
    py::object objPyObject = iObj->getPyObject();
    py::tuple result = interp_->saveStorage(containerPyObject, objPyObject);
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
  }
  Obj unpickleOrGet(int64_t id, const PickledObject& obj) override {
    if (unpickled_objects.find(id) != unpickled_objects.end()){
      return createObj(unpickled_objects[id]);
    }

    InitLockAcquire guard(interp_->init_lock_);
    // re-check if something else loaded this before we acquired the
    // init_lock_
    if (unpickled_objects.find(id) != unpickled_objects.end()){
      return createObj(unpickled_objects[id]);
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
    unpickled_objects[id] = &result;
    return createObj(&result);
  }
  void unload(int64_t id) override {
    ConcreteInterpreterObj obj = unpickled_objects[id];
    obj.unload();
  }

  c10::IValue toIValue(Obj obj) const override {
    return obj.toIValue();
  }

  Obj call(Obj obj, std::vector<Obj> args) override {
    return obj(args);
  }

  Obj call(Obj obj, std::vector<c10::IValue> args) override {
    return obj(args);
  }

  Obj callKwargs(
      Obj obj,
      std::vector<c10::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) override {
    return obj.callKwargs(args, kwargs);
  }

  Obj callKwargs(Obj obj, std::unordered_map<std::string, c10::IValue> kwargs)
      override {
    return obj.callKwargs(kwargs);
  }

  bool hasattr(Obj obj, const char* attr) override {
    return obj.hasattr(attr);
  }

  Obj attr(Obj obj, const char* attr) override {
    return obj.attr(attr);
  }

  bool isOwner(Obj obj){
    ConcreteInterpreterObj* iObj = (ConcreteInterpreterObj*) obj.getBaseObj();
    py::object pyObj = iObj->getPyObject();
    return createdObjs_.find(&pyObj) != createdObjs_.end();
  }

  ~ConcreteInterpreterSessionImpl() override {
    objects_.clear();
  }
  ConcreteInterpreterImpl* interp_;
  ScopedAcquire acquire_;
  std::vector<py::object> objects_;
  std::unordered_map<int64_t, py::object*> unpickled_objects;
  std::unordered_set<py::object*> createdObjs_;
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
  return new ConcreteInterpreterImpl(extra_python_paths, plugin_paths);
}
