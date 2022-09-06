#pragma once
#include <ATen/ATen.h>
#include <ATen/core/ivalue.h>
#include <caffe2/serialize/inline_container.h>

#include <multipy/runtime/interpreter/Optional.hpp>
#include <multipy/runtime/interpreter/plugin_registry.h>
#include <multipy/runtime/Exception.h>

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <unordered_map>
namespace py = pybind11;
using namespace py::literals;

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
#define TORCH_DEPLOY_SAFE_CATCH_RETHROW                                      \
  }                                                                          \
  catch (std::exception & err) {                                             \
    throw std::runtime_error(                                                \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) +             \
        ": Exception Caught inside torch::deploy embedded library: \n" +     \
        err.what());                                                         \
  }                                                                          \
  catch (...) {                                                              \
    throw std::runtime_error(                                                \
        std::string(__FILE__) + ":" + std::to_string(__LINE__) +             \
        ": Unknown Exception Caught inside torch::deploy embedded library"); \
  }

namespace torch {
namespace deploy {

// this is a wrapper class that refers to a PyObject* instance in a particular
// interpreter. We can't use normal PyObject or pybind11 objects here
// because these objects get used in a user application which will not directly
// link against libpython. Instead all interaction with the Python state in each
// interpreter is done via this wrapper class, and methods on
// InterpreterSession.
class Obj {
 public:
  Obj() : pyObject_(nullptr) {}
  Obj(py::object* pyObject)
      : pyObject_(pyObject) {}

  at::IValue toIValue();
  Obj operator()(at::ArrayRef<Obj> args);
  Obj operator()(at::ArrayRef<at::IValue> args);
  Obj callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs);
  Obj callKwargs(std::unordered_map<std::string, c10::IValue> kwargs);

  bool hasattr(const char* attr);
  Obj attr(const char* attr);
  py::object getPyObject() const;
  py::object call(py::handle args, py::handle kwargs = nullptr);
  Obj call(at::ArrayRef<c10::IValue> args);
  Obj call(at::ArrayRef<Obj> args);
  void unload();

 private:
  py::object* pyObject_;
};

} //deploy
} // torch
