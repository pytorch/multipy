#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/lazy/core/debug_util.h>

#include "plugin_registry.h"

namespace multipy {
namespace torch {

namespace {

class TorchConverter : public Converter {
 public:
  TorchConverter() {
    registerConverter(this);
  }

  ~TorchConverter() override {
    deregisterConverter(this);
  }

  optional<c10::IValue> toTypeInferredIValue(py::handle input) override {
    return ::torch::jit::toTypeInferredIValue(input);
  }
  optional<py::object> toPyObject(c10::IValue ivalue) override {
    return ::torch::jit::toPyObject(ivalue);
  }
  optional<at::Storage> createStorage(PyObject* obj) override {
    return ::torch::createStorage(obj);
  }
  optional<PyObject*> createPyObject(const at::Storage& storage) override {
    return ::torch::createPyObject(storage);
  }
  optional<THPDtype*> getTHPDtype(at::ScalarType scalarType) override {
    return ::torch::getTHPDtype(scalarType);
  }
};

TorchConverter converter;
} // namespace
} // namespace torch
} // namespace multipy
