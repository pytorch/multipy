#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/lazy/core/debug_util.h>
#include <optional>

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

  std::optional<at::IValue> toTypeInferredIValue(py::handle input) override {
    return ::torch::jit::toTypeInferredIValue(input);
  }
  std::optional<py::object> toPyObject(at::IValue ivalue) override {
    return ::torch::jit::toPyObject(ivalue);
  }
  std::optional<at::Storage> createStorage(PyObject* obj) override {
    return ::torch::createStorage(obj);
  }
  std::optional<PyObject*> createPyObject(const at::Storage& storage) override {
    return ::torch::createPyObject(storage);
  }
  std::optional<THPDtype*> getTHPDtype(at::ScalarType scalarType) override {
    return ::torch::getTHPDtype(scalarType);
  }
};

TorchConverter converter;
} // namespace
} // namespace torch
} // namespace multipy
