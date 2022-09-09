#pragma once

#include <ATen/core/ivalue.h>
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <torch/csrc/Dtype.h>

#include <multipy/runtime/interpreter/Optional.hpp>

namespace py = pybind11;

namespace multipy {

class Converter {
 public:
  virtual ~Converter() = default;

  virtual multipy::optional<c10::IValue> toTypeInferredIValue(
      py::handle input) = 0;
  virtual multipy::optional<py::object> toPyObject(c10::IValue ivalue) = 0;
  virtual multipy::optional<at::Storage> createStorage(PyObject* obj) = 0;
  virtual multipy::optional<PyObject*> createPyObject(
      const at::Storage& storage) = 0;
  virtual multipy::optional<THPDtype*> getTHPDtype(
      at::ScalarType scalarType) = 0;
};

void registerConverter(Converter*);
void deregisterConverter(Converter*);

c10::IValue toTypeInferredIValue(py::handle input);
py::object toPyObject(c10::IValue ivalue);
at::Storage createStorage(PyObject* obj);
PyObject* createPyObject(const at::Storage& storage);
THPDtype* getTHPDtype(at::ScalarType scalarType);
} // namespace multipy
