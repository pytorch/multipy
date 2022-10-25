#pragma once

#include <ATen/core/ivalue.h>
#include <pybind11/embed.h>
#include <pybind11/functional.h>
#include <torch/csrc/Dtype.h>
#include <torch/csrc/utils/pybind.h>

#include <multipy/runtime/interpreter/Optional.hpp>

namespace py = pybind11;

namespace multipy {

/// A `Converter` is used in order to convert `PyObject`s/`py::object` into
/// an `IValue` or some other representation such as storage.
class Converter {
 public:
  virtual ~Converter() = default;

  /// Converts a `py::handle` to an `IValue`
  virtual multipy::optional<at::IValue> toTypeInferredIValue(
      py::handle input) = 0;

  /// Converts an `IValue` into a `py::object`
  virtual multipy::optional<py::object> toPyObject(at::IValue ivalue) = 0;

  /// Converts an `PyObject` into a `Storage`
  virtual multipy::optional<at::Storage> createStorage(PyObject* obj) = 0;

  /// Creates a `PyObject` from `storage`
  virtual multipy::optional<PyObject*> createPyObject(
      const at::Storage& storage) = 0;

  // Returns the `THPDtype` of `scalarType`
  virtual multipy::optional<THPDtype*> getTHPDtype(
      at::ScalarType scalarType) = 0;
};

/// Registers a converter to be used by torch::deploy / multipy.
/// The order of the registration of the converters is dictated by the order of
/// compilation.
void registerConverter(Converter*);
/// Deregisters a converter from torch::deploy / multipy
/// The order of the deregistration of the converters is dictated by the order
/// of compilation.
void deregisterConverter(Converter*);

at::IValue toTypeInferredIValue(py::handle input);
py::object toPyObject(at::IValue ivalue);
at::Storage createStorage(PyObject* obj);
PyObject* createPyObject(const at::Storage& storage);
THPDtype* getTHPDtype(at::ScalarType scalarType);
} // namespace multipy
