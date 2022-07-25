#pragma once

#include <ATen/core/ivalue.h>
#include <pybind11/embed.h>
#include <pybind11/functional.h>

#include <multipy/runtime/interpreter/Optional.hpp>

namespace py = pybind11;

namespace multipy {

class Converter {
 public:
  virtual ~Converter() = default;

  virtual multipy::optional<at::IValue> toTypeInferredIValue(
      py::handle input) = 0;
  virtual multipy::optional<py::object> toPyObject(at::IValue ivalue) = 0;
};

void registerConverter(Converter*);

at::IValue toTypeInferredIValue(py::handle input);
py::object toPyObject(at::IValue ivalue);
} // namespace multipy
