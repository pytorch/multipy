#include "multipy/runtime/interpreter/plugin_registry.h"

#include <vector>

namespace multipy {

std::vector<Converter*>& getConverters() {
  static std::vector<Converter*> converters;
  return converters;
}

void registerConverter(Converter* c) {
  getConverters().emplace_back(c);
}

at::IValue toTypeInferredIValue(py::handle input) {
  for (auto c : getConverters()) {
    auto out = c->toTypeInferredIValue(input);
    if (out) {
      return *out;
    }
  }
  throw std::runtime_error("failed to convert to IValue");
}
py::object toPyObject(at::IValue ivalue) {
  for (auto c : getConverters()) {
    auto out = c->toPyObject(ivalue);
    if (out) {
      return *out;
    }
  }
  throw std::runtime_error("failed to convert to py::object");
}
} // namespace multipy
