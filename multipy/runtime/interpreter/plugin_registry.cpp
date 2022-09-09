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

void deregisterConverter(Converter* c) {
  auto& converters = getConverters();
  auto it = std::find(converters.begin(), converters.end(), c);
  if (it != converters.end()) {
    converters.erase(it);
  }
}

c10::IValue toTypeInferredIValue(py::handle input) {
  for (auto c : getConverters()) {
    auto out = c->toTypeInferredIValue(input);
    if (out) {
      return *out;
    }
  }
  throw std::runtime_error("failed to convert to IValue");
}
py::object toPyObject(c10::IValue ivalue) {
  for (auto c : getConverters()) {
    auto out = c->toPyObject(ivalue);
    if (out) {
      return *out;
    }
  }
  throw std::runtime_error("failed to convert to py::object");
}
at::Storage createStorage(PyObject* obj) {
  for (auto c : getConverters()) {
    auto out = c->createStorage(obj);
    if (out) {
      return *out;
    }
  }
  throw std::runtime_error("failed to createStorage");
}
PyObject* createPyObject(const at::Storage& storage) {
  for (auto c : getConverters()) {
    auto out = c->createPyObject(storage);
    if (out) {
      return *out;
    }
  }
  throw std::runtime_error("failed to createPyObject");
}
THPDtype* getTHPDtype(at::ScalarType scalarType) {
  for (auto c : getConverters()) {
    auto out = c->getTHPDtype(scalarType);
    if (out) {
      return *out;
    }
  }
  throw std::runtime_error("failed to createPyObject");
}
} // namespace multipy
