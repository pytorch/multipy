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
    // GetPythonFramesFunction depends on the current python so we need to tear
    // it down.
    // https://www.internalfb.com/code/fbsource/[ead2dd7fd4a6]/fbcode/caffe2/torch/csrc/lazy/python/init.cpp?lines=293
    ::torch::lazy::GetPythonFramesFunction() = nullptr;

    // Deregister all pytorch operators since they might have been dynamically
    // loaded so they won't deregister correctly on teardown.
    /*
    for (auto op : ::torch::jit::getAllOperators()) {
      ::torch::jit::deregisterOperator(op->schema());
    }
    */
  }

  optional<at::IValue> toTypeInferredIValue(py::handle input) override {
    return ::torch::jit::toTypeInferredIValue(input);
  }
  optional<py::object> toPyObject(at::IValue ivalue) override {
    return ::torch::jit::toPyObject(ivalue);
  }
};

TorchConverter converter;
} // namespace
} // namespace torch
} // namespace multipy
