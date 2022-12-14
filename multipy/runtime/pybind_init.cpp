#include <multipy/runtime/deploy.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/lazy/core/debug_util.h>

namespace py = pybind11;

using namespace torch::deploy;

namespace {
at::IValue detachIValue(at::IValue&& iv) {
  if (iv.isTensor()) {
    // detach tensors to avoid cross interpreter autograd state
    return std::move(iv).toTensor().detach();
  }
  return iv;
}
at::IValue toIValue(const py::handle& obj) {
  return detachIValue(torch::jit::toTypeInferredIValue(obj));
}
} // namespace

PYBIND11_MODULE(multipy_pybind, m) {
  m.doc() = "multipy python bindings";

  py::class_<InterpreterManager>(m, "InterpreterManager")
      .def(py::init<size_t>())
      .def("acquire_one", &InterpreterManager::acquireOne)
      .def(
          "__len__",
          [](InterpreterManager& self) -> int {
            return self.allInstances().size();
          })
      .def(
          "__getitem__",
          [](InterpreterManager& self, int i) -> InterpreterSession {
            return self.allInstances().at(i).acquireSession();
          });

  py::class_<Interpreter>(m, "Interpreter")
      .def("acquire_session", &Interpreter::acquireSession);

  py::class_<InterpreterSession>(m, "InterpreterSession")
      .def("global_", &InterpreterSession::global);

  py::class_<Obj>(m, "Obj")
      .def(
          "__call__",
          [](Obj& self, py::args args, const py::kwargs& kwargs) -> Obj {
            std::vector<at::IValue> iargs;
            std::unordered_map<std::string, at::IValue> ikwargs;

            for (auto& arg : args) {
              iargs.emplace_back(toIValue(arg));
            }
            for (auto& arg : kwargs) {
              ikwargs.emplace(
                  arg.first.cast<std::string>(), toIValue(arg.second));
            }

            return self.callKwargs(iargs, ikwargs);
          })
      .def(
          "__getattr__",
          [](Obj& self, std::string attr) -> Obj {
            return self.attr(attr.c_str());
          })
      .def("deref", [](Obj& self) -> py::object {
        return ::torch::jit::toPyObject(detachIValue(self.toIValue()));
      });
}
