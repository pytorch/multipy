namespace torch {
namespace deploy {

// this is a wrapper class that refers to a PyObject* instance in a particular
// interpreter. We can't use normal PyObject or pybind11 objects here
// because these objects get used in a user application which will not directly
// link against libpython. Instead all interaction with the Python state in each
// interpreter is done via this wrapper class, and methods on
// InterpreterSession.
struct Obj {
  friend struct InterpreterSessionImpl;
  Obj() : interaction_(nullptr), id_(0), pyObject_(nullptr) {}
  Obj(InterpreterSessionImpl* interaction, int64_t id, py::object* pyObject)
      : interaction_(interaction), id_(id), pyObject_(pyObject) {}
  Obj(py::object* pyObject)
      : interaction_(nullptr), id_(0), pyObject_(pyObject) {}

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
  InterpreterSessionImpl* interaction_;
  int64_t id_;
  py::object* pyObject_;
};

} //deploy
} // torch
