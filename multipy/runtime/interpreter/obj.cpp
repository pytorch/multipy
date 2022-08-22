namespace torch {
namespace deploy {

inline at::IValue Obj::toIValue() {
  TORCH_DEPLOY_TRY
  py::handle pyObj = getPyObject();
  return multipy::toTypeInferredIValue(pyObj);
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

inline Obj Obj::call(at::ArrayRef<Obj> args) {
    TORCH_DEPLOY_TRY
    py::tuple m_args(args.size());
    for (size_t i = 0, N = args.size(); i != N; ++i) {
      m_args[i] = args[i].getPyObject();
    }
    py::object pyObj = call(m_args);
    return Obj(&pyObj);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  inline Obj Obj::call(at::ArrayRef<c10::IValue> args) {
      TORCH_DEPLOY_TRY
      py::tuple m_args(args.size());
      for (size_t i = 0, N = args.size(); i != N; ++i) {
        m_args[i] = multipy::toPyObject(args[i]);
      }
      py::object pyObj = call(m_args);
      return Obj(&pyObj);
      TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  inline py::object Obj::call(py::handle args, py::handle kwargs) {
    TORCH_DEPLOY_TRY
    PyObject* result = PyObject_Call((*getPyObject()).ptr(), args.ptr(), kwargs.ptr());
    if (!result) {
      throw py::error_already_set();
    }
    return py::reinterpret_steal<py::object>(result);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }


 inline Obj Obj::callKwargs(
      std::vector<at::IValue> args,
      std::unordered_map<std::string, c10::IValue> kwargs) {

    TORCH_DEPLOY_TRY
    py::tuple py_args(args.size());
    for (size_t i = 0, N = args.size(); i != N; ++i) {
      py_args[i] = multipy::toPyObject(args[i]);
    }

    py::dict py_kwargs;
    for (auto kv : kwargs) {
      py_kwargs[py::cast(std::get<0>(kv))] =
          multipy::toPyObject(std::get<1>(kv));
    }
    py::object pyObj =call(py_args, py_kwargs);
    return Obj(&pyObj);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

  inline Obj Obj::callKwargs(std::unordered_map<std::string, c10::IValue> kwargs)
      {
    TORCH_DEPLOY_TRY
    std::vector<at::IValue> args;
    return callKwargs(args, kwargs);
    TORCH_DEPLOY_SAFE_CATCH_RETHROW
  }

inline bool Obj::hasattr(const char* attr) {
  TORCH_DEPLOY_TRY
  return py::hasattr(getPyObject(), attr);
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

inline Obj Obj::attr(const char* attr) {
  TORCH_DEPLOY_TRY
  py::object pyObj = getPyObject().attr(attr);
  return Obj(&pyObj);
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

inline void Obj::unload(){
  TORCH_DEPLOY_TRY
  MULTIPY_CHECK(pyObject_, "pyObject has already been freed");
  free(pyObject_);
  pyObject_ = nullptr;
  TORCH_DEPLOY_SAFE_CATCH_RETHROW
}

inline py::object Obj::getPyObject() const {
  MULTIPY_CHECK(pyObject_, "pyObject has already been freed");
  return *pyObject_;
}


} // deploy
} // torch
