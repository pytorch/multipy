# Hello World (Using InterpreterSession directly)

Here we use `torch::deploy` to print `Hello World` to the console without using `torch.package`. Instead we simply acquire an individual `InterpreterSession`, and use it to print `Hello World` directly.

## Write the C++ part

<!-- #md -->
```cpp

#include <multipy/runtime/deploy.h>
#include <multipy/runtime/path_environment.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
  // create two interpreters
  multipy::runtime::InterpreterManager manager(2);

  // Acquire a session on one of the interpreters
  auto I = manager.acquireOne();

  // from builtins import print
  // print("Hello world!")
  I.global("builtins", "print")({"Hello world!"});
}

```
<!-- #endmd -->

Here we introduce the flexibility of ``torch::deploy``'s `InterpreterSession` in that we are able to effectively treat them as python objects. This allows us to add further flexibility to the code exported by `torch.package` by interacting with it in C++.

``manager.acquireOne`` allows us to create an individual subinterpreter we can interact with.

``InterpreterSession::global(const char* module, const char* name)`` allows us to access python modules such as `builtins` and their attributes such as `print`. This function outputs an `Obj` from which is a wrapper around `print`. From here we call `print` by using `{"Hello world!"}` as its argument(s).

<br>

## Build and execute Hello World

Assuming the above C++ program was stored in a file called, `hello_world_example.cpp`, a
minimal `CMakeLists.txt` file would look like:

<!-- #md -->
```cmake
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(multipy_tutorial)

set(MULTIPY_PATH ".." CACHE PATH "The repo where multipy is built or the PYTHONPATH")

# include the multipy utils to help link against
include(${MULTIPY_PATH}/multipy/runtime/utils.cmake)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# add headers from multipy
include_directories(${MULTIPY_PATH})

# link the multipy prebuilt binary
add_library(multipy_internal STATIC IMPORTED)
set_target_properties(multipy_internal
    PROPERTIES
    IMPORTED_LOCATION
    ${MULTIPY_PATH}/multipy/runtime/build/libtorch_deploy.a)
caffe2_interface_library(multipy_internal multipy)

# build our examples
add_executable(hello_world_example hello_world_example.cpp)
target_link_libraries(hello_world_example PUBLIC "-Wl,--no-as-needed -rdynamic" dl pthread util multipy c10 torch_cpu)
```
<!-- #endmd -->


From here we execute the hello world program

<!-- #md -->
```bash
mkdir build
cd build
cmake -S . -B build/ -DMULTIPY_PATH="<Path to Multipy Library>" -DPython3_EXECUTABLE="$(which python3)" && \
cmake --build build/ --config Release -j
./hello_world
```
<!-- #endmd -->
