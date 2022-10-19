# Quickstart

## Packaging a model `for torch::deploy`

``torch::deploy`` can load and run Python models that are packaged with
``torch.package``. You can learn more about ``torch.package`` in the ``torch.package`` [documentation](https://pytorch.org/docs/stable/package.html#tutorials).

For now, let's create a simple model that we can load and run in ``torch::deploy``.

<!-- #md -->
```python
from torch.package import PackageExporter
import torchvision

# Instantiate some model
model = torchvision.models.resnet.resnet18()

# Package and export it.
with PackageExporter("my_package.pt") as e:
    e.intern("torchvision.**")
    e.extern("numpy.**")
    e.extern("sys")
    e.extern("PIL.*")
    e.extern("typing_extensions")
    e.save_pickle("model", "model.pkl", model)
```
<!-- #endmd -->

Note that since "numpy", "sys", "PIL" were marked as "extern", `torch.package` will
look for these dependencies on the system that loads this package. They will not be packaged
with the model.

Now, there should be a file named ``my_package.pt`` in your working directory.

<br>

## Load the model in C++

<!-- #md -->
```cpp
#include <multipy/runtime/deploy.h>
#include <multipy/runtime/path_environment.h>
#include <torch/script.h>
#include <torch/torch.h>

#include <iostream>
#include <memory>

int main(int argc, const char* argv[]) {
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    // Start an interpreter manager governing 4 embedded interpreters.
    std::shared_ptr<torch::deploy::Environment> env =
        std::make_shared<torch::deploy::PathEnvironment>(
            std::getenv("PATH_TO_EXTERN_PYTHON_PACKAGES") // Ensure to set this environment variable (e.g. /home/user/anaconda3/envs/multipy-example/lib/python3.8/site-packages)
        );
    torch::deploy::InterpreterManager manager(4, env);

    try {
        // Load the model from the torch.package.
        torch::deploy::Package package = manager.loadPackage(argv[1]);
        torch::deploy::ReplicatedObj model = package.loadPickle("model", "model.pkl");
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cerr << e.msg();
        return -1;
    }

    std::cout << "ok\n";
}

```
<!-- #endmd -->

This small program introduces many of the core concepts of ``torch::deploy``.

An ``InterpreterManager`` abstracts over a collection of independent Python
interpreters, allowing you to load balance across them when running your code.

``PathEnvironment`` enables you to specify the location of Python
packages on your system which are external, but necessary, for your model.

Using the ``InterpreterManager::loadPackage`` method, you can load a
``torch.package`` from disk and make it available to all interpreters.

``Package::loadPickle`` allows you to retrieve specific Python objects
from the package, like the ResNet model we saved earlier.

Finally, the model itself is a ``ReplicatedObj``. This is an abstract handle to
an object that is replicated across multiple interpreters. When you interact
with a ``ReplicatedObj`` (for example, by calling ``forward``), it will select
an free interpreter to execute that interaction.

<br>

## Build and execute the C++ example

Assuming the above C++ program was stored in a file called, `example-app.cpp`, a
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

add_executable(example-app example-app.cpp)
target_link_libraries(example-app PUBLIC "-Wl,--no-as-needed -rdynamic" dl pthread util multipy c10 torch_cpu)
```
<!-- #endmd -->

Currently, it is necessary to build ``torch::deploy`` as a static library.
In order to correctly link to a static library, the utility ``caffe2_interface_library``
is used to appropriately set and unset ``--whole-archive`` flag.

Furthermore, the ``-rdynamic`` flag is needed when linking to the executable
to ensure that symbols are exported to the dynamic table, making them accessible
to the deploy interpreters (which are dynamically loaded).

**Updating LIBRARY_PATH and LD_LIBRARY_PATH**

In order to locate dependencies provided by PyTorch (e.g. `libshm`), we need to update the `LIBRARY_PATH` and `LD_LIBRARY_PATH` environment variables to include the path to PyTorch's C++ libraries. If you installed PyTorch using pip or conda, this path is usually in the site-packages. An example of this is provided below.

<!-- #md -->
```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/user/anaconda3/envs/multipy-example/lib/python3.8/site-packages/torch/lib"
export LIBRARY_PATH="$LIBRARY_PATH:/home/user/anaconda3/envs/multipy-example/lib/python3.8/site-packages/torch/lib"
```
<!-- #endmd -->

The last step is configuring and building the project. Assuming that our code
directory is laid out like this:

<!-- #md -->
```
example-app/
    CMakeLists.txt
    example-app.cpp
```
<!-- #endmd -->


We can now run the following commands to build the application from within the
``example-app/`` folder:

<!-- #md -->
```bash
cmake -S . -B build -DMULTIPY_PATH="/home/user/repos/multipy" # the parent directory of multipy (i.e. the git repo)
cmake --build build --config Release -j
```
<!-- #endmd -->

Now we can run our app:

<!-- #md -->
```bash
./example-app /path/to/my_package.pt
```
<!-- #endmd -->
