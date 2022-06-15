[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)


# \[experimental\] MultiPy

> :warning: **This is project is still a prototype.** Only Linux x86 is supported, and the API may change without warning.

`MultiPy` (formerly `torch::deploy` and `torch.package`) is a system that allows you to run multi-threaded python code in C++. It offers `multipy.package` (formerly `torch.package`) in order to package code into a mostly hermetic format to deliver to `multipy::runtime` (formerly `torch::deploy`) which is a runtime which takes packaged
code and runs it using multiple embedded Python interpreters in a C++ process without a shared global interpreter lock (GIL). For more information on how `MultiPy` works
internally, please see the related [arXiv paper](https://arxiv.org/pdf/2104.00254.pdf).

## Installation
### Installing `multipy::runtime`
`libtorch_interpreter.so`,`libtorch_deploy.a`, `utils.cmake`, and the header files of `multipy::runtime` can be installed from our [nightly release](https://github.com/pytorch/multipy/releases/download/nightly/multipy_runtime.tar.gz)

In order to run pytorch models, we need to use libtorch which can be setup using the instructions [here](https://pytorch.org/cppdocs/installing.html)

### Installing `multipy.package`
We will soon create a pypi distribution to `multipy.package`. For now one can use `torch.package` from `pytorch` as the functionality is exactly the same. The documentation for `torch.package` can be found [here](https://pytorch.org/docs/stable/package.html). Installation instructions for pytorch can be found [here](https://pytorch.org/get-started/locally/).

### How to build `multipy::runtime` from source
Currently we require that [pytorch be built from source](https://pytorch.org/get-started/locally/#mac-from-source) in order to build `multipy.runtime` from source. Please refer to that documentation for the requirements needed to build `pytorch` when running `USE_DEPLOY=1 python setup.py develop`.

```bash
# checkout repo
git checkout https://github.com/pytorch/multipy.git
git submodule sync && git submodule update --init --recursive

cd multipy/MultiPy/runtime

# Currently multipy::runtime requires that we build pytorch from source since we need to expose some objects in torch (ie. torch_python, etc.) for multipy::runtime to work.
cd ../pytorch
USE_DEPLOY=1 python setup.py develop
cd ../..

# build runtime
mkdir build
cd build
cmake ..
cmake --build . --config Release

## Quickstart

```

### Packaging a model `for multipy::runtime`

``multipy::runtime`` can load and run Python models that are packaged with
``multipy.package``. You can learn more about ``multipy.package`` in the
``multipy.package`` [documentation](https://pytorch.org/docs/stable/package.html#tutorials) (currently the documentation for `multipy.package` is the same as `torch.package` where we just replace `multipy.package` for all instances of `torch.package`).

For now, let's create a simple model that we can load and run in ``multipy::runtime``.

```python
from multipy.package import PackageExporter
import torchvision

# Instantiate some model
model = torchvision.models.resnet.resnet18()

# Package and export it.
with PackageExporter("my_package.pt") as e:
    e.intern("torchvision.**")
    e.extern("numpy.**")
    e.extern("sys")
    e.extern("PIL.*")
    e.save_pickle("model", "model.pkl", model)
```

Note that since "numpy", "sys" and "PIL" were marked as "extern", `multipy.package` will
look for these dependencies on the system that loads this package. They will not be packaged
with the model.

Now, there should be a file named ``my_package.pt`` in your working directory.


### Loading and running the model in C++
```cpp
#include <torch/csrc/deploy/deploy.h>
#include <torch/csrc/deploy/path_environment.h>
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
    std::shared_ptr<multipy::runtime::Environment> env =
        std::make_shared<multipy::runtime::PathEnvironment>(
            std::getenv("PATH_TO_EXTERN_PYTHON_PACKAGES")
        );
    multipy::runtime::InterpreterManager manager(4, env);

    try {
        // Load the model from the multipy.package.
        multipy::runtime::Package package = manager.loadPackage(argv[1]);
        multipy::runtime::ReplicatedObj model = package.loadPickle("model", "model.pkl");
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        std::cerr << e.msg();
        return -1;
    }

    std::cout << "ok\n";
}

```

This small program introduces many of the core concepts of ``multipy::runtime``.

An ``InterpreterManager`` abstracts over a collection of independent Python
interpreters, allowing you to load balance across them when running your code.

``PathEnvironment`` enables you to specify the location of Python
packages on your system which are external, but necessary, for your model.

Using the ``InterpreterManager::loadPackage`` method, you can load a
``multipy.package`` from disk and make it available to all interpreters.

``Package::loadPickle`` allows you to retrieve specific Python objects
from the package, like the ResNet model we saved earlier.

Finally, the model itself is a ``ReplicatedObj``. This is an abstract handle to
an object that is replicated across multiple interpreters. When you interact
with a ``ReplicatedObj`` (for example, by calling ``forward``), it will select
an free interpreter to execute that interaction.


Building and running the application when build from source

Locate `libtorch_deployinterpreter.o` on your system. This should have been
built when PyTorch was built from source. In the same PyTorch directory, locate
the deploy source files. Set these locations to an environment variable for the build.
An example of where these can be found on a system is shown below.

Assuming the above C++ program was stored in a file called, `example-app.cpp`, a
minimal CMakeLists.txt file would look like:

```cmake
    cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
    project(deploy_tutorial)

    find_package(fmt REQUIRED)
    find_package(Torch REQUIRED)

    add_library(torch_deploy STATIC
        ${DEPLOY_INTERPRETER_PATH}/libtorch_deployinterpreter.o
        ${DEPLOY_DIR}/deploy.cpp
        ${DEPLOY_DIR}/loader.cpp
        ${DEPLOY_DIR}/path_environment.cpp
        ${DEPLOY_DIR}/elf_file.cpp)

    # for python builtins
    target_link_libraries(torch_deploy PRIVATE
        crypt pthread dl util m z ffi lzma readline nsl ncursesw panelw)
    target_link_libraries(torch_deploy PUBLIC
        shm torch fmt::fmt-header-only)

    # this file can be found in multipy/runtime/utils.cmake
    caffe2_interface_library(torch_deploy torch_deploy_interface)

    add_executable(example-app example.cpp)
    target_link_libraries(example-app PUBLIC
        "-Wl,--no-as-needed -rdynamic" dl torch_deploy_interface "${TORCH_LIBRARIES}")
```

Currently, it is necessary to build ``multipy::runtime`` as a static library.
In order to correctly link to a static library, the utility ``caffe2_interface_library``
is used to appropriately set and unset ``--whole-archive`` flag.

Furthermore, the ``-rdynamic`` flag is needed when linking to the executable
to ensure that symbols are exported to the dynamic table, making them accessible
to the deploy interpreters (which are dynamically loaded).

The last step is configuring and building the project. Assuming that our code
directory is laid out like this:
```
example-app/
    CMakeLists.txt
    example-app.cpp
```


We can now run the following commands to build the application from within the
``example-app/`` folder:

```bash
mkdir build
cd build
# Point CMake at the built version of PyTorch we just installed.
cmake ..
cmake --build . --config Release
```



Now we can run our app:

```bash
./example-app /path/to/my_package.pt
```

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

MultiPy is BSD licensed, as found in the [LICENSE](LICENSE) file.

## Legal

[Terms of Use](https://opensource.facebook.com/legal/terms)
[Privacy Policy](https://opensource.facebook.com/legal/privacy)

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
