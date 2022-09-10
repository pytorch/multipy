[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)


# \[experimental\] MultiPy

> :warning: **This is project is still a prototype.** Only Linux x86 is supported, and the API may change without warning. Furthermore, please **USE PYTORCH NIGHTLY** when using `multipy::runtime`!

`MultiPy` (formerly `torch::deploy` and `torch.package`) is a system that allows you to run multi-threaded python code in C++. It offers `multipy.package` (formerly `torch.package`) in order to package code into a mostly hermetic format to deliver to `multipy::runtime` (formerly `torch::deploy`) which is a runtime which takes packaged
code and runs it using multiple embedded Python interpreters in a C++ process without a shared global interpreter lock (GIL). For more information on how `MultiPy` works
internally, please see the related [arXiv paper](https://arxiv.org/pdf/2104.00254.pdf).

## Installation

You'll first need to install the `multipy` python module which includes
`multipy.package`.

```shell
pip install "git+https://github.com/pytorch/multipy.git"
```

### Installing `multipy::runtime` **(recommended)**

The C++ binaries (`libtorch_interpreter.so`,`libtorch_deploy.a`, `utils.cmake`), and the header files of `multipy::runtime` can be installed from our [nightly release](https://github.com/pytorch/multipy/releases/tag/nightly-runtime-abi-0). The ABI for the nightly release is 0. You can find a version of the release with ABI=1 [here](https://github.com/pytorch/multipy/releases/tag/nightly-runtime-abi-1).

```
wget https://github.com/pytorch/multipy/releases/download/nightly-runtime-abi-0/multipy_runtime.tar.gz
tar -xvzf multipy_runtime.tar.gz
```

In order to run PyTorch models, we need to link to libtorch (PyTorch's C++ distribution) which is provided when you [pip or conda install pytorch](https://pytorch.org/).
If you're not sure which ABI value to use, it's important to note that the pytorch C++ binaries, provided when you pip or conda install, are compiled with an ABI value of 0. If you're using libtorch from the pip or conda distribution of pytorch then ensure to use multipy installation with an ABI of 0 (`nightly-runtime-abi-0`).

### Building `multipy::runtime` via Docker

The easiest way to build multipy from source is to build it via docker.

```shell
git clone https://github.com/pytorch/multipy.git
cd multipy
export DOCKER_BUILDKIT=1
docker build -t multipy .
```

The built artifacts will be located in `/opt/dist`

To run the tests:

```shell
docker run --rm multipy multipy/runtime/build/test_deploy
```

### Installing `multipy::runtime` from source

Multipy needs a local copy of python with `-fPIC` enabled as well as a recent copy of pytorch.

#### Dependencies: Conda

```shell
conda install python=3.8
conda install -c conda-forge libpython-static=3.8 # libpython.a

# cuda
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly

# cpu only
conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
```

#### Dependencies: PyEnv

```shell
# install libpython.a with -fPIC enabled
export CFLAGS="-fPIC -g"
pyenv install --force 3.8.6
virtualenv -p ~/.pyenv/versions/3.8.6/bin/python3 ~/venvs/multipy
source ~/venvs/multipy/bin/activate

# cuda
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113

# cpu only
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
```

#### Building

```bash
# checkout repo
git checkout https://github.com/pytorch/multipy.git
git submodule sync && git submodule update --init --recursive

python setup.py install # install multipy.package
cd multipy/multipy/runtime

# build runtime
mkdir build
cd build
# use cmake -DABI_EQUALS_1=ON .. instead if you want ABI=1
cmake ..
cmake --build . --config Release
```

### Running unit tests for `multipy::runtime`

We first need to generate the neccessary examples. First make sure your python enviroment has [torch](https://pytorch.org). Afterwards, once `multipy::runtime` is built

```
cd multipy/multipy/runtime
python example/generate_examples.py
cd build
./test_deploy
```

## Example

### Packaging a model `for multipy::runtime`

``multipy::runtime`` can load and run Python models that are packaged with
``torch.package``. You can learn more about ``torch.package`` in the ``torch.package`` [documentation](https://pytorch.org/docs/stable/package.html#tutorials).

For now, let's create a simple model that we can load and run in ``multipy::runtime``.

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

Note that since "numpy", "sys", "PIL" were marked as "extern", `torch.package` will
look for these dependencies on the system that loads this package. They will not be packaged
with the model.

Now, there should be a file named ``my_package.pt`` in your working directory.

<br>

### Load the model in C++
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
    std::shared_ptr<multipy::runtime::Environment> env =
        std::make_shared<multipy::runtime::PathEnvironment>(
            std::getenv("PATH_TO_EXTERN_PYTHON_PACKAGES") // Ensure to set this environment variable (e.g. /home/user/anaconda3/envs/multipy-example/lib/python3.8/site-packages)
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

<br>

### Build and execute the C++ example

Assuming the above C++ program was stored in a file called, `example-app.cpp`, a
minimal `CMakeLists.txt` file would look like:

```cmake
cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(multipy_tutorial)

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# add headers from multipy
include_directories(${PATH_TO_MULTIPY_DIR})

add_library(torch_deploy_internal STATIC IMPORTED)

set_target_properties(multipy_internal
    PROPERTIES
    IMPORTED_LOCATION
    ${PATH_TO_MULTIPY_DIR}/multipy/runtime/lib/libtorch_deploy.a)

caffe2_interface_library(multipy_internal multipy)

add_executable(example-app example-app.cpp)
target_link_libraries(example-app PUBLIC
    "-Wl,--no-as-needed -rdynamic"
    shm crypt pthread dl util m ffi lzma readline nsl ncursesw panelw z multipy "${TORCH_LIBRARIES}")
```

Currently, it is necessary to build ``multipy::runtime`` as a static library.
In order to correctly link to a static library, the utility ``caffe2_interface_library``
is used to appropriately set and unset ``--whole-archive`` flag.

Furthermore, the ``-rdynamic`` flag is needed when linking to the executable
to ensure that symbols are exported to the dynamic table, making them accessible
to the deploy interpreters (which are dynamically loaded).

**Updating LIBRARY_PATH and LD_LIBRARY_PATH**

In order to locate dependencies provided by PyTorch (e.g. `libshm`), we need to update the `LIBRARY_PATH` and `LD_LIBRARY_PATH` environment variables to include the path to PyTorch's C++ libraries. If you installed PyTorch using pip or conda, this path is usually in the site-packages. An example of this is provided below.

```bash
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/user/anaconda3/envs/multipy-example/lib/python3.8/site-packages/torch/lib"
export LIBRARY_PATH="$LIBRARY_PATH:/home/user/anaconda3/envs/multipy-example/lib/python3.8/site-packages/torch/lib"
```

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
cmake -S . -B build/
    -DCMAKE_PREFIX_PATH="$(python -c 'import torch.utils; print(torch.utils.cmake_prefix_path)')" \
    -DPATH_TO_MULTIPY_DIR="/home/user/repos/" # whereever the multipy release was unzipped during installation

cd build
make -j
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
