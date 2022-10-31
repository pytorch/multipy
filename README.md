[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE) ![Runtime Tests](https://github.com/pytorch/multipy/actions/workflows/runtime_tests.yaml/badge.svg)


# `torch::deploy` (MultiPy)

`torch::deploy` (MultiPy for non-PyTorch use cases) is a C++ library that enables you to run eager mode PyTorch models in production without any modifications to your model to support tracing. `torch::deploy` provides a way to run using multiple independent Python interpreters in a single process without a shared global interpreter lock (GIL). For more information on how `torch::deploy` works
internally, please see the related [arXiv paper](https://arxiv.org/pdf/2104.00254.pdf).

To learn how to use `torch::deploy` see [Installation](#installation) and [Examples](#examples).

Requirements:

* PyTorch 1.13+ or PyTorch nightly
* Linux (ELF based)
  * x86_64 (Beta)
  * arm64/aarch64 (Prototype)

> ℹ️ This is project is in Beta. `torch::deploy` is ready for use in production environments but may have some rough edges that we're continuously working on improving. We're always interested in hearing feedback and usecases that you might have. Feel free to reach out!

## Installation

### Building via Docker

The easiest way to build deploy and install the interpreter dependencies is to do so via docker.

```shell
git clone --recurse-submodules https://github.com/pytorch/multipy.git
cd multipy
export DOCKER_BUILDKIT=1
docker build -t multipy .
```

The built artifacts are located in `multipy/runtime/build`.

To run the tests:

```shell
docker run --rm multipy multipy/runtime/build/test_deploy
```

### Installing via `pip install`

We support installing both python modules and the runtime libs using `pip
install`, with the caveat of having to manually install the C++ dependencies
first. This serves as a single-command source build, essentially being a wrapper
around `python setup.py develop`, once all the dependencies have been installed.


To start with, the multipy repo should be cloned first:

```shell
git clone --recurse-submodules https://github.com/pytorch/multipy.git
cd multipy

# (optional) if using existing checkout
git submodule sync && git submodule update --init --recursive
```

#### Installing System Dependencies

The runtime system dependencies are specified in `build-requirements.txt`. To install them on Debian-based systems, one could run:

```shell
sudo apt update
xargs sudo apt install -y -qq --no-install-recommends <build-requirements.txt
```

#### Python Environment Setup

We support both `conda` and `pyenv`+`virtualenv` to create isolated environments to build and run in. Since `multipy` requires a position-independent version of python to launch interpreters with, for `conda` environments we use the prebuilt `libpython-static=3.x` libraries from `conda-forge` to link with at build time, and for `virtualenv`/`pyenv` we compile python with `-fPIC` to create the linkable library.

> **NOTE** We support Python versions 3.7 through 3.10 for `multipy`; note that for `conda` environments the `libpython-static` libraries are available for `3.8` onwards. With `virtualenv`/`pyenv` any version from 3.7 through 3.10 can be used, as the PIC library is built explicitly.

<details>
<summary>Click to expand</summary>

Example commands for installing conda:
```shell
curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
chmod +x ~/miniconda.sh && \
~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh
```
Virtualenv / pyenv can be installed as follows:
```shell
pip3 install virtualenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```
</details>


#### Installing python, pytorch and related dependencies

Multipy requires the latest version of pytorch to run models successfully, and we recommend fetching the latest _nightlies_ for pytorch and also cuda, if required.

##### In a `conda` environment, we would do the following:
```shell
conda create -n newenv
conda activate newenv
conda install python=3.8
conda install -c conda-forge libpython-static=3.8

# cuda
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly

# cpu only
conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
```

##### For a `pyenv` / `virtualenv` setup, one could do:
```shell
export CFLAGS="-fPIC -g"
~/.pyenv/bin/pyenv install --force 3.8.6
virtualenv -p ~/.pyenv/versions/3.8.6/bin/python3 ~/venvs/multipy
source ~/venvs/multipy/bin/activate
pip install -r dev-requirements.txt

# cuda
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cu113

# cpu only
pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu

```

#### Running `pip install`

Once all the dependencies are successfully installed, most importantly including a PIC-library of python and the latest nightly of pytorch, we can run the following, in either `conda` or `virtualenv`, to install both the python modules and the runtime/interpreter libraries:
```shell
# from base multipy directory
pip install -e .
```
The C++ binaries should be available in `/opt/dist`.

Alternatively, one can install only the python modules without invoking `cmake` as follows:
```shell
pip install  -e . --install-option="--cmakeoff"
```

> **NOTE** As of 10/11/2022 the linking of prebuilt static fPIC versions of python downloaded from `conda-forge` can be problematic on certain systems (for example Centos 8), with linker errors like `libpython_multipy.a: error adding symbols: File format not recognized`. This seems to be an issue with `binutils`, and the steps in https://wiki.gentoo.org/wiki/Project:Toolchain/Binutils_2.32_upgrade_notes/elfutils_0.175:_unable_to_initialize_decompress_status_for_section_.debug_info can help. Alternatively, the user can go with the `virtualenv`/`pyenv` flow above.

## Development

### Manually building `multipy::runtime` from source

Both `docker` and `pip install` options above are wrappers around the `cmake`
build of multipy's runtime. For development purposes it's often helpful to
invoke `cmake` separately.

See the install section for how to correctly setup the Python environment.

```bash
# checkout repo
git clone --recurse-submodules https://github.com/pytorch/multipy.git
cd multipy

# (optional) if using existing checkout
git submodule sync && git submodule update --init --recursive

cd multipy
# install python parts of `torch::deploy` in multipy/multipy/utils
pip install -e . --install-option="--cmakeoff"

cd multipy/runtime

# configure runtime to build/
cmake -S . -B build
# if you need to override the ABI setting you can pass
cmake -S . -B build -D_GLIBCXX_USE_CXX11_ABI=<0/1>

# compile the files in build/
cmake --build build --config Release -j
```

### Running unit tests for `multipy::runtime`

We first need to generate the neccessary examples. First make sure your python environment has [torch](https://pytorch.org). Afterwards, once `multipy::runtime` is built, run the following (executed automatically for `docker` and `pip` above):

```
cd multipy/multipy/runtime
python example/generate_examples.py
cd build
./test_deploy
```

## Examples

See the [examples directory](./examples) for complete examples.

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
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(multipy_tutorial)

set(MULTIPY_PATH ".." CACHE PATH "The repo where multipy is built or the PYTHONPATH")

# include the multipy utils to help link against
include(${MULTIPY_PATH}/multipy/runtime/utils.cmake)

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
cmake -S . -B build -DMULTIPY_PATH="/home/user/repos/multipy" # the parent directory of multipy (i.e. the git repo)
cmake --build build --config Release -j
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
