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

## The Easy Path to Installation

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

## Getting Started with `torch::deploy`
Once you have `torch::deploy` built, check out our [tutorials](https://pytorch.org/multipy/latest/tutorials/tutorial_root.html) and
[API documentation](https://pytorch.org/multipy/latest/api/library_root.html).

## Contributing

We welcome PRs! See the [CONTRIBUTING](CONTRIBUTING.md) file.

## License

MultiPy is BSD licensed, as found in the [LICENSE](LICENSE) file.

## Legal

[Terms of Use](https://opensource.facebook.com/legal/terms)
[Privacy Policy](https://opensource.facebook.com/legal/privacy)

Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
