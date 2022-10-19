# Installation

## Building `torch::deploy` via Docker

The easiest way to build multipy, along with fetching all interpreter dependencies, is to do so via docker.

<!-- #md -->
```shell
git clone https://github.com/pytorch/multipy.git
cd multipy
export DOCKER_BUILDKIT=1
docker build -t multipy .
```
<!-- #endmd -->

The built artifacts are located in `multipy/runtime/build`.

To run the tests:

<!-- #md -->
```shell
docker run --rm multipy multipy/runtime/build/test_deploy
```
<!-- #endmd -->

## Installing via `pip install`

We support installing both python modules and the runtime libs using `pip install`, with the caveat of having to manually install the dependencies first.

To start with, the multipy repo should be cloned first:

<!-- #md -->
```shell
git clone https://github.com/pytorch/multipy.git
cd multipy
git submodule sync && git submodule update --init --recursive
```
<!-- #endmd -->


### Installing system dependencies

The runtime system dependencies are specified in `build-requirements.txt`. To install them on Debian-based systems, one could run:

<!-- #md -->
```shell
sudo apt update
xargs sudo apt install -y -qq --no-install-recommends <build-requirements.txt
```
<!-- #endmd -->

### Installing environment encapsulators

We support both `conda` and `pyenv`+`virtualenv` to create isolated environments to build and run in. Since `multipy` requires a position-independent version of python to launch interpreters with, for `conda` environments we use the prebuilt `libpython-static=3.x` libraries from `conda-forge` to link with at build time, and for `virtualenv`/`pyenv` we compile python with `-fPIC` to create the linkable library.

> **NOTE** We support Python versions 3.7 through 3.10 for `multipy`; note that for `conda` environments the `libpython-static` libraries are available for `3.8` onwards. With `virtualenv`/`pyenv` any version from 3.7 through 3.10 can be used, as the PIC library is built explicitly.

<details>
<summary>Click to expand</summary>

Example commands for installing conda:
<!-- #md -->
```shell
curl -fsSL -v -o ~/miniconda.sh -O  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
chmod +x ~/miniconda.sh && \
~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh
```
<!-- #endmd -->

Virtualenv / pyenv can be installed as follows:

<!-- #md -->
```shell
pip3 install virtualenv
git clone https://github.com/pyenv/pyenv.git ~/.pyenv
```
<!-- #endmd -->

</details>


### Installing python, pytorch and related dependencies

Multipy requires the latest version of pytorch to run models successfully, and we recommend fetching the latest _nightlies_ for pytorch and also cuda, if required.

#### In a `conda` environment, we would do the following:
<!-- #md -->
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
<!-- #endmd -->

#### For a `pyenv` / `virtualenv` setup, one could do:

<!-- #md -->
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
<!-- #endmd -->

### Running `pip install`

Once all the dependencies are successfully installed, most importantly including a PIC-library of python and the latest nightly of pytorch, we can run the following, in either `conda` or `virtualenv`, to install both the python modules and the runtime/interpreter libraries:
<!-- #md -->
```shell
# from base multipy directory
pip install -e .
```
<!-- #endmd -->

The C++ binaries should be available in `/opt/dist`.

Alternatively, one can install only the python modules without invoking `cmake` as follows:
<!-- #md -->
```shell
pip install  -e . --install-option="--cmakeoff"
```
<!-- #endmd -->

> **NOTE** As of 10/11/2022 the linking of prebuilt static fPIC versions of python downloaded from `conda-forge` can be problematic on certain systems (for example Centos 8), with linker errors like `libpython_multipy.a: error adding symbols: File format not recognized`. This seems to be an issue with `binutils`, and the steps in https://wiki.gentoo.org/wiki/Project:Toolchain/Binutils_2.32_upgrade_notes/elfutils_0.175:_unable_to_initialize_decompress_status_for_section_.debug_info can help. Alternatively, the user can go with the `virtualenv`/`pyenv` flow above.


## Running `torch::deploy` build steps from source

Both `docker` and `pip install` options above are wrappers around the `cmake build` of multipy's runtime. If the user wishes to run the build steps manually instead, as before the dependencies would have to be installed in the user's (isolated) environment of choice first. After that the following steps can be executed:

### Building

<!-- #md -->
```bash
# checkout repo
git checkout https://github.com/pytorch/multipy.git
git submodule sync && git submodule update --init --recursive

cd multipy
# install python parts of `torch::deploy` in multipy/multipy/utils
pip install -e . --install-option="--cmakeoff"

cd multipy/runtime

# build runtime
mkdir build
cd build
# use cmake -DABI_EQUALS_1=ON .. instead if you want ABI=1
cmake ..
cmake --build . --config Release
```
<!-- #endmd -->

## Running unit tests for `torch::deploy`

We first need to generate the neccessary examples. First make sure your python enviroment has [torch](https://pytorch.org). Afterwards, once `torch::deploy` is built, run the following (executed automatically for `docker` and `pip` above):

<!-- #md -->
```
cd multipy/multipy/runtime
python example/generate_examples.py
cd build
./test_deploy
```
<!-- #endmd -->
