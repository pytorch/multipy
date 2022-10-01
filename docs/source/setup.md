# Setup

## Installation

You'll first need to install [PyTorch](https://pytorch.org/get-started/locally/) which includes
`torch.package`.

### Building `torch::deploy` via Docker

The easiest way to build multipy from source is to build it via docker.
<!-- #md -->
```shell
git clone https://github.com/pytorch/multipy.git
cd multipy
export DOCKER_BUILDKIT=1
docker build -t multipy .
```
<!-- #endmd -->
The built artifacts will be located in `/opt/dist`

To run the tests:
<!-- #md -->
```shell
docker run --rm multipy multipy/runtime/build/test_deploy
```
<!-- #endmd -->
### Installing `torch::deploy` from source

Multipy needs a local copy of python with `-fPIC` enabled as well as a recent copy of pytorch.

#### Dependencies: Conda
<!-- #md -->
```shell
conda install python=3.8
conda install -c conda-forge libpython-static=3.8 # libpython.a

# cuda
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch-nightly

# cpu only
conda install pytorch torchvision torchaudio cpuonly -c pytorch-nightly
```
<!-- #endmd -->
#### Dependencies: PyEnv
<!-- #md -->
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
<!-- #endmd -->
#### Building
<!-- #md -->
```bash
# checkout repo
git checkout https://github.com/pytorch/multipy.git
git submodule sync && git submodule update --init --recursive

cd multipy/multipy/runtime

# build runtime
mkdir build
cd build
# use cmake -DABI_EQUALS_1=ON .. instead if you want ABI=1
cmake ..
cmake --build . --config Release
```
<!-- #endmd -->
### Running unit tests for `torch::deploy`

We first need to generate the neccessary examples. First make sure your python enviroment has [torch](https://pytorch.org). Afterwards, once `torch::deploy` is built

<!-- #md -->
```
cd multipy/multipy/runtime
python example/generate_examples.py
cd build
./test_deploy
```
<!-- #endmd -->
