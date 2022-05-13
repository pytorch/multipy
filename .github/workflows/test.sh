sudo yum update -y
sudo yum -y install git python3-pip
sudo pip3 install --upgrade pip
sudo yum -y install zlib-devel
sudo yum -y install clang llvm
export CC=clang
export CXX=clang++
sudo yum -y install xz-devel bzip2-devel libnsl2-devel readline-devel expat-devel gdbm-devel glibc-devel gmp-devel libffi-devel libGL-devel libX11-devel ncurses-devel openssl-devel sqlite-devel tcl-devel tix-devel tk-devel
sudo yum -y install lzma
sudo yum -y install uuid
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh
bash ~/miniconda.sh -b -p $HOME/miniconda
export PATH="/home/ec2-user/miniconda/bin:$PATH"
export CONDA="/home/ec2-user/miniconda"
conda create --name multipy_runtime_env python=3.8
conda run -n multipy_runtime_env python -m pip install astunparse numpy ninja pyyaml mkl mkl-include setuptools cmake cffi typing_extensions future six requests dataclasses pytest
cd /home/ec2-user/MultiPy/runtime/third-party/fmt
mkdir build
cd build
conda run -n multipy_runtime_env cmake ..
cd ../../pybind11
mkdir build
cd build
conda run -n multipy_runtime_env cmake ..
cd /home/ec2-user/MultiPy/runtime/third-party/pytorch
USE_DEPLOY=1
conda run -n multipy_runtime_env python setup.py develop
cd /home/ec2-user/MultiPy/runtime
mkdir build
cd build
pwd
conda run -n multipy_runtime_env cmake ..
conda run -n multipy_runtime_env cmake --build . --config Release
