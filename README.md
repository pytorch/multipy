# MultiPy

## How to Build MultiPy Runtime

Also before building please change PYTORCH_ROOT in `runtime/CMakeLists.txt` and `runtime/interpreter/CMakeLists.txt`

```
cd MultiPy/runtime

# build third party libraries
cd third-party/fmt
mkdir build
cd build
cmake ..

cd ../pybind11
mkdir build
cd build
cmake ..
cd ../..

mkdir build
cd build

# build runtime
# replace {PYTORCH_ROOT} with your path to pytorch/torch

with-proxy cmake -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF -DCMAKE_PREFIX_PATH="{PYTORCH_ROOT}/lib/;../third-party/pybind11/build;../third-party/fmt/build" ..
cmake —build . —config Release
```

