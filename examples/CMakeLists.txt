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

# build our examples
add_executable(hello_world_example hello_world/hello_world_example.cpp)
target_link_libraries(hello_world_example PUBLIC "-Wl,--no-as-needed -rdynamic" dl pthread util multipy c10 torch_cpu)

add_executable(quickstart quickstart/quickstart.cpp)
target_link_libraries(quickstart PUBLIC "-Wl,--no-as-needed -rdynamic" dl pthread util multipy c10 torch_cpu)

add_executable(movable_example movable_example/movable_example.cpp)
target_link_libraries(movable_example PUBLIC "-Wl,--no-as-needed -rdynamic" dl pthread util multipy c10 torch_cpu)
