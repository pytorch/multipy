# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.19)
project(MultipyRuntime)

# set ABI by default to 0

option(ABI_EQUALS_1 "Set ABI value to 1, by default it is set to 0. Pytorch by default builds with ABI set to 1." OFF)
option(BUILD_CUDA_TESTS "Set to ON in order to build cuda tests. By default we do not" OFF)
option(GDB_ON "Sets to debug mode (for gdb), defaults to OFF" OFF)

OPTION(LEGACY_PYTHON_PRE_3_8 "Whether to use Python 3.7 codepaths." OFF)

if(ABI_EQUALS_1)
  set(ABI_VALUE 1)
else()
  set(ABI_VALUE 0)
endif()

if(GDB_ON)
  set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "CMAKE_BUILD_TYPE - ${CMAKE_BUILD_TYPE}" )

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${ABI_VALUE})
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${ABI_VALUE} -fabi-version=11 -fno-lto")

SET(INTERPRETER_DIR "${DEPLOY_DIR}/interpreter" )
SET(INTERPRETER_DIR "${DEPLOY_DIR}/interpreter" PARENT_SCOPE)

set(DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
get_filename_component(MULTIPY_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

find_package (Python3 COMPONENTS Interpreter Development)
set(PYTORCH_ROOT "${Python3_SITELIB}")

add_subdirectory(interpreter)
add_subdirectory(third-party/fmt)
include_directories(BEFORE "${PYTORCH_ROOT}/torch/include")
include_directories(BEFORE "${PYTORCH_ROOT}/torch/include/torch/csrc/api/include/")
include_directories(BEFORE "${Python3_INCLUDE_DIRS}")


include(${DEPLOY_DIR}/utils.cmake)


# we do not want to have torch_deployinterpreter linked against libstdc++ or libc because
# when loading it with RTLD_DEEPBIND it will resolve std::cout/stdout to the copy in libc++/libc instead of the
# ones in the main process (see https://gcc.gnu.org/bugzilla/show_bug.cgi?id=42679).
# However, we can't just instruct the linker to not link against these libraries because these
# libraries use function versioning. Without linking them, the shared library would not know the right
# symbol versions and instead try to link against the old ones. Our solution is to link the library
# normally then remove the DT_NEEDED entries in the ELF file that instruct the loaded to load the sublibraries.
# This gives us the right version numbers but no direct dependency on libstdc++/libc. When loaded these
# symbols will fallback to resolution through the main execution and get the correct values
add_executable(remove_dt_needed remove_dt_needed.cpp)
target_link_libraries(remove_dt_needed PRIVATE fmt::fmt-header-only)

add_custom_command(
  OUTPUT libtorch_deployinterpreter.o
  # remove the DT_NEEDED entries
  COMMAND $<TARGET_FILE:remove_dt_needed> $<TARGET_FILE:torch_deployinterpreter> libtorch_deployinterpreter_all.so
  # package the result into an object we can link into the libdeploy binary.
  COMMAND ld -r -b binary -o libtorch_deployinterpreter.o libtorch_deployinterpreter_all.so
  COMMAND objcopy --rename-section .data=.torch_deploy_payload.interpreter_all,readonly,contents -N _binary_libtorch_deployinterpreter_all_so_start -N _binary_libtorch_deployinterpreter_all_so_end libtorch_deployinterpreter.o
  COMMAND rm libtorch_deployinterpreter_all.so
  DEPENDS torch_deployinterpreter remove_dt_needed
  VERBATIM
)

add_custom_command(
  OUTPUT libmultipy_torch.o
  # copy the file
  COMMAND cp $<TARGET_FILE:multipy_torch> libmultipy_torch.so
  # package the result into an object we can link into the libdeploy binary.
  COMMAND ld -r -b binary -o libmultipy_torch.o libmultipy_torch.so
  COMMAND objcopy --rename-section .data=.torch_deploy_payload.multipy_torch,readonly,contents -N _binary_libmultipy_torch_so_start -N _binary_libmultipy_torch_so_end libmultipy_torch.o
  COMMAND rm libmultipy_torch.so
  DEPENDS multipy_torch
  VERBATIM
)

add_library(torch_deploy STATIC
  libtorch_deployinterpreter.o
  libmultipy_torch.o
  ${DEPLOY_DIR}/deploy.cpp
  ${DEPLOY_DIR}/loader.cpp
  ${DEPLOY_DIR}/embedded_file.cpp
  ${DEPLOY_DIR}/path_environment.cpp
  ${DEPLOY_DIR}/elf_file.cpp
)
target_link_libraries(torch_deploy PRIVATE crypt pthread dl util m z ffi lzma readline nsl ncursesw panelw) # for python builtins
target_link_libraries(torch_deploy PUBLIC  shm torch fmt::fmt-header-only)
target_include_directories(torch_deploy PRIVATE ${CMAKE_SOURCE_DIR}/../..)

# copied from caffe2_interface_library
caffe2_interface_library(torch_deploy torch_deploy_interface)


set(INTERPRETER_TEST_SOURCES
  ${DEPLOY_DIR}/test_deploy.cpp
)
set(INTERPRETER_TEST_SOURCES_GPU
  ${DEPLOY_DIR}/test_deploy_gpu.cpp
)

# TODO: Currently tests can only be done when ABI=1 as the testing infrustructure
# used by ASSERT_TRUE requires ABI=1 in Github actions, we should fix this!

LINK_DIRECTORIES("${PYTORCH_ROOT}/torch/lib")
add_executable(test_deploy ${INTERPRETER_TEST_SOURCES})
# target_compile_definitions(test_deploy PUBLIC TEST_CUSTOM_LIBRARY)
if (${LEGACY_PYTHON_PRE_3_8})
  message(STATUS "LEGACY_PYTHON_PRE_3_8 set in parent cmakefile.")
  target_compile_definitions(test_deploy PUBLIC LEGACY_PYTHON_PRE_3_8)
endif()
target_include_directories(test_deploy PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(test_deploy
  PUBLIC "-Wl,--no-as-needed -rdynamic" gtest dl torch_deploy_interface c10 torch_cpu
)
target_include_directories(test_deploy PRIVATE ${CMAKE_SOURCE_DIR}/../..)

if(BUILD_CUDA_TESTS)
  LINK_DIRECTORIES("${PYTORCH_ROOT}/torch/lib")
  add_executable(test_deploy_gpu ${INTERPRETER_TEST_SOURCES_GPU})
  target_compile_definitions(test_deploy_gpu PUBLIC TEST_CUSTOM_LIBRARY)
  target_include_directories(test_deploy_gpu PRIVATE ${PYTORCH_ROOT}/torch)
  target_include_directories(test_deploy_gpu PRIVATE ${CMAKE_SOURCE_DIR}/../..)
  target_link_libraries(test_deploy_gpu
   PUBLIC "-Wl,--no-as-needed -rdynamic" gtest dl torch_deploy_interface c10 torch_cpu
 )
endif()

LINK_DIRECTORIES("${PYTORCH_ROOT}/torch/lib")
add_library(test_deploy_lib SHARED test_deploy_lib.cpp)
target_include_directories(test_deploy_lib BEFORE PRIVATE ${PYTHON_INC_DIR})
target_include_directories(test_deploy_lib PRIVATE ${CMAKE_SOURCE_DIR}/../..)

LINK_DIRECTORIES("${PYTORCH_ROOT}/torch/lib")
add_executable(deploy_benchmark ${DEPLOY_DIR}/example/benchmark.cpp)
target_include_directories(deploy_benchmark PRIVATE ${PYTORCH_ROOT}/torch)
target_include_directories(deploy_benchmark PRIVATE ${CMAKE_SOURCE_DIR}/../..)
target_link_libraries(deploy_benchmark
  PUBLIC "-Wl,--no-as-needed -rdynamic" torch_deploy_interface c10 torch_cpu
)

LINK_DIRECTORIES("${PYTORCH_ROOT}/torch/lib")
add_executable(interactive_embedded_interpreter ${DEPLOY_DIR}/interactive_embedded_interpreter.cpp)
target_include_directories(interactive_embedded_interpreter PRIVATE ${PYTORCH_ROOT}/torch)
target_include_directories(interactive_embedded_interpreter PRIVATE ${CMAKE_SOURCE_DIR}/../..)
target_link_libraries(interactive_embedded_interpreter
  PUBLIC "-Wl,--no-as-needed -rdynamic" torch_deploy_interface c10 torch_cpu
)

install(TARGETS test_deploy DESTINATION tests/bin)

if(BUILD_CUDA_TESTS)
 install(TARGETS test_deploy_gpu DESTINATION tests/bin)
endif()

install(TARGETS test_deploy DESTINATION tests/bin)

install (
    DIRECTORY ${CMAKE_SOURCE_DIR}
    DESTINATION dist/multipy
    FILES_MATCHING
    PATTERN "*.h*"
    PATTERN "utils.cmake"
    PATTERN "interpreter/third_party" EXCLUDE
    PATTERN "interpreter/cpython" EXCLUDE
    PATTERN "interpreter/frozen" EXCLUDE
    PATTERN "third-party" EXCLUDE
    PATTERN "build" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
    PATTERN "example" EXCLUDE
    )

install (
    DIRECTORY ${CMAKE_SOURCE_DIR}
    DESTINATION multipy
    FILES_MATCHING
    PATTERN "example/generated/*"
    PATTERN "test_deploy_python.py"
    PATTERN "interpreter/third_party" EXCLUDE
    PATTERN "interpreter/cpython" EXCLUDE
    PATTERN "interpreter/frozen" EXCLUDE
    PATTERN "third-party" EXCLUDE
    PATTERN "build" EXCLUDE
    PATTERN "unity" EXCLUDE
    PATTERN "__pycache__" EXCLUDE
    )

install(TARGETS torch_deploy DESTINATION dist/multipy/runtime/lib)
install(TARGETS torch_deployinterpreter DESTINATION dist/multipy/runtime/lib)
