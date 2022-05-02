cmake_minimum_required(VERSION 3.19)
project(MultipyRuntime)
set(CMAKE_BUILD_TYPE Debug)
set(DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
add_subdirectory(third-party/fmt)
add_subdirectory(third-party/pybind11)
add_subdirectory(interpreter)
find_package(fmt REQUIRED)
find_package(pybind11 REQUIRED)
include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/../../pytorch/torch/include")
include_directories(BEFORE "${CMAKE_CURRENT_SOURCE_DIR}/../../pytorch/torch/include/torch/csrc/api/include/")
include_directories(BEFORE "/home/sahanp/local/pytorch/torch/csrc/api/include/torch")
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)
include("utils.cmake")
macro(caffe2_interface_library SRC DST)
  add_library(${DST} INTERFACE)
  add_dependencies(${DST} ${SRC})
  # Depending on the nature of the source library as well as the compiler,
  # determine the needed compilation flags.
  get_target_property(__src_target_type ${SRC} TYPE)
  # Depending on the type of the source library, we will set up the
  # link command for the specific SRC library.
  if(${__src_target_type} STREQUAL "STATIC_LIBRARY")
    # In the case of static library, we will need to add whole-static flags.
    if(APPLE)
      target_link_libraries(
          ${DST} INTERFACE -Wl,-force_load,\"$<TARGET_FILE:${SRC}>\")
    elseif(MSVC)
      # In MSVC, we will add whole archive in default.
      target_link_libraries(
         ${DST} INTERFACE "$<TARGET_FILE:${SRC}>")
      target_link_options(
         ${DST} INTERFACE "-WHOLEARCHIVE:$<TARGET_FILE:${SRC}>")
    else()
      # Assume everything else is like gcc
      target_link_libraries(${DST} INTERFACE
          "-Wl,--whole-archive,\"$<TARGET_FILE:${SRC}>\" -Wl,--no-whole-archive")
    endif()
    # Link all interface link libraries of the src target as well.
    # For static library, we need to explicitly depend on all the libraries
    # that are the dependent library of the source library. Note that we cannot
    # use the populated INTERFACE_LINK_LIBRARIES property, because if one of the
    # dependent library is not a target, cmake creates a $<LINK_ONLY:src> wrapper
    # and then one is not able to find target "src". For more discussions, check
    #   https://gitlab.kitware.com/cmake/cmake/issues/15415
    #   https://cmake.org/pipermail/cmake-developers/2013-May/019019.html
    # Specifically the following quote
    #
    # """
    # For STATIC libraries we can define that the PUBLIC/PRIVATE/INTERFACE keys
    # are ignored for linking and that it always populates both LINK_LIBRARIES
    # LINK_INTERFACE_LIBRARIES.  Note that for STATIC libraries the
    # LINK_LIBRARIES property will not be used for anything except build-order
    # dependencies.
    # """
    target_link_libraries(${DST} INTERFACE
        $<TARGET_PROPERTY:${SRC},LINK_LIBRARIES>)
  elseif(${__src_target_type} STREQUAL "SHARED_LIBRARY")
    if("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
      target_link_libraries(${DST} INTERFACE
          "-Wl,--no-as-needed,\"$<TARGET_FILE:${SRC}>\" -Wl,--as-needed")
    else()
      target_link_libraries(${DST} INTERFACE ${SRC})
    endif()
    # Link all interface link libraries of the src target as well.
    # For shared libraries, we can simply depend on the INTERFACE_LINK_LIBRARIES
    # property of the target.
    target_link_libraries(${DST} INTERFACE
        $<TARGET_PROPERTY:${SRC},INTERFACE_LINK_LIBRARIES>)
  else()
    message(FATAL_ERROR
        "You made a CMake build file error: target " ${SRC}
        " must be of type either STATIC_LIBRARY or SHARED_LIBRARY. However, "
        "I got " ${__src_target_type} ".")
  endif()
  # For all other interface properties, manually inherit from the source target.
  set_target_properties(${DST} PROPERTIES
    INTERFACE_COMPILE_DEFINITIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_DEFINITIONS>
    INTERFACE_COMPILE_OPTIONS
    $<TARGET_PROPERTY:${SRC},INTERFACE_COMPILE_OPTIONS>
    INTERFACE_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_INCLUDE_DIRECTORIES>
    INTERFACE_SYSTEM_INCLUDE_DIRECTORIES
    $<TARGET_PROPERTY:${SRC},INTERFACE_SYSTEM_INCLUDE_DIRECTORIES>)
endmacro()

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

add_library(torch_deploy_internal STATIC libtorch_deployinterpreter.o ${DEPLOY_DIR}/deploy.cpp ${DEPLOY_DIR}/loader.cpp ${DEPLOY_DIR}/path_environment.cpp ${DEPLOY_DIR}/elf_file.cpp)
target_link_libraries(torch_deploy_internal PRIVATE crypt pthread dl util m z ffi lzma readline nsl ncursesw panelw) # for python builtins
target_link_libraries(torch_deploy_internal PUBLIC  shm torch fmt::fmt-header-only)
caffe2_interface_library(torch_deploy_internal torch_deploy)

set(INTERPRETER_TEST_SOURCES
  ${DEPLOY_DIR}/test_deploy.cpp
)
set(INTERPRETER_TEST_SOURCES_GPU
  ${DEPLOY_DIR}/test_deploy_gpu.cpp
)

LINK_DIRECTORIES("~/local/pytorch/torch/lib")
add_executable(test_deploy ${INTERPRETER_TEST_SOURCES})
target_compile_definitions(test_deploy PUBLIC TEST_CUSTOM_LIBRARY)
target_include_directories(test_deploy PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(test_deploy
  PUBLIC "-Wl,--no-as-needed -rdynamic" gtest dl torch_deploy  c10 torch_cpu
)

LINK_DIRECTORIES("~/local/pytorch/torch/lib")
add_executable(test_deploy_gpu ${INTERPRETER_TEST_SOURCES_GPU})
target_compile_definitions(test_deploy_gpu PUBLIC TEST_CUSTOM_LIBRARY)
target_include_directories(test_deploy_gpu PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(test_deploy_gpu
  PUBLIC "-Wl,--no-as-needed -rdynamic" gtest dl torch_deploy  c10 torch_cpu
)

LINK_DIRECTORIES("~/local/pytorch/torch/lib")
add_library(test_deploy_lib SHARED test_deploy_lib.cpp)
add_dependencies(test_deploy_lib cpython)
target_include_directories(test_deploy_lib BEFORE PRIVATE ${PYTHON_INC_DIR})
target_link_libraries(test_deploy_lib PRIVATE pybind11::pybind11)

LINK_DIRECTORIES("~/local/pytorch/torch/lib")
add_executable(deploy_benchmark ${DEPLOY_DIR}/example/benchmark.cpp)
target_include_directories(deploy_benchmark PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(deploy_benchmark
  PUBLIC "-Wl,--no-as-needed -rdynamic" torch_deploy  c10 torch_cpu
)

LINK_DIRECTORIES("~/local/pytorch/torch/lib")
add_executable(interactive_embedded_interpreter ${DEPLOY_DIR}/interactive_embedded_interpreter.cpp)
target_include_directories(interactive_embedded_interpreter PRIVATE ${PYTORCH_ROOT}/torch)
target_link_libraries(interactive_embedded_interpreter
  PUBLIC "-Wl,--no-as-needed -rdynamic" torch_deploy c10 torch_cpu
)

if(INSTALL_TEST)
  install(TARGETS test_deploy DESTINATION bin)
  install(TARGETS test_deploy_gpu DESTINATION bin)
endif()

install(TARGETS torch_deploy DESTINATION lib)
