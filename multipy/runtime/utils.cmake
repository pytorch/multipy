# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# prefer the active Python version instead of the latest -- only works on cmake
# 3.15+
# https://cmake.org/cmake/help/latest/module/FindPython3.html#hints
if (NOT DEFINED _GLIBCXX_USE_CXX11_ABI)
  # infer the ABI setting from the installed version of PyTorch
  execute_process(
    COMMAND python -c "import torch; print(1 if torch._C._GLIBCXX_USE_CXX11_ABI else 0)"
    OUTPUT_VARIABLE _GLIBCXX_USE_CXX11_ABI
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE ret
  )
  if(ret EQUAL "1")
    message(FATAL_ERROR "Failed to detect ABI version")
  endif()
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${_GLIBCXX_USE_CXX11_ABI})
message(STATUS "_GLIBCXX_USE_CXX11_ABI - ${_GLIBCXX_USE_CXX11_ABI}")

set(Python3_FIND_STRATEGY LOCATION)

find_package (Python3 COMPONENTS Interpreter Development)
set(PYTORCH_ROOT "${Python3_SITELIB}")

# if pytorch was installed in develop mode we need to resolve the egg-link
set(PYTORCH_EGG_LINK "${PYTORCH_ROOT}/torch.egg-link")
if (EXISTS "${PYTORCH_EGG_LINK}")
  file (STRINGS "${PYTORCH_EGG_LINK}" PYTORCH_ROOT LIMIT_COUNT 1)
endif()

message(STATUS "PYTORCH_ROOT - ${PYTORCH_ROOT}" )

include_directories(BEFORE "${PYTORCH_ROOT}/torch/include")
include_directories(BEFORE "${PYTORCH_ROOT}/torch/include/torch/csrc/api/include/")
include_directories(BEFORE "${Python3_INCLUDE_DIRS}")
LINK_DIRECTORIES("${PYTORCH_ROOT}/torch/lib")

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
