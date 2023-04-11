# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

SET(INTERPRETER_DIR "${DEPLOY_DIR}/interpreter" )
SET(INTERPRETER_DIR "${DEPLOY_DIR}/interpreter" PARENT_SCOPE)

SET(MULTIPY_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../../utils")

message(STATUS "Python3_EXECUTABLE - ${Python3_EXECUTABLE}" )
message(STATUS "Python3_SITELIB - ${Python3_SITELIB}" )
message(STATUS "Python3_LIBRARIES - ${Python3_LIBRARIES}" )

# Find static libpython with a preference for nolto since cross compiler version
# LTO is problematic. We also recursively search in the directory for static
# libpython, as it's not guarenteed to be in the same directory as libpython such
# as with pyenv.

FUNCTION(find_file_with_different_suffix VAR FILE ORIGINAL_SUFFIX NEW_SUFFIX)
  string(REPLACE ${ORIGINAL_SUFFIX} ${NEW_SUFFIX} MODIFIED_FILE ${FILE})

  if (NOT EXISTS ${MODIFIED_FILE})
    get_filename_component(MODIFIED_FILENAME ${MODIFIED_FILE} NAME)
    string(REPLACE "${MODIFIED_FILENAME}" "**/${MODIFIED_FILENAME}" MODIFIED_FILE_GLOB ${MODIFIED_FILE} )
    file(GLOB_RECURSE MODIFIED_FILE ${MODIFIED_FILE_GLOB})
  endif()
  set(${VAR} ${MODIFIED_FILE} PARENT_SCOPE)
ENDFUNCTION()

find_file_with_different_suffix(Python3_STATIC_LIBRARIES ${Python3_LIBRARIES} ".so" ".nolto.a")

if (NOT EXISTS ${Python3_STATIC_LIBRARIES})
  find_file_with_different_suffix(Python3_STATIC_LIBRARIES ${Python3_LIBRARIES} ".so" ".a")
endif()

# If we still can't find the library, we fail early
if (NOT EXISTS ${Python3_STATIC_LIBRARIES})
  get_filename_component(Python_FILENAME ${Python3_LIBRARIES} NAME)
  string(REPLACE ".so" ".nolto.a" Python3_STATIC_NOLTO_FILENAME ${Python_FILENAME})
  string(REPLACE ".so" ".a" Python3_STATIC_FILENAME ${Python_FILENAME})
  message(FATAL_ERROR "Cannot find ${Python3_STATIC_NOLTO_FILENAME} or ${Python3_STATIC_FILENAME} in ${libpython_DIR}. Are you using a static version of python? Please refer to https://github.com/pytorch/multipy for install instructions using conda and pyenv.")
endif()

message(STATUS "Python3_STATIC_LIBRARIES - ${Python3_STATIC_LIBRARIES}" )

include_directories(BEFORE ${CMAKE_SOURCE_DIR}/../..)

# add gtest dependency
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

IF(CMAKE_BUILD_TYPE MATCHES Debug)
  set(OBJCOPY_FLAGS "")
else()
  # binutils prior to 2.32 have issues linking against newer libraries due to
  # debug info changes. By default we can strip these out to ensure it
  # compiles.
  set(OBJCOPY_FLAGS "--strip-debug")
endif()

# instantiate a library based on the objects that make up torch_python
# make sure system python isn't used here
add_custom_command(
  OUTPUT libpython_multipy.a
  COMMAND cp ${Python3_STATIC_LIBRARIES} libpython_multipy.a
  COMMAND chmod +w libpython_multipy.a
  COMMAND "${CMAKE_OBJCOPY}" ${OBJCOPY_FLAGS} --weaken-symbol=_PyImport_FindSharedFuncptr libpython_multipy.a
)
add_custom_target(libpython_multipy DEPENDS libpython_multipy.a)

# Build the interpreter lib, designed to be standalone and dlopened
# We bake the python and torch_python binding objs into libinterpreter
set(INTERPRETER_LIB_SOURCES
  ${INTERPRETER_DIR}/interpreter_impl.cpp
  ${INTERPRETER_DIR}/builtin_registry.cpp
  ${INTERPRETER_DIR}/import_find_sharedfuncptr.cpp
  ${INTERPRETER_DIR}/plugin_registry.cpp
  ${INTERPRETER_DIR}/../loader.cpp
  ${LINKER_SCRIPT}
)
add_library(torch_deployinterpreter SHARED ${INTERPRETER_LIB_SOURCES} ${LINKER_SCRIPT})
add_library(multipy_torch SHARED plugin_torch.cpp)

add_dependencies(torch_deployinterpreter libpython_multipy)
target_link_libraries(torch_deployinterpreter PRIVATE  "-Wl,--no-as-needed -rdynamic" ${CMAKE_CURRENT_BINARY_DIR}/libpython_multipy.a)

# need to ensure headers are present before any .cpp in interpreter are compiled,
# but cpp themselves don't clearly depend on cpython so there is a race otherwise
if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  target_compile_options(torch_deployinterpreter PRIVATE -fno-gnu-unique)
endif()
target_include_directories(torch_deployinterpreter PRIVATE ${INTERPRETER_DIR})
target_include_directories(torch_deployinterpreter BEFORE PUBLIC ${Python3_INCLUDE_DIRS})

target_link_libraries(torch_deployinterpreter PRIVATE fmt::fmt-header-only)
target_link_libraries(torch_deployinterpreter PRIVATE gtest)
target_link_libraries(torch_deployinterpreter PRIVATE torch_python)
target_link_libraries(torch_deployinterpreter PRIVATE multipy_torch)
