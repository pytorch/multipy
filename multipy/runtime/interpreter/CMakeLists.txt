# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

SET(INTERPRETER_DIR "${DEPLOY_DIR}/interpreter" )
SET(INTERPRETER_DIR "${DEPLOY_DIR}/interpreter" PARENT_SCOPE)

SET(MULTIPY_UTILS "${CMAKE_CURRENT_SOURCE_DIR}/../../utils")

OPTION(LEGACY_PYTHON_PRE_3_8 "Whether to use Python 3.7 codepaths." OFF)

message(STATUS "Python3_EXECUTABLE - ${Python3_EXECUTABLE}" )
message(STATUS "Python3_SITELIB - ${Python3_SITELIB}" )
message(STATUS "Python3_LIBRARIES - ${Python3_LIBRARIES}" )

# Find static libpython with a preference for nolto since cross compiler version
# LTO is problematic.
string(REPLACE ".so" ".nolto.a" Python3_STATIC_LIBRARIES ${Python3_LIBRARIES})
if (NOT EXISTS ${Python3_STATIC_LIBRARIES})
  string(REPLACE ".so" ".a" Python3_STATIC_LIBRARIES ${Python3_LIBRARIES})
endif()
message(STATUS "Python3_STATIC_LIBRARIES - ${Python3_STATIC_LIBRARIES}" )

include_directories(BEFORE ${CMAKE_SOURCE_DIR}/../..)

if (${LEGACY_PYTHON_PRE_3_8})
  message(STATUS "LEGACY_PYTHON_PRE_3_8 set")
  add_compile_definitions(LEGACY_PYTHON_PRE_3_8)
endif()

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
target_link_libraries(torch_deployinterpreter PRIVATE torch_python)
target_link_libraries(torch_deployinterpreter PRIVATE gtest)
target_link_libraries(torch_deployinterpreter PRIVATE multipy_torch)

unset(LEGACY_PYTHON_PRE_3_8)
