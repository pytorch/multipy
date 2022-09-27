cmake_minimum_required(VERSION 3.19 FATAL_ERROR)
project(multipy_tutorial)
find_package (Python3 COMPONENTS Interpreter Development)
message(STATUS "Python3_SITELIB - ${Python3_SITELIB}" )

find_package(Torch REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_USE_CXX11_ABI=0")
set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=0")

# add headers from multipy
set(PATH_TO_MULTIPY_DIR "../../../multipy")
include_directories(BEFORE ${PATH_TO_MULTIPY_DIR}/..)
add_subdirectory(${PATH_TO_MULTIPY_DIR}/runtime multipy/runtime)

LINK_DIRECTORIES("${Python3_SITELIB}/torch/lib")
add_executable(movable-example movable-example.cpp)
target_link_libraries(movable-example PUBLIC "-Wl,--no-as-needed -rdynamic" torch_deploy_interface)
