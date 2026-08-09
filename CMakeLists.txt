cmake_minimum_required(VERSION 3.16)

project(hnnet LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-16" CACHE STRING "C++ compiler" FORCE)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hnnet_lib INTERFACE)
add_library(hnnet::hnnet_lib ALIAS hnnet_lib)
target_include_directories(hnnet_lib INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_subdirectory(utility)
target_link_libraries(hnnet_lib INTERFACE utility::utility)

add_subdirectory(examples/perceptron)

message(STATUS "hnnet library configured successfully")
