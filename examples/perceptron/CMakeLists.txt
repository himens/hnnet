cmake_minimum_required(VERSION 3.16)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-16" CACHE STRING "C++ compiler" FORCE)
project(hnnet-perceptron LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hnnet-perceptron
    src/main.cpp
)

target_include_directories(hnnet-perceptron PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

set_target_properties(hnnet-perceptron PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "hnnet-perceptron executable configured successfully")
