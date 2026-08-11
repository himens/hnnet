cmake_minimum_required(VERSION 3.16)

project(hnnet-perceptron LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-16" CACHE STRING "C++ compiler" FORCE)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hnnet-perceptron-gate
    src/perceptron-gate.cpp
)

add_executable(hnnet-perceptron-classifier
    src/perceptron-classifier.cpp
)

target_link_libraries(hnnet-perceptron-gate PRIVATE hnnet::hnnet)
target_link_libraries(hnnet-perceptron-classifier PRIVATE hnnet::hnnet)

target_include_directories(hnnet-perceptron-gate PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_include_directories(hnnet-perceptron-classifier PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

set_target_properties(hnnet-perceptron-gate hnnet-perceptron-classifier PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "Perceptron executables configured successfully")
