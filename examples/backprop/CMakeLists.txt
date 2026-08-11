cmake_minimum_required(VERSION 3.16)

project(hnnet-backprop LANGUAGES CXX)

set(CMAKE_CXX_COMPILER "/usr/bin/g++-16" CACHE STRING "C++ compiler" FORCE)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(hnnet-backprop-gate
    src/backprop-gate.cpp
)
    
target_link_libraries(hnnet-backprop-gate PRIVATE hnnet::hnnet)

target_include_directories(hnnet-backprop-gate PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

set_target_properties(hnnet-backprop-gate PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "Backprop executables configured successfully")