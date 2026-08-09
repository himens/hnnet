cmake_minimum_required(VERSION 3.16)

project(utility LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(utility_lib INTERFACE)
add_library(utility::utility ALIAS utility_lib)

target_include_directories(utility_lib INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

message(STATUS "utility library configured successfully")