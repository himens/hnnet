cmake_minimum_required(VERSION 4.4.0)

project(utils LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(utils_lib INTERFACE)
add_library(utils::utils ALIAS utils_lib)
target_include_directories(utils_lib INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

message(STATUS "utils library configured successfully")
