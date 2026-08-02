cmake_minimum_required(VERSION 3.16)
project(hnnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hnnet_lib INTERFACE)
target_include_directories(hnnet_lib INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_executable(hnnet
    src/main.cpp
)

set_target_properties(hnnet PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

target_link_libraries(hnnet PRIVATE hnnet_lib)

message(STATUS "hnnet executable configured successfully")
