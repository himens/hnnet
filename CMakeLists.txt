cmake_minimum_required(VERSION 4.4.0)

project(hnnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(hnnet_lib INTERFACE)
add_library(hnnet::hnnet ALIAS hnnet_lib)
target_include_directories(hnnet_lib INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")

add_subdirectory(utility)
target_link_libraries(hnnet_lib INTERFACE utility::utility)

find_package(OpenMP REQUIRED) # make it optional
target_link_libraries(hnnet_lib INTERFACE OpenMP::OpenMP_CXX)

add_subdirectory(examples/perceptron)
add_subdirectory(examples/backprop)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(hnnet_lib INTERFACE -Wfatal-errors)
endif()

message(STATUS "hnnet library configured successfully")
