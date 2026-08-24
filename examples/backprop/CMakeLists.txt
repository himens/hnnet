cmake_minimum_required(VERSION 3.28)

project(hnnet-backprop LANGUAGES CXX)

# Lo standard C++26 e il compilatore sono ereditati dal CMakeLists.txt radice.

add_executable(hnnet-backprop-gate
    src/backprop-gate.cpp
)

add_executable(hnnet-backprop-mnist
    src/backprop-mnist.cpp
)

target_link_libraries(hnnet-backprop-gate PRIVATE hnnet::hnnet)
target_link_libraries(hnnet-backprop-mnist PRIVATE hnnet::hnnet)

target_include_directories(hnnet-backprop-gate PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

target_include_directories(hnnet-backprop-mnist PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/../../include"
)

set_target_properties(hnnet-backprop-gate hnnet-backprop-mnist PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

message(STATUS "Backprop executables configured successfully")
