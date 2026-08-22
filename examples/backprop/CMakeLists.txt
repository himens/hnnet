cmake_minimum_required(VERSION 3.16)

project(hnnet-backprop LANGUAGES CXX)

# Lo standard C++26 e il compilatore sono ereditati dal CMakeLists.txt radice.

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