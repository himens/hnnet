cmake_minimum_required(VERSION 3.28)

# Il compilatore va forzato prima di project(), altrimenti CMake rileva
# il compilatore CXX di default e ne cachea l'identificazione/versione.
#set(CMAKE_CXX_COMPILER "/usr/bin/g++-16" CACHE STRING "C++ compiler" FORCE)

project(hnnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
#if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
#    # CMake 3.28 non conosce GCC 16, quindi CMAKE_CXX_STANDARD/REQUIRED rifiuterebbe
#    # il valore 26: impostiamo il flag direttamente, senza passare dal meccanismo di
#    # CMAKE_CXX_STANDARD.
#    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-std=c++26>)
#endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # Riduce la verbosità dei messaggi di errore su concept/template annidati
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fno-diagnostics-show-template-tree>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-fmax-errors=3>)
endif()

add_library(hnnet_lib INTERFACE)
add_library(hnnet::hnnet ALIAS hnnet_lib)
target_include_directories(hnnet_lib INTERFACE
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)

add_subdirectory(utility)
target_link_libraries(hnnet_lib INTERFACE utility::utility)

add_subdirectory(examples/perceptron)
add_subdirectory(examples/backprop)

message(STATUS "hnnet library configured successfully")
