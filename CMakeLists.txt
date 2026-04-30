cmake_minimum_required(VERSION 3.16)
project(MiniTensorEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MTE_BUILD_TESTS "Build Mini Tensor Engine tests" ON)

add_library(mte
    src/io.cpp
    src/layers.cpp
    src/tensor.cpp
)

target_include_directories(mte
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(mte
    PRIVATE
        -Wall
        -Wextra
        -Wpedantic
)

add_executable(mte_infer src/main.cpp)
target_link_libraries(mte_infer PRIVATE mte)

if(MTE_BUILD_TESTS)
    enable_testing()

    add_executable(mte_tests tests/test_tensor.cpp)
    target_link_libraries(mte_tests PRIVATE mte)

    add_test(NAME mte_tests COMMAND mte_tests)
endif()
