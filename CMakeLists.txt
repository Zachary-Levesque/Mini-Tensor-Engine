cmake_minimum_required(VERSION 3.16)
project(MiniTensorEngine VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MTE_BUILD_TESTS "Build Mini Tensor Engine tests" ON)
option(MTE_ENABLE_AVX2 "Enable AVX2 matmul dot-product kernel when supported" ON)

add_library(mte
    src/io.cpp
    src/layers.cpp
    src/model.cpp
    src/quantize.cpp
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

if(MTE_ENABLE_AVX2)
    if(MSVC)
        target_compile_options(mte PRIVATE /arch:AVX2)
        target_compile_definitions(mte PRIVATE MTE_ENABLE_AVX2=1)
    else()
        include(CheckCXXCompilerFlag)
        check_cxx_compiler_flag("-mavx2" MTE_COMPILER_SUPPORTS_AVX2)
        if(MTE_COMPILER_SUPPORTS_AVX2)
            target_compile_options(mte PRIVATE -mavx2)
            target_compile_definitions(mte PRIVATE MTE_ENABLE_AVX2=1)
        endif()
    endif()
endif()

add_executable(mte_infer src/main.cpp)
target_link_libraries(mte_infer PRIVATE mte)

add_executable(mte_benchmark src/benchmark.cpp)
target_link_libraries(mte_benchmark PRIVATE mte)

if(MTE_BUILD_TESTS)
    enable_testing()

    add_executable(mte_tests tests/test_tensor.cpp)
    target_link_libraries(mte_tests PRIVATE mte)

    add_test(NAME mte_tests COMMAND mte_tests)
endif()
