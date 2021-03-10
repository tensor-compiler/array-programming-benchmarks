cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(taco-bench)
set(CMAKE_BUILD_TYPE "release")
option(GRAPHBLAS "Build GraphBLAS benchmarks" OFF)

if(GRAPHBLAS)
  message("-- Will build GraphBLAS benchmarks")
  add_definitions(-DGRAPHBLAS)
endif(GRAPHBLAS)

add_subdirectory(taco)
add_subdirectory(benchmark)
if(GRAPHBLAS)
  add_subdirectory(suitesparse)
endif(GRAPHBLAS)

include_directories(taco taco/include benchmark/include suitesparse/Include)

file(GLOB TEST_SOURCES *.cpp *.h)

set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14")

add_executable(taco-bench ${TEST_SOURCES} bench.h)
target_link_libraries(taco-bench benchmark::benchmark)
target_link_libraries(taco-bench taco)
if(GRAPHBLAS)
  target_link_libraries(taco-bench graphblas)
endif(GRAPHBLAS)

