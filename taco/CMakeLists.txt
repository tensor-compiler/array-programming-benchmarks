cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(taco-bench)
set(CMAKE_BUILD_TYPE "release")
option(GRAPHBLAS "Build GraphBLAS benchmarks" OFF)
option(LANKA "Build benchmarks for the Lanka cluster." OFF)

if(GRAPHBLAS)
  message("-- Will build GraphBLAS benchmarks")
  add_definitions(-DGRAPHBLAS)
endif(GRAPHBLAS)

add_subdirectory(taco)
add_subdirectory(benchmark)
if(GRAPHBLAS)
  add_subdirectory(suitesparse)
endif(GRAPHBLAS)

include_directories(taco taco/include benchmark/include suitesparse/Include taco/src)

file(GLOB TEST_SOURCES *.cpp *.h)

set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14")
if (GRAPHBLAS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp -fpermissive -fexcess-precision=fast -fno-math-errno -fwrapv")
endif(GRAPHBLAS)

add_executable(taco-bench ${TEST_SOURCES} bench.h)
target_link_libraries(taco-bench benchmark::benchmark)
target_link_libraries(taco-bench taco)
# Only link stdc++fs on lanka. Mac's don't need this set.
if (LANKA)
  target_link_libraries(taco-bench stdc++fs)
endif(LANKA)
if(GRAPHBLAS)
  target_link_libraries(taco-bench graphblas)
endif(GRAPHBLAS)

