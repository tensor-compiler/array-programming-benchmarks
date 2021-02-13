cmake_minimum_required(VERSION 2.8.12 FATAL_ERROR)
project(taco-bench)
set(CMAKE_BUILD_TYPE "release")

add_subdirectory(taco)
add_subdirectory(benchmark)

include_directories(taco taco/include benchmark/include)

file(GLOB TEST_SOURCES *.cpp)

set(CMAKE_CXX_FLAGS "${C_CXX_FLAGS} -std=c++14")

add_executable(taco-bench ${TEST_SOURCES})
target_link_libraries(taco-bench benchmark::benchmark)
target_link_libraries(taco-bench taco)

