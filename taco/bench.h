#ifndef TACO_BENCH_BENCH_H
#define TACO_BENCH_BENCH_H

#include "benchmark/benchmark.h"
#include "taco/tensor.h"

// Register a benchmark with the following options:
// * Millisecond output display
// * 10 data points
// * Reporting of avg/stddev/median
// * Wall-clock time, rather than CPU time.
#define TACO_BENCH(bench)         \
  BENCHMARK(bench)                \
  ->Unit(benchmark::kMillisecond) \
  ->Repetitions(10)               \
  ->Iterations(1)                 \
  ->ReportAggregatesOnly(true)    \
  ->UseRealTime()

// TACO_BENCH_ARG is similar to TACO_BENCH but allows for passing
// of an arbitrarily typed argument to the benchmark function.
// TODO (rohany): Make this take in only 1 argument.
// TODO (rohany): Don't specify the time here, but do it at the command line.

#define TACO_BENCH_ARG(bench, name, arg)  \
  BENCHMARK_CAPTURE(bench, name, arg)     \
  ->Unit(benchmark::kMicrosecond)         \
  ->Repetitions(10)                       \
  ->Iterations(1)                         \
  ->ReportAggregatesOnly(true)            \
  ->UseRealTime()

#define TACO_BENCH_ARGS(bench, name, ...)       \
  BENCHMARK_CAPTURE(bench, name, __VA_ARGS__)   \
  ->Unit(benchmark::kMicrosecond)               \
  ->Repetitions(10)                             \
  ->Iterations(1)                               \
  ->ReportAggregatesOnly(true)                  \
  ->UseRealTime()

std::string getTacoTensorPath();
taco::TensorBase loadRandomTensor(std::string name, std::vector<int> dims, float sparsity, taco::Format format);

template<typename T, typename T2>
taco::Tensor<T> shiftLastMode(std::string name, taco::Tensor<T2> original) {
  taco::Tensor<T> result(name, original.getDimensions(), original.getFormat());
  std::vector<int> coords(original.getOrder());
  for (auto& value : taco::iterate<T2>(original)) {
    for (int i = 0; i < original.getOrder(); i++) {
      coords[i] = value.first[i];
    }
    int lastMode = original.getOrder() - 1;
    coords[lastMode] = (coords[lastMode] + 1) % original.getDimension(lastMode);
    // TODO (rohany): Temporarily use a constant value here.
    result.insert(coords, T(2));
  }
  result.pack();
  return result;
}

template<typename T>
taco::Tensor<T> readIntoType(std::string name, std::string path, taco::ModeFormat format) {
  auto tensor = taco::read(path, format);
  taco::Tensor<T> result(name, tensor.getDimensions(), tensor.getFormat());
  std::vector<int> coords(tensor.getOrder());
  for (auto& value : taco::iterate<double>(tensor)) {
    for (int i = 0; i < tensor.getOrder(); i++) {
      coords[i] = value.first[i];
    }
    result.insert(coords, T(value.second));
  }
  result.pack();
  return result;
}

#endif //TACO_BENCH_BENCH_H
