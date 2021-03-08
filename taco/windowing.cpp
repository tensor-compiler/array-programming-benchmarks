#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;

// applyBenchSizes is used to configure the benchmarks to run with the
// input arguments.
static void applyBenchSizes(benchmark::internal::Benchmark* b) {
  // Currently considering these size square tensors.
  b->ArgsProduct({{5000, 10000, 20000}});
}

// WindowConfig corresponds to what sort of window size should be used
// when evaluating standard windowing benchmarks. This should stay in
// line with the definition in numpy/windowing.py.
enum WindowConfig {
  Constant,
  ConstantFraction,
  AlmostWhole,
  Whole,
  NoWindow,
};

#define FOREACH_WINDOW_CONFIG(__func__) \
  __func__(Constant, Constant) \
  __func__(ConstantFraction, ConstantFraction) \
  __func__(AlmostWhole, AlmostWhole) \
  __func__(Whole, Whole) \
  __func__(NoWindow, NoWindow)

Tensor<double> windowedTensorOp(Tensor<double> input, int dim, WindowConfig config) {
  IndexVar i, j;
  switch (config) {
    case Constant: {
      Tensor<double> result("B", {500, 500}, input.getFormat());
      result(i, j) = input(i(250, 750), j(250, 750)) + input(i(250, 750), j(250, 750));
      return result;
    }
    case ConstantFraction: {
      int size = dim / 4;
      int start = dim / 4;
      Tensor<double> result("B", {size, size}, input.getFormat());
      result(i, j) = input(i(start, start + size), j(start, start + size)) + input(i(start, start + size), j(start, start + size));
      return result;
    }
    case AlmostWhole: {
      Tensor<double> result("B", {dim-2, dim-2}, input.getFormat());
      result(i, j) = input(i(1, dim-1), j(1, dim-1)) + input(i(1, dim-1), j(1, dim-1));
      return result;
    }
    case Whole: {
      Tensor<double> result("B", {dim, dim}, input.getFormat());
      result(i, j) = input(i(0, dim), j(0, dim)) + input(i(0, dim), j(0, dim));
      return result;
    }
    case NoWindow: {
      Tensor<double> result("B", {dim, dim}, input.getFormat());
      result(i, j) = input(i, j) + input(i, j);
      return result;
    }
    default:
      assert(false);
  }
}

static void bench_add_sparse_window(benchmark::State& state, const Format& f, WindowConfig config) {
  int dim = state.range(0);
  auto sparsity = 0.01;
  Tensor<double> matrix = loadRandomTensor("A", {dim, dim}, sparsity, f);
  matrix.pack();

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    auto result = windowedTensorOp(matrix, dim, config);
    result.compile();
    result.assemble();
    state.ResumeTiming();
    // The actual computation.
    result.compute();
  }
}

#define DECLARE_ADD_SPARSE_WINDOW_BENCH(configName, config) \
  TACO_BENCH_ARGS(bench_add_sparse_window, csr/configName, CSR, config)->Apply(applyBenchSizes); \
  TACO_BENCH_ARGS(bench_add_sparse_window, csc/configName, CSC, config)->Apply(applyBenchSizes);

FOREACH_WINDOW_CONFIG(DECLARE_ADD_SPARSE_WINDOW_BENCH)

static void bench_add_sparse_strided_window(benchmark::State& state, const Format& f) {
  int dim = state.range(0);
  auto sparsity = 0.01;
  Tensor<double> matrix = loadRandomTensor("A", {dim, dim}, sparsity, f);
  matrix.pack();

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<double> result("B", {(dim-2)/4, (dim-2)/4}, f);
    IndexVar i, j;
    result(i, j) = matrix(i(1, dim-1, 4), j(1, dim-1, 4)) + matrix(i(1, dim-1, 4), j(1, dim-1, 4));
    result.compile();
    result.assemble();
    state.ResumeTiming();
    // The actual computation.
    result.compute();
  }
}

// Have benchmarking report milliseconds and run for 10 iterations.
// Run an instance with both CSR and CSC formats.
TACO_BENCH_ARG(bench_add_sparse_strided_window, csr, CSR)->Apply(applyBenchSizes);
TACO_BENCH_ARG(bench_add_sparse_strided_window, csc, CSC)->Apply(applyBenchSizes);
