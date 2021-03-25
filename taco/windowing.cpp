#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;

// The tensor sizes that we want to run each windowing benchmark with.
std::vector<int64_t> tensorSizes({5000, 10000, 20000});

// WindowConfig corresponds to what sort of window size should be used
// when evaluating standard windowing benchmarks. This should stay in
// line with the definition in numpy/windowing.py.
enum WindowConfig {
  Constant,
  ConstantFraction,
  AlmostWhole,
  Whole,
  NoWindowing,
};

#define FOREACH_WINDOW_CONFIG(__func__) \
  __func__(Constant, Constant) \
  __func__(ConstantFraction, ConstantFraction) \
  __func__(AlmostWhole, AlmostWhole) \
  __func__(Whole, Whole) \
  __func__(NoWindowing, NoWindowing)

Tensor<double> windowedTensorOp(Tensor<double> input1, Tensor<double> input2, int dim, WindowConfig config) {
  IndexVar i, j;
  switch (config) {
    case Constant: {
      Tensor<double> result("B", {500, 500}, input1.getFormat());
      result(i, j) = input1(i(250, 750), j(250, 750)) + input2(i(250, 750), j(250, 750));
      return result;
    }
    case ConstantFraction: {
      int size = dim / 4;
      int start = dim / 4;
      Tensor<double> result("B", {size, size}, input1.getFormat());
      result(i, j) = input1(i(start, start + size), j(start, start + size)) + input2(i(start, start + size), j(start, start + size));
      return result;
    }
    case AlmostWhole: {
      Tensor<double> result("B", {dim-2, dim-2}, input1.getFormat());
      result(i, j) = input1(i(1, dim-1), j(1, dim-1)) + input2(i(1, dim-1), j(1, dim-1));
      return result;
    }
    case Whole: {
      Tensor<double> result("B", {dim, dim}, input1.getFormat());
      result(i, j) = input1(i(0, dim), j(0, dim)) + input2(i(0, dim), j(0, dim));
      return result;
    }
    case NoWindowing: {
      Tensor<double> result("B", {dim, dim}, input1.getFormat());
      result(i, j) = input1(i, j) + input2(i, j);
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
  Tensor<double> matrix2 = loadRandomTensor("A2", {dim, dim}, sparsity, f, 1 /* variant */);
  matrix.pack();

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    auto result = windowedTensorOp(matrix, matrix2, dim, config);
    result.setAssembleWhileCompute(true);
    result.compile();
    state.ResumeTiming();
    // The actual computation.
    result.compute();
  }
}

#define DECLARE_ADD_SPARSE_WINDOW_BENCH(configName, config) \
  TACO_BENCH_ARGS(bench_add_sparse_window, csr/configName, CSR, config)->ArgsProduct({tensorSizes});

FOREACH_WINDOW_CONFIG(DECLARE_ADD_SPARSE_WINDOW_BENCH)

static void bench_add_sparse_strided_window(benchmark::State& state, const Format& f) {
  int dim = state.range(0);
  int strideWidth = state.range(1);
  auto sparsity = 0.01;
  Tensor<double> matrix = loadRandomTensor("A", {dim, dim}, sparsity, f);
  Tensor<double> matrix2 = loadRandomTensor("A2", {dim, dim}, sparsity, f, 1 /* variant */);
  matrix.pack();

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<double> result("B", {dim/strideWidth, dim/strideWidth}, f);
    IndexVar i, j;
    result(i, j) = matrix(i(0, dim, strideWidth), j(0, dim, strideWidth)) + matrix2(i(0, dim, strideWidth), j(0, dim, strideWidth));
    result.setAssembleWhileCompute(true);
    result.compile();
    state.ResumeTiming();
    // The actual computation.
    result.compute();
  }
}
std::vector<int64_t> strides({2, 4, 8});
TACO_BENCH_ARG(bench_add_sparse_strided_window, csr, CSR)
  ->ArgsProduct({tensorSizes, strides});
TACO_BENCH_ARG(bench_add_sparse_strided_window, csc, CSC)
  ->ArgsProduct({tensorSizes, strides});

static void bench_add_sparse_index_set(benchmark::State& state, const Format& f) {
  int dim = state.range(0);
  int fraction = state.range(1);
  auto sparsity = 0.01;
  Tensor<double> matrix = loadRandomTensor("A", {dim, dim}, sparsity, f);
  Tensor<double> matrix2 = loadRandomTensor("A2", {dim, dim}, sparsity, f, 1 /* variant */);
  std::vector<int> indexSet;
  for (int i = 0; i < dim / fraction; i++) {
    indexSet.push_back(i * fraction);
  }

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<double> result("B", {dim, dim / fraction}, f);
    IndexVar i("i"), j("j");
    result(i, j) = matrix(i, j(indexSet)) + matrix2(i, j(indexSet));
    result.setAssembleWhileCompute(true);
    result.compile();
    state.ResumeTiming();

    result.compute();
  }
}
// Not doing index sets currently.
// std::vector<int64_t> fractions({2, 4, 8});
// TACO_BENCH_ARG(bench_add_sparse_index_set, csr, CSR)
//   ->ArgsProduct({tensorSizes, fractions});
