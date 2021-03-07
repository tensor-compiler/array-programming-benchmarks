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

static void bench_add_sparse_window(benchmark::State& state, const Format& f) {
  int dim = state.range(0);
  auto sparsity = 0.01;
  Tensor<double> matrix = loadRandomTensor("A", {dim, dim}, sparsity, f);
  matrix.pack();

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<double> result("B", {dim-2, dim-2}, f);
    IndexVar i, j;
    result(i, j) = matrix(i(1, dim-1), j(1, dim-1)) + matrix(i(1, dim-1), j(1, dim-1));
    result.compile();
    result.assemble();
    state.ResumeTiming();
    // The actual computation.
    result.compute();
  }
}

// Have benchmarking report milliseconds and run for 10 iterations.
// Run an instance with both CSR and CSC formats.
TACO_BENCH_ARG(bench_add_sparse_window, csr, CSR)->Apply(applyBenchSizes);
TACO_BENCH_ARG(bench_add_sparse_window, csc, CSC)->Apply(applyBenchSizes);

static void bench_add_sparse_strided_window(benchmark::State& state, const Format& f) {
  int dim = state.range(0);
  auto sparsity = 0.01;
  Tensor<float> matrix("A", {dim, dim}, f);

  srand(4357);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < sparsity) {
        matrix.insert({i, j}, (float) ((int) (rand_float*3/sparsity)));
      }
    }
  }
  matrix.pack();


  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> result("B", {(dim-2)/4, (dim-2)/4}, f);
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
