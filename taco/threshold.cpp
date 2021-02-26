#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;

// bench_add_sparse_threshold adds two tensors of a given dimension and sparsity.
// It does this range on the TACO side to see when TACO becomes faster than NumPY.
static void bench_add_sparse_threshold(benchmark::State& state, float sparsity) {
  int dim = state.range(0);
  Tensor<float> matrix1("A", {dim, dim}, CSR);
  Tensor<float> matrix2("B", {dim, dim}, CSR);

  srand(4357);
  // TODO (rohany): Move this into a helper method.
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < sparsity) {
        matrix1.insert({i, j}, (float) ((int) (rand_float*3/sparsity)));
      }

      rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < sparsity) {
        matrix2.insert({i, j}, (float) ((int) (rand_float*3/sparsity)));
      }
    }
  }
  matrix1.pack(); matrix2.pack();


  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> result("C", {dim, dim}, CSR);
    IndexVar i, j;
    result(i, j) = matrix1(i, j) + matrix2(i, j);
    result.compile();
    result.assemble();
    state.ResumeTiming();
    // The actual computation.
    result.compute();
  }
}
static void applyBenchSizes(benchmark::internal::Benchmark* b) {
  b->ArgsProduct({{250, 500, 750, 1000, 2500, 5000, 7500, 8000}});
}
TACO_BENCH_ARG(bench_add_sparse_threshold, 0.001, 0.001)->Apply(applyBenchSizes);
TACO_BENCH_ARG(bench_add_sparse_threshold, 0.01, 0.01)->Apply(applyBenchSizes);
TACO_BENCH_ARG(bench_add_sparse_threshold, 0.05, 0.05)->Apply(applyBenchSizes);
TACO_BENCH_ARG(bench_add_sparse_threshold, 0.25, 0.25)->Apply(applyBenchSizes);
