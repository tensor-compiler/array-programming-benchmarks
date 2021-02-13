#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"

using namespace taco;

static void BM_basic_windowing(benchmark::State& state) {
  auto dim = 10000;
  auto sparsity = 0.01;
  // CSR format.
  Tensor<float> matrix("A", {dim, dim}, {Dense, Sparse});
  Tensor<float> result("B", {dim-2, dim-2}, {Dense, Sparse});

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

  IndexVar i, j;
  result(i, j) = matrix(i(1, dim-1), j(1, dim-1)) + matrix(i(1, dim-1), j(1, dim-1));
  result.compile();
  result.assemble();

  for (auto _ : state) {
    // This code gets timed. Setup goes outside the loop.
    result.compute();
  }
}
// Have benchmarking report milliseconds and run for 10 iterations.
BENCHMARK(BM_basic_windowing)->Unit(benchmark::kMillisecond)->Iterations(10);

