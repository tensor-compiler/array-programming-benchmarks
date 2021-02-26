#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"

using namespace taco;

struct GeneralAdd {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() >= 2) << "Add operator needs at least two operands";
    ir::Expr add = ir::Add::make(v[0], v[1]);
    for (size_t idx = 2; idx < v.size(); ++idx) {
      add = ir::Add::make(add, v[idx]);
    }
    return add;
  }
};

struct xorAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    IterationAlgebra noIntersect = Complement(Intersect(regions[0], regions[1]));
    return Intersect(noIntersect, Union(regions[0], regions[1]));
  }
};

// TODO (rohany): We can extract most of the logic out of here and parametrize the test by
//  a particular Func.
static void bench_xor_sparse(benchmark::State& state, float sparsity) {
  int dim = state.range(0);

  Func xorOp("xor", GeneralAdd(), xorAlgebra());

  // TODO (rohany): We can parametrize over the sparsities here.
  Tensor<int> A("A", {dim, dim}, CSR);
  Tensor<int> B("B", {dim, dim}, CSR);

  srand(4357);
  for (int i = 0; i < dim; i++) {
    for (int j = 0; j < dim; j++) {
      float rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < sparsity) {
        A.insert({i, j}, 1);
      }

      rand_float = (float)rand()/(float)(RAND_MAX);
      if (rand_float < sparsity) {
        B.insert({i, j}, 1);
      }
    }
  }
  A.pack(); B.pack();

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> result("C", {dim, dim}, CSR);
    IndexVar i, j;
    result(i, j) = xorOp(A(i, j), B(i, j));
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
TACO_BENCH_ARG(bench_xor_sparse, 0.01, 0.01)->Apply(applyBenchSizes);
