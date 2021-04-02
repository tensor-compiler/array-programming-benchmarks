#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/index_notation/tensor_operator.h"

using namespace taco;

struct NoOp {
  ir::Expr operator()(const std::vector<ir::Expr>& v) {
    return 1;
  }
};

struct UnionAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Union(regions[0], regions[1]);
  }
};

struct IntersectAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Intersect(regions[0], regions[1]);
  }
};

struct AMinusBAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    auto nointersect = Complement(Intersect(regions[0], regions[1]));
    return Intersect(regions[0], nointersect);
  }
};

struct AAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Union(regions[0], Intersect(regions[0], regions[1]));
  }
};

struct XorAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    IterationAlgebra noIntersect = Complement(Intersect(regions[0], regions[1]));
    return Intersect(noIntersect, Union(regions[0], regions[1]));
  }
};

Func unionOp("union", NoOp(), UnionAlgebra());
Func intersectOp("intersect", NoOp(), IntersectAlgebra());
Func aMinusB("aMinusB", NoOp(), AMinusBAlgebra());
Func aBIntersectOp("aBIntersect", NoOp(), AAlgebra());
Func aubminusanb("xor", NoOp(), XorAlgebra());
static void bench_iteration_spaces(benchmark::State& state, Func op) {
  int dim = state.range(0);
  auto sparsity = 0.01;
  auto f = CSR;
  Tensor<double> matrix = loadRandomTensor("A", {dim, dim}, sparsity, f);
  Tensor<double> matrix1 = loadRandomTensor("B", {dim, dim}, sparsity, f, 1 /* variant */);

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<double> result("result", {dim, dim}, f);
    IndexVar i("i"), j("j");
    result(i, j) = op(matrix(i, j), matrix1(i, j));
    // result.setAssembleWhileCompute(true);
    result.compile();
    std::cout << result.getSource() << std::endl;
    state.ResumeTiming();
    // result.compute();
  }
}
std::vector<std::vector<int64_t>> sizes = {{1000, 2500, 5000, 10000, 20000, 40000}};
TACO_BENCH_ARGS(bench_iteration_spaces, aub, unionOp)->ArgsProduct(sizes);
TACO_BENCH_ARGS(bench_iteration_spaces, anb, intersectOp)->ArgsProduct(sizes);
TACO_BENCH_ARGS(bench_iteration_spaces, a-b, aMinusB)->ArgsProduct(sizes);
TACO_BENCH_ARGS(bench_iteration_spaces, auanb, aBIntersectOp)->ArgsProduct(sizes);
TACO_BENCH_ARGS(bench_iteration_spaces, aub-anb, aubminusanb)->ArgsProduct(sizes);
