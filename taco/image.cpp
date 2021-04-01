#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"

using namespace taco;

// XOR Op and Algebra
struct GeneralAdd {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() >= 1) << "Add operator needs at least one operand";
    if (v.size() == 1)
        return ir::Add::make(v[0], ir::Literal::zero(v[0].type()));
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

struct andAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Intersect(regions[0], regions[1]);
  }
};

Func xorOp1("logical_xor", GeneralAdd(), xorAlgebra());
Func andOp1("logical_and", GeneralAdd(), andAlgebra());

static void bench_imaging_xor(benchmark::State& state, const Format& f) {
  int dim = state.range(0);
  auto sparsity = 0.01;
  Tensor<int64_t> matrix = castToType<int64_t>("A", loadImageTensor("A", 0, f));
  Tensor<int64_t> matrix1 = castToType<int64_t>("B", loadImageTensor("B", 0, f, 1 /* variant */));

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", {dim, dim}, f, 1);
    IndexVar i("i"), j("j");
    result(i, j) = xorOp1(matrix(i, j), matrix1(i, j));
    result.setAssembleWhileCompute(true);
    result.compile();
    state.ResumeTiming();
    result.compute();
    result = result.removeExplicitZeros(result.getFormat());
    
    int nnz = 0;
    for (auto& it : iterate<int64_t>(result)) {
      nnz++;
    }
    std::cout << "Result NNZ = " << nnz << std::endl;

  }
}
 TACO_BENCH_ARGS(bench_imaging_xor, csr, CSR)
   ->ArgsProduct({{5000, 10000, 20000}});

static void bench_imaging_fused(benchmark::State& state, const Format& f) {
  int dim = state.range(0);
  auto sparsity = 0.01;
  Tensor<int64_t> matrix = castToType<int64_t>("A", loadImageTensor("A", 0, f));
  Tensor<int64_t> matrix1 = castToType<int64_t>("B", loadImageTensor("B", 0, f, 1 /* variant */));
  Tensor<int64_t> matrix2 = castToType<int64_t>("C", loadImageTensor("C", 0, f, 2 /* variant */));

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", {dim, dim}, f, 1);
    IndexVar i("i"), j("j");
    result(i, j) = xorOp1(andOp1(matrix(i, j), matrix2(i, j)), andOp1(matrix1(i, j), matrix2(i, j)));
    result.setAssembleWhileCompute(false);
    result.compile();
    state.ResumeTiming();
    result.assemble();
    result.compute();
    result = result.removeExplicitZeros(result.getFormat());

    int nnz = 0;
    for (auto& it : iterate<int64_t>(result)) {
      nnz++;
    }
    std::cout << "Result NNZ = " << nnz << std::endl;
  }
}
 TACO_BENCH_ARGS(bench_imaging_fused, csr, CSR)
   ->ArgsProduct({{5000, 10000, 20000}});
