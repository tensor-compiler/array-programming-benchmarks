#include "bench.h"
#include "benchmark/benchmark.h"
#include "codegen/codegen_c.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"
#include "taco/lower/lower.h"

#include "codegen/codegen.h"

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

struct Boolean {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    taco_iassert(v.size() >= 1) << "Add operator needs at least one operand";
    return ir::Literal::make(int64_t(1), v[0].type());
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

struct xorAndAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    auto m1 = Intersect(regions[0], regions[2]);
    auto m2 = Intersect(regions[1], regions[2]);
    auto noIntersect = Complement(Intersect(Intersect(regions[0], regions[1]), regions[2]));
    return Intersect(noIntersect, Union(m1, m2));
  }
};

Func xorOp1("logical_xor", Boolean(), xorAlgebra());
Func andOp1("logical_and", Boolean(), andAlgebra());
Func xorAndOp("fused_xor_and", Boolean(), xorAndAlgebra());
static void bench_image_xor(benchmark::State& state, const Format& f) {
  int num = state.range(0);
  auto t1 = 0.5;
  auto t2 = 0.55;
  Tensor<int64_t> matrix1 = castToType<int64_t>("A", loadImageTensor("A", num, f, t1, 1 /* variant */));
  Tensor<int64_t> matrix2 = castToType<int64_t>("B", loadImageTensor("B", num, f, t2, 2 /* variant */));
  auto dims = matrix1.getDimensions();

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", dims, f, 1);
    IndexVar i("i"), j("j");
    result(i, j) = xorOp1(matrix1(i, j), matrix2(i, j));
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
    std::cout << result << std::endl;
  }
}
static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int i = 1; i <= 1; ++i)
      b->Args({i});
}
TACO_BENCH_ARGS(bench_image_xor, csr, CSR)->Apply(CustomArguments);

static void bench_image_fused(benchmark::State& state, const Format& f) {
  int num = state.range(0);
  auto t1 = 0.5;
  auto t2 = 0.55;
  Tensor<int64_t> matrix1 = castToType<int64_t>("A", loadImageTensor("A", num, f, t1, 1 /* variant */));
  Tensor<int64_t> matrix2 = castToType<int64_t>("B", loadImageTensor("B", num, f, t2, 2 /* variant */));
  Tensor<int64_t> matrix3 = castToType<int64_t>("C", loadImageTensor("C", num, f, 3 /* variant */));
  auto dims = matrix1.getDimensions();

  int nnz = 0;
  for (auto& it : iterate<int64_t>(matrix1)) {
    nnz++;
  }
  std::cout << "Matrix1 NNZ = " << nnz << std::endl;
  nnz = 0;
  for (auto& it : iterate<int64_t>(matrix2)) {
    nnz++;
  }
  std::cout << "Matrix2 NNZ = " << nnz << std::endl;
  nnz = 0;
  for (auto& it : iterate<int64_t>(matrix3)) {
    nnz++;
  }
  std::cout << "Matrix3 NNZ = " << nnz << std::endl;
  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", dims, f, 0);
    Tensor<int64_t> temp1("t1", dims, f, 0);
    Tensor<int64_t> temp2("t2", dims, f, 0);
    IndexVar i("i"), j("j");
//    temp1(i,j) = andOp1(matrix1(i, j), matrix3(i, j));
//    temp2(i,j) = andOp1(matrix2(i, j), matrix3(i, j));
//    result(i, j) = xorOp1(temp1(i,j), temp2(i,j));
//    result(i, j) = xorOp1(andOp1(matrix1(i, j), matrix3(i, j)), andOp1(matrix2(i, j), matrix3(i, j)));
    result(i, j) = xorAndOp(matrix1(i, j), matrix2(i, j), matrix3(i, j));
    IndexStmt stmt = result.getAssignment().concretize();
    result.setAssembleWhileCompute(true);
    result.compile();
    state.ResumeTiming();
    result.compute();
    temp1 = temp1.removeExplicitZeros(temp1.getFormat());
    temp2 = temp2.removeExplicitZeros(temp2.getFormat());
    result = result.removeExplicitZeros(result.getFormat());
    int nnz = 0;
    for (auto& it : iterate<int64_t>(result)) {
      nnz++;
    }

    std::cout << "Result NNZ = " << nnz << std::endl;
      std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(std::cout, ir::CodeGen::ImplementationGen);
      ir::Stmt compute = lower(stmt, "compute",  false, true);
      codegen->compile(compute, true);
//    std::cout << result << std::endl;
  }
}
TACO_BENCH_ARGS(bench_image_fused, csr, CSR)->Apply(CustomArguments);