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

struct testConstructionAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    auto m1 = Union(Complement(regions[0]), Complement(regions[2]));
    auto m2 = Union(Complement(regions[1]), Complement(regions[2]));
    return Intersect(m1, m2);
  }
};

Func xorOp1("logical_xor", Boolean(), xorAlgebra());
Func andOp1("logical_and", Boolean(), andAlgebra());
Func xorAndOp("fused_xor_and", Boolean(), xorAndAlgebra());
Func testOp("test", Boolean(), testConstructionAlgebra());
static void bench_image_xor(benchmark::State& state, const Format& f) {
  int num = state.range(0);
  auto t1 = 0.5;
  auto t2 = 0.55;
  Tensor<int64_t> matrix1 = castToTypeZero<int64_t>("A", loadImageTensor("A", num, f, t1, 1 /* variant */));
  Tensor<int64_t> matrix2 = castToTypeZero<int64_t>("B", loadImageTensor("B", num, f, t2, 2 /* variant */));
  auto dims = matrix1.getDimensions();

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", dims, f, 1);
    IndexVar i("i"), j("j");
    result(i, j) = testOp(matrix1(i, j), matrix2(i, j));
    result.setAssembleWhileCompute(true);
    result.compile();
    state.ResumeTiming();
    result.compute();
    result = result.removeExplicitZeros(result.getFormat());

    int nnz = 0;
    for (auto& it : iterate<int64_t>(result)) {
      nnz++;
    }
//    std::cout << "Result NNZ = " << nnz << std::endl;
//    std::cout << result << std::endl;
  }
}
static void CustomArguments(benchmark::internal::Benchmark* b) {
  for (int i = 1; i <= 11; ++i)
      b->Args({i});
}
TACO_BENCH_ARGS(bench_image_xor, csr, CSR)->Apply(CustomArguments);

static void bench_image_fused(benchmark::State& state, const Format& f) {
  int num = state.range(0);
  auto t1 = 0.5;
  auto t2 = 0.55;
  Tensor<int64_t> matrix1 = castToTypeZero<int64_t>("A", loadImageTensor("A", num, f, t1, 1 /* variant */));
  Tensor<int64_t> matrix2 = castToTypeZero<int64_t>("B", loadImageTensor("B", num, f, t2, 2 /* variant */));
  Tensor<int64_t> matrix3 = castToTypeZero<int64_t>("C", loadImageTensor("C", num, f, 3 /* variant */));
  auto dims = matrix1.getDimensions();

//  write("temp/taco-mat1-" + std::to_string(num) + ".tns", matrix1);
//  write("temp/taco-mat2-" + std::to_string(num) + ".tns", matrix2);
//  write("temp/taco-mat3-" + std::to_string(num) + ".tns", matrix3);
//  int nnz = 0;
//  for (auto& it : iterate<int64_t>(matrix1)) {
//    nnz++;
//  }
//  std::cout << "Matrix1 NNZ = " << nnz << std::endl;
//  nnz = 0;
//  for (auto& it : iterate<int64_t>(matrix2)) {
//    nnz++;
//  }
//  std::cout << "Matrix2 NNZ = " << nnz << std::endl;
//  nnz = 0;
//  for (auto& it : iterate<int64_t>(matrix3)) {
//    nnz++;
//  }
//  std::cout << "Matrix3 NNZ = " << nnz << std::endl;

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", dims, f, 0);

    IndexVar i("i"), j("j");
    result(i, j) = xorAndOp(matrix1(i, j), matrix2(i, j), matrix3(i, j));
    IndexStmt stmt = result.getAssignment().concretize();
    result.setAssembleWhileCompute(true);
    result.compile();
    state.ResumeTiming();
    result.compute();
    result = result.removeExplicitZeros(result.getFormat());

//    int nnz = 0;
//    for (auto& it : iterate<int64_t>(result)) {
//      nnz++;
//    }
//    std::cout << "Result NNZ = " << nnz << std::endl;
//    write("temp/taco-result" + std::to_string(num) + ".tns", result);
    // Used to print out generated TACO code
//    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(std::cout, ir::CodeGen::ImplementationGen);
//    ir::Stmt compute = lower(stmt, "compute",  false, true);
//    codegen->compile(compute, true);
  }
}
TACO_BENCH_ARGS(bench_image_fused, csr, CSR)->Apply(CustomArguments);

static void bench_image_window(benchmark::State& state, const Format& f, double window_size) {
  int num = state.range(0);
  auto t1 = 0.5;
  auto t2 = 0.55;
  Tensor<int64_t> matrix1 = castToTypeZero<int64_t>("A", loadImageTensor("A", num, f, t1, 1 /* variant */));
  Tensor<int64_t> matrix2 = castToTypeZero<int64_t>("B", loadImageTensor("B", num, f, t2, 2 /* variant */));
  auto dims = matrix1.getDimensions();

  int mid0 = (dims[0]/2.0);
  int mid1 = (dims[1]/2.0);
  int win_len0 = int(window_size * dims[0]);
  int win_len1 = int(window_size * dims[1]);

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", {2*win_len0, 2*win_len1}, f, 0);

    IndexVar i("i"), j("j");
    result(i, j) = xorOp1(matrix1(i(mid0-win_len0, mid0+win_len0), j(mid1-win_len1, mid1+win_len1)),
                         matrix2(i(mid0-win_len0, mid0+win_len0), j(mid1-win_len1, mid1+win_len1)));
    IndexStmt stmt = result.getAssignment().concretize();
    result.setAssembleWhileCompute(true);
    result.compile();
    state.ResumeTiming();
    result.compute();
    result = result.removeExplicitZeros(result.getFormat());

//        int nnz = 0;
//    for (auto& it : iterate<int64_t>(result)) {
//      nnz++;
//    }
//    std::cout << "Result NNZ = " << nnz << std::endl;

//    write("temp/taco-result" + std::to_string(num) + ".tns", result);
    // Used to print out generated TACO code
//    std::shared_ptr<ir::CodeGen> codegen = ir::CodeGen::init_default(std::cout, ir::CodeGen::ImplementationGen);
//    ir::Stmt compute = lower(stmt, "compute",  false, true);
//    codegen->compile(compute, true);
  }
}
TACO_BENCH_ARGS(bench_image_window, csr/0.25, CSR, 0.25)->Apply(CustomArguments);
TACO_BENCH_ARGS(bench_image_window, csr/0.2, CSR, 0.2)->Apply(CustomArguments);
TACO_BENCH_ARGS(bench_image_window, csr/0.15, CSR, 0.15)->Apply(CustomArguments);
TACO_BENCH_ARGS(bench_image_window, csr/0.1, CSR, 0.1)->Apply(CustomArguments);