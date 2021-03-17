#if GRAPHBLAS

#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"

extern "C" {
#include "GraphBLAS.h"
}

#include <vector>
#include <limits>

using namespace taco;

ir::Expr addImpl(const std::vector<ir::Expr>& v) {
  return ir::Add::make(v[0], v[1]);
}
Func AddOp("add", addImpl, {Annihilator(std::numeric_limits<double>::infinity()), Identity(0), Commutative(), Associative()});

ir::Expr minImpl(const std::vector<ir::Expr>& v) {
  return ir::Min::make(v[0], v[1]);
}
Func MinOp("min", minImpl, {Identity(std::numeric_limits<double>::infinity()), Commutative(), Associative()});

ir::Expr maskImpl(const std::vector<ir::Expr>& v) {
  return v[0];
}
struct MaskAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& r) {
    return Intersect(r[0], Complement(r[1]));
  }
};
Func MaskOp("mask", maskImpl, MaskAlgebra());

//static void bench_mxv_taco(benchmark::State& state) {
//  Format dv({Dense});
//
//  Tensor<double> T = read("/data/scratch/s3chou/formats-bench/data/webbase_1M.mtx", CSR);
//  Tensor<double> A(T.getDimensions(), CSR, std::numeric_limits<double>::infinity());
//  for (const auto& c : T) {
//    A.insert(c.first.toVector(), c.second);
//  }
//  A.pack();
//
//  // TODO: Only run for square matrices
//
//  Tensor<double> x({A.getDimension(1)}, dv, std::numeric_limits<double>::infinity());
//  x.insert({0}, 0.0);
//  x.pack();
//
//  IndexVar i, j;
//
//  taco_set_num_threads(12);
//  for (auto _ : state) {
//    state.PauseTiming();
//
//    Tensor<double> y({A.getDimension(0)}, dv, std::numeric_limits<double>::infinity());
//    y(i) = Reduction(MinOp(), j, AddOp(A(i,j), x(j)));
//    //y(i) = MinOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
//    //y(i) = MaskOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
//    //y(i) = MinOp(MaskOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i)), x(i));
//    //y(i) = MaskOp(MinOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i)), x(i));
//    //y(i) = MinOp(FilterOp(x(i)) * Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
//
//    y.compile();
//    y.assemble();
//
//    state.ResumeTiming();
//
//    y.compute();
//  }
//  taco_set_num_threads(1);
//}
//TACO_BENCH(bench_mxv_taco);

//static void bench_mxv_suitesparse(benchmark::State& state) {
//  GrB_init(GrB_BLOCKING);
//  GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
//  GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
//
//  int nthreads_max = 12;
//  GxB_Global_Option_set(GxB_NTHREADS, nthreads_max);
//
//  Tensor<double> T = read("/data/scratch/s3chou/formats-bench/data/webbase_1M.mtx", CSR);
//  GrB_Index M = T.getDimension(0);
//  GrB_Index N = T.getDimension(1);
//  GrB_Matrix A;
//  GrB_Matrix_new(&A, GrB_FP64, M, N);
//  std::vector<GrB_Index> I, J;
//  std::vector<double> V;
//  for (const auto& c : T) {
//    I.push_back(c.first[0]);
//    J.push_back(c.first[1]);
//    V.push_back(c.second);
//  }
//  GrB_Matrix_build_FP64(A, I.data(), J.data(), V.data(), V.size(), GrB_PLUS_FP64);
//  //GrB_Index nnz;
//  //GrB_Matrix_nvals(&nnz, A);
//  
//  GrB_Vector x;
//  GrB_Vector_new(&x, GrB_FP64, N);
//  GrB_Vector_assign_FP64(x, NULL, NULL, 1, GrB_ALL, N, NULL);
//  //GrB_Vector_setElement_FP64(
//
//  GrB_Vector y;
//  GrB_Vector_new(&y, GrB_FP64, M);
//  //GrB_Vector_assign_FP64(y, NULL, NULL, 0, GrB_ALL, M, NULL);
//
//  GrB_Descriptor desc;
//  GrB_Descriptor_set (desc, GrB_OUTP, GrB_REPLACE);
//
//  for (auto _ : state) {
//    GrB_mxv(y, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP64, A, x, desc);
//    //GrB_vxm(x, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP64, x, A, desc);
//  }
//}

Format dv({Dense});
int nthreads = 4;

struct GraphBLASFixture {
  GraphBLASFixture() {
    const auto path = "/data/scratch/s3chou/formats-bench/data/webbase_1M.mtx";
    Tensor<double> T = read(path, CSR);

    // TODO: Only run for square matrices

    A_trop_taco = Tensor<double>(T.getDimensions(), CSR, std::numeric_limits<double>::infinity());

    GrB_init(GrB_BLOCKING);
    GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
    GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
    GxB_Global_Option_set(GxB_NTHREADS, nthreads);
  
    GrB_Index M = T.getDimension(0);
    GrB_Index N = T.getDimension(1);
    GrB_Matrix_new(&A_trop_gb, GrB_FP64, M, N);

    std::vector<GrB_Index> I, J;
    std::vector<double> V;
    for (const auto& c : T) {
      I.push_back(c.first[0]);
      J.push_back(c.first[1]);
      V.push_back(c.second);
      A_trop_taco.insert(c.first.toVector(), c.second);
    }
    GrB_Matrix_build_FP64(A_trop_gb, I.data(), J.data(), V.data(), V.size(), GrB_PLUS_FP64);
    A_trop_taco.pack();
  
    GrB_Vector_new(&x_trop_gb, GrB_FP64, N);
    GrB_Vector_assign_FP64(x_trop_gb, NULL, NULL, 1, GrB_ALL, N, NULL);
  
    x_trop_taco = Tensor<double>({T.getDimension(1)}, dv, std::numeric_limits<double>::infinity());
    x_trop_taco.insert({0}, 0.0);
    x_trop_taco.pack();
  }

  GrB_Matrix A_trop_gb;
  GrB_Vector x_trop_gb;
  Tensor<double> A_trop_taco;
  Tensor<double> x_trop_taco;
};

GraphBLASFixture fixture;

static void bench_mxv_suitesparse(benchmark::State& state) {
  GrB_init(GrB_BLOCKING);
  GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
  GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
  GxB_Global_Option_set(GxB_NTHREADS, nthreads);

  GrB_Descriptor desc;
  GrB_Descriptor_set (desc, GrB_OUTP, GrB_REPLACE);
  
  GrB_Vector y = NULL;;
  for (auto _ : state) {
    state.PauseTiming();

    GrB_Vector_free(&y);

    state.ResumeTiming();

    GrB_Vector_new(&y, GrB_FP64, fixture.A_trop_taco.getDimension(0));
    GrB_mxv(y, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP64, fixture.A_trop_gb, fixture.x_trop_gb, desc);
    //GrB_vxm(x, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP64, x, A, desc);
  }
  GrB_Vector_free(&y);
}

static void bench_mxv_taco(benchmark::State& state) {
  taco_set_num_threads(nthreads);
  for (auto _ : state) {
    state.PauseTiming();

    IndexVar i, j;
    Tensor<double> y({fixture.A_trop_taco.getDimension(0)}, dv, std::numeric_limits<double>::infinity());
    //y(i) = Reduction(MinOp(), j, AddOp(fixture.A_trop_taco(i,j), fixture.x_trop_taco(j)));
    y(i) = MaskOp(Reduction(MinOp(), j, AddOp(fixture.A_trop_taco(i,j), fixture.x_trop_taco(j))), fixture.x_trop_taco(i));
    //y(i) = MinOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
    //y(i) = MaskOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
    //y(i) = MinOp(MaskOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i)), x(i));
    //y(i) = MaskOp(MinOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i)), x(i));
    //y(i) = MinOp(FilterOp(x(i)) * Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));

    y.compile();

    state.ResumeTiming();

    y.assemble();
    y.compute();
  }
  taco_set_num_threads(1);
}

GRAPHBLAS_BENCH(bench_mxv_suitesparse, 1000);
GRAPHBLAS_BENCH(bench_mxv_taco, 1000);

#endif
