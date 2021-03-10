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

struct AddImpl {
  ir::Expr operator()(const std::vector<ir::Expr>& v) {
    return ir::Add::make(v[0], v[1]);
  }
};
Func AddOp("add", AddImpl(), {Annihilator(std::numeric_limits<double>::infinity()), Identity(0), Commutative(), Associative()});

struct MinImpl{
  ir::Expr operator()(const std::vector<ir::Expr>& v) {
    return ir::Min::make(v[0], v[1]);
  }
};
Func MinOp("min", MinImpl(), {Identity(std::numeric_limits<double>::infinity()), Commutative(), Associative()});

struct MaskImpl {
  ir::Expr operator()(const std::vector<ir::Expr>& v) {
    return v[0];
  }
};
struct MaskAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& r) {
    return Intersect(r[0], Complement(r[1]));
  }
};
Func MaskOp("mask", MaskImpl(), MaskAlgebra());

static void bench_mxv_taco(benchmark::State& state) {
  Format dv({Dense});

  Tensor<double> T = read("/data/scratch/s3chou/formats-bench/data/webbase_1M.mtx", CSR);
  Tensor<double> A(T.getDimensions(), CSR, std::numeric_limits<double>::infinity());
  for (const auto& c : T) {
    A.insert(c.first.toVector(), c.second);
  }
  A.pack();

  // TODO: Only run for square matrices

  Tensor<double> x({A.getDimension(1)}, dv, std::numeric_limits<double>::infinity());
  x.insert({0}, 0.0);
  x.pack();

  IndexVar i, j;

  taco_set_num_threads(12);
  for (auto _ : state) {
    state.PauseTiming();

    Tensor<double> y({A.getDimension(0)}, dv, std::numeric_limits<double>::infinity());
    y(i) = MinOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
    //y(i) = MaskOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
    //y(i) = MinOp(MaskOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i)), x(i));
    //y(i) = MaskOp(MinOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i)), x(i));
    //y(i) = MinOp(FilterOp(x(i)) * Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));

    y.compile();
    y.assemble();

    state.ResumeTiming();

    y.compute();
  }
  taco_set_num_threads(1);
}

static void bench_mxv_suitesparse(benchmark::State& state) {
  GrB_init(GrB_BLOCKING);

  Tensor<double> T = read("/data/scratch/s3chou/formats-bench/data/webbase_1M.mtx", CSR);

  for (const auto& c : T) {
    //A.insert(c.first.toVector(), c.second);
  }

  GrB_Vector x = nullptr;
  GrB_Index n;
  GrB_Vector_new(&x, GrB_FP64, n);

  for (auto _ : state) {
  }
}

TACO_BENCH(bench_mxv_taco);
TACO_BENCH(bench_mxv_suitesparse);

#endif
