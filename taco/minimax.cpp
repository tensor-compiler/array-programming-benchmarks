#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"

#include "taco/util/env.h"

using namespace taco;

struct Min {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    if (v.size() == 1) {
      return v[0];
    }
    return ir::Min::make(v);
  }
};

struct Max {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    if (v.size() == 1) {
      return v[0];
    }
    return ir::Max::make(v);
  }
};

// We don't need to provide algebras since we're only iterating over one tensor.
Func minOp("min", Min());
Func maxOp("max", Max());

IndexExpr genMinMaxExpr(Tensor<double>& game, std::vector<IndexVar>& indexVars, int index) {
  Func op = (index % 2 == 0) ? maxOp : minOp;
  if (index == (game.getOrder() - 1)) {
    std::vector<IndexVar> slice;
    for (int i = 0; i <= index; i++) {
      slice.push_back(indexVars[i]);
    }
    return Reduction(op(), indexVars[index], game(slice));
  }
  return Reduction(op(), indexVars[index], genMinMaxExpr(game, indexVars, index + 1));
}

static void bench_minimax(benchmark::State& state) {
  auto order_str = getEnvVar("MINMAX_ORDER");
  if (order_str == "") {
    state.error_occurred();
    return;
  }
  int order = std::stoi(order_str) + 2;

  state.counters["order"] = order - 2;

  std::vector<ModeFormatPack> modes(order, Sparse);
  Format f(modes);
  taco::Tensor<double> game = loadMinMaxTensor("A", order, f);

  // This benchmark needs this hack activated to generate correct code.
  if(util::getFromEnv("TACO_CONCRETIZE_HACK", "0") == "0") {
    state.SkipWithError("must set TACO_CONCRETIZE_HACK=1");
    return;
  }

  std::vector<IndexVar> ivars = {
      IndexVar("i"),
      IndexVar("j"),
      IndexVar("k"),
      IndexVar("l"),
      IndexVar("m"),
      IndexVar("n"),
      IndexVar("o"),
      IndexVar("p"),
      IndexVar("q"),
      IndexVar("r"),
      IndexVar("s"),
      IndexVar("t"),
  };

  std::vector<int> dims = {20, 20, 43, 43, 43, 43, 43};
  dims.resize(order);
  // TODO (rohany, owhsu): We need to actually generate the input game state.
  for (auto _ : state) {
    state.PauseTiming();
    Tensor<float> result("C");
    result = genMinMaxExpr(game, ivars, 0);
    result.compile();
//    std::cout << result.getSource() << std::endl;
    state.ResumeTiming();
    result.compute();
  }
}
TACO_BENCH(bench_minimax);
