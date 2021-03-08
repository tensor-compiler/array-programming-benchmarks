#include <fstream>
#include "bench.h"
#include "benchmark/benchmark.h"

#include "taco/tensor.h"
#include "taco/format.h"
#include "taco/index_notation/index_notation.h"
#include "taco/index_notation/tensor_operator.h"

using namespace taco;

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

struct RightShift{
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    if (v.size() == 1)
      return v[0];

    ir::Expr shift = ir::BinOp::make(v[0], v[1], " >> ");
    for (size_t idx = 2; idx < v.size(); ++idx) {
      shift = ir::BinOp::make(shift, v[idx], " >> ");
    }
    return shift;
  }
};

struct leftIncAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Union(regions[0], Intersect(regions[0], regions[1]));
  }
};

struct Ldexp {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    if (v.size() == 1)
      return v[0];

    ir::Expr shift = ir::BinOp::make(v[0], v[1], "", "* pow(2.0, ", ")");
    for (size_t idx = 2; idx < v.size(); ++idx) {
      shift = ir::BinOp::make(shift, v[idx], "", "* pow(2.0, ", ")");
    }
    return shift;
  }
};

template <int I, class...Ts>
decltype(auto) get(Ts&&... ts) {
  return std::get<I>(std::forward_as_tuple(ts...));
}

template <class ...ExtraArgs>
void printTensor(TensorBase tensor,  std::string location, std::string benchName, int dim, ExtraArgs&&... extra_args) {
  auto& sparsity = get<0>(extra_args...);
  auto opType = get<1>(extra_args...);

  std::string sparseStr = std::to_string(sparsity);
  sparseStr = sparseStr.substr(2, sparseStr.size());
  std::string filename = location + "/" + benchName + "_" + opType + "_" + \
                          std::to_string(dim) + "_" + sparseStr + "_" + tensor.getName()  + ".txt";
  std::ofstream outfile(filename, std::ofstream::out);
  outfile << util::toString(tensor);
  outfile.close();
}

// TODO (rohany): We can extract most of the logic out of here and parametrize the test by
//  a particular Func.
template <class ...ExtraArgs>
static void bench_ufunc_sparse(benchmark::State& state, ExtraArgs&&... extra_args) {
  int dim = state.range(0);

  auto& sparsity = get<0>(extra_args...);
  auto opType = get<1>(extra_args...);

  Func xorOp("xor", GeneralAdd(), xorAlgebra());
  Func op = xorOp;
  if (opType == ">>") {
    Func rightShiftOp(">>", RightShift(), leftIncAlgebra());
    op = rightShiftOp;
  } else if (opType == "2^") {
    Func ldexpOp("2^", Ldexp(), leftIncAlgebra());
    op = ldexpOp;
  }

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
  
  // Output tensors to file
  // FIXME (owhsu): Why for dim == 10, does the CSR dense mode repeat indices?
  //                This is causing a problem for the format of csr_matrix(...) in pytest
  //                See <repo>/data/* for examples
  printTensor(A, "./data", __FUNCTION__ , dim, extra_args...);
  printTensor(B, "./data", __FUNCTION__ , dim, extra_args...);

  for (auto _ : state) {
    // Setup.
    state.PauseTiming();
    Tensor<float> result("C", {dim, dim}, CSR);
    IndexVar i, j;
    result(i, j) = op(A(i, j), B(i, j));
    result.compile();
    result.assemble();
    state.ResumeTiming();

    // The actual computation.
    result.compute();
    state.PauseTiming();
    printTensor(result, "./data", __FUNCTION__, dim, extra_args...);
    state.ResumeTiming();
  }
}
static void applyBenchSizes(benchmark::internal::Benchmark* b) {
  // b->ArgsProduct({{250, 500, 750, 1000, 2500, 5000, 7500, 8000}});
  b->ArgsProduct({{10}});
}

TACO_BENCH_ARGS(bench_ufunc_sparse, xor_0.01, 0.01, "xor")->Apply(applyBenchSizes);
TACO_BENCH_ARGS(bench_ufunc_sparse, rightShift_0.01, 0.01, ">>")->Apply(applyBenchSizes);

static void bench_frostt_ufunc(benchmark::State& state, std::string tnsPath) {
  auto path = getTacoTensorPath();
  auto frosttTensorPath = path;
  if (frosttTensorPath[frosttTensorPath.size() - 1] != '/') {
    frosttTensorPath += "/";
  }
  frosttTensorPath += "FROSTT/";
  frosttTensorPath += tnsPath;

  // TODO (rohany): What format do we want to do here?
  auto frosttTensor = read(frosttTensorPath, Sparse);
  Tensor<double> other = shiftLastMode<double>("other", frosttTensor);

  // TODO (rohany): Parametrize by the ufunc as well.
  // TODO (rohany): Certain ufuncs need for operands to be of a certain type.
  //  ldexp for example requires the right hand side to be an integer (for python
  //  at least). Not sure how we'll handle that for this.
  Func ldExp("2^", Ldexp(), leftIncAlgebra());

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<double> result("result", frosttTensor.getDimensions(), frosttTensor.getFormat());
    switch (frosttTensor.getOrder()) {
      case 4: {
        IndexVar i, j, k, l;
        result(i, j, k, l) = ldExp(frosttTensor(i, j, k, l), other(i, j, k, l));
        break;
      }
      case 5: {
        IndexVar i, j, k, l, m;
        result(i, j, k, l, m) = ldExp(frosttTensor(i, j, k, l, m), other(i, j, k, l, m));
        break;
      }
      default:
        state.SkipWithError("invalid tensor dimension");
        return;
    }
    result.compile();
    result.assemble();
    state.ResumeTiming();

    result.compute();
  }
}

// TODO (rohany): We can define another macro to "nest" defining benchmarks
//  for each of the ufuncs that we want to operate on.
#define FOREACH_FROSTT_TENSOR(__func__) \
  __func__(nips, "nips.tns") \
  __func__(uber_pickups, "uber-pickups.tns") \
  __func__(chicaco_crime, "chicago-crime.tns") \
  __func__(lbnl_network, "lbnl-network.tns")

#define DECLARE_FROSTT_UFUNC_BENCH(name, path) \
  TACO_BENCH_ARGS(bench_frostt_ufunc, name, path);

FOREACH_FROSTT_TENSOR(DECLARE_FROSTT_UFUNC_BENCH)