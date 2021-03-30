#include <fstream>
// We're using c++14, so wer're stuck with experimental filesystem.
#include <experimental/filesystem>
#include <tuple>

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

struct orAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Union(regions[0], regions[1]);
  }
};

// Right shift op and Algebra
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

// LdExp Op (algbra same as right shift)
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

// Power op and algebra (X U Y^(c) if 1 compressed out, X^(c) U Y if 0 compressed out)
struct Power {
  ir::Expr operator()(const std::vector<ir::Expr> &v) {
    if (v.size() == 1)
      return v[0];

    ir::Expr pow = ir::BinOp::make(v[0], v[1], "pow(", ", ", ")");
    for (size_t idx = 2; idx < v.size(); ++idx) {
      pow = ir::BinOp::make(pow, v[idx], "pow(", ", ", ")");
    }
    return pow;
  }
};

struct unionRightCompAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Union(regions[0], Complement(regions[1]));
  }
};

struct rightIncAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return regions[1];
  }
};


struct compAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& regions) {
    return Complement(regions[0]);
  }
};

struct nestedXorAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr> & regions) {
    IterationAlgebra intersect2 = Union(Intersect(regions[2], Union(regions[0], regions[1])), Intersect(regions[0], Union(regions[2], regions[1])));
    IterationAlgebra intersect3 = Intersect(Intersect(regions[0], regions[1]), regions[2]);
    IterationAlgebra unionComplement = Complement(Union(Union(regions[0], regions[1]), regions[2]));
    return Union(Complement(Union(intersect2, unionComplement)), intersect3);
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

Func ldExp("ldexp", Ldexp(), leftIncAlgebra());
Func rightShift("right_shift", RightShift(), leftIncAlgebra());
Func xorOp("logical_xor", GeneralAdd(), xorAlgebra());
Func andOp("logical_and", GeneralAdd(), andAlgebra());
Func orOp("logical_or", GeneralAdd(), orAlgebra());
Func nestedXorOp("fused_xor", GeneralAdd(), nestedXorAlgebra());
Func pow0Comp("power_0compression", Power(), unionRightCompAlgebra());
Func pow1Comp("power_1compression", Power(), rightIncAlgebra());
static void bench_ufunc_fused(benchmark::State& state, const Format& f) {
  int dim = state.range(0);
  auto sparsity = 0.01;
  Tensor<int64_t> matrix = castToType<int64_t>("A", loadRandomTensor("A", {dim, dim}, sparsity, f));
  Tensor<int64_t> matrix1 = castToType<int64_t>("B", loadRandomTensor("B", {dim, dim}, sparsity, f, 1 /* variant */));
  Tensor<int64_t> matrix2 = castToType<int64_t>("C", loadRandomTensor("C", {dim, dim}, sparsity, f, 2 /* variant */));

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", {dim, dim}, f);
    IndexVar i("i"), j("j");
    result(i, j) = nestedXorOp(matrix(i, j), matrix1(i, j), matrix2(i, j));
    result.setAssembleWhileCompute(true);
    result.compile();
    state.ResumeTiming();

    result.compute();
    result = result.removeExplicitZeros(result.getFormat());

  }
}
 TACO_BENCH_ARGS(bench_ufunc_fused, csr, CSR)
   ->ArgsProduct({{5000, 10000, 20000}});

// UfuncInputCache is a cache for the input to ufunc benchmarks. These benchmarks
// operate on a tensor loaded from disk and the same tensor shifted slightly. Since
// these operations are run multiple times, we can save alot in benchmark startup
// time from caching these inputs.
struct UfuncInputCache {
  template<typename U>
  std::pair<taco::Tensor<int64_t>, taco::Tensor<int64_t>> getUfuncInput(std::string path, U format, bool countNNZ = false, bool includeThird = false) {
    // See if the paths match.
    if (this->lastPath == path) {
      // TODO (rohany): Not worrying about whether the format was the same as what was asked for.
      return std::make_pair(this->inputTensor, this->otherTensor);
    }

    // Otherwise, we missed the cache. Load in the target tensor and process it.
    this->lastLoaded = taco::read(path, format);
    // We assign lastPath after lastLoaded so that if taco::read throws an exception
    // then lastPath isn't updated to the new path.
    this->lastPath = path;
    this->inputTensor = castToType<int64_t>("A", this->lastLoaded);
    this->otherTensor = shiftLastMode<int64_t, int64_t>("B", this->inputTensor);
    if (countNNZ) {
      this->nnz = 0;
      for (auto& it : iterate<int64_t>(this->inputTensor)) {
        this->nnz++;
      }
    }
    if (includeThird) {
      this->thirdTensor = shiftLastMode<int64_t, int64_t>("C", this->otherTensor);
    }
    return std::make_pair(this->inputTensor, this->otherTensor);
  }

  taco::Tensor<double> lastLoaded;
  std::string lastPath;

  taco::Tensor<int64_t> inputTensor;
  taco::Tensor<int64_t> otherTensor;
  taco::Tensor<int64_t> thirdTensor;
  int64_t nnz;
};
UfuncInputCache inputCache;

std::string ufuncBenchKey(std::string tensorName, std::string funcName) {
  return tensorName + "-" + funcName + "-taco";
}

static void bench_frostt_ufunc(benchmark::State& state, std::string tnsPath, Func op) {
  auto frosttTensorPath = getTacoTensorPath();
  frosttTensorPath += "FROSTT/";
  frosttTensorPath += tnsPath;

  auto pathSplit = taco::util::split(tnsPath, "/");
  auto filename = pathSplit[pathSplit.size() - 1];
  auto tensorName = taco::util::split(filename, ".")[0];
  state.SetLabel(tensorName);

  // TODO (rohany): What format do we want to do here?
  Tensor<int64_t> frosttTensor, other;
  std::tie(frosttTensor, other) = inputCache.getUfuncInput(frosttTensorPath, Sparse);

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", frosttTensor.getDimensions(), frosttTensor.getFormat());
    result.setAssembleWhileCompute(true);
    switch (frosttTensor.getOrder()) {
      case 3: {
        IndexVar i, j, k;
        result(i, j, k) = op(frosttTensor(i, j, k), other(i, j, k));
        break;
      }
      case 4: {
        IndexVar i, j, k, l;
        result(i, j, k, l) = op(frosttTensor(i, j, k, l), other(i, j, k, l));
        break;
      }
      case 5: {
        IndexVar i, j, k, l, m;
        result(i, j, k, l, m) = op(frosttTensor(i, j, k, l, m), other(i, j, k, l, m));
        break;
      }
      default:
        state.SkipWithError("invalid tensor dimension");
        return;
    }
    result.compile();
    state.ResumeTiming();

    result.compute();

    state.PauseTiming();
//    int nnz = 0;
//    for (auto& it : iterate<int64_t>(result)) {
//      nnz++;
//    }
//    std::cout << "Result NNZ = " << nnz << std::endl;

    if (auto validationPath = getValidationOutputPath(); validationPath != "") {
      auto key = ufuncBenchKey(tensorName, op.getName());
      auto outpath = validationPath + key + ".tns";
      taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
    }
  }
}

#define FOREACH_FROSTT_TENSOR(__func__) \
  __func__(nips, "nips.tns") \
  __func__(uber-pickups, "uber-pickups.tns") \
  __func__(chicago-crime, "chicago-crime.tns") \
  __func__(enron, "enron.tns") \
  __func__(nell-2, "nell-2.tns") \
  __func__(vast, "vast.tns")

  // Other FROSTT tensors that may or may not be too large to load.
  // __func__(delicious, "delicious.tns") \
  // __func__(flickr, "flickr.tns") \
  // __func__(nell-1, "nell-1.tns") \
  // __func__(patents, "patents.tns") \
  // __func__(reddit, "reddit.tns") \
  // __func__(amazon-reviews, "amazon-reviews.tns") \
  // lbnl-network is fine, but python can't load it.
  // __func__(lbnl-network, "lbnl-network.tns") \
 

#define DECLARE_FROSTT_UFUNC_BENCH(name, path) \
  TACO_BENCH_ARGS(bench_frostt_ufunc, name/xor, path, xorOp); \
  TACO_BENCH_ARGS(bench_frostt_ufunc, name/ldExp, path, ldExp); \
  TACO_BENCH_ARGS(bench_frostt_ufunc, name/rightShift, path, rightShift); \
  //TACO_BENCH_ARGS(bench_frostt_ufunc, name/pow0Comp, path, pow0Comp); \
  //TACO_BENCH_ARGS(bench_frostt_ufunc, name/pow1Comp, path, pow1Comp); \

FOREACH_FROSTT_TENSOR(DECLARE_FROSTT_UFUNC_BENCH)

enum FusedUfuncOp {
  XOR_AND = 1,
  XOR_OR = 2,
  XOR_XOR = 3,
};

static void bench_frostt_ufunc_fused(benchmark::State& state, std::string tnsPath, FusedUfuncOp op) {
  auto frosttTensorPath = getTacoTensorPath();
  frosttTensorPath += "FROSTT/";
  frosttTensorPath += tnsPath;

  auto pathSplit = taco::util::split(tnsPath, "/");
  auto filename = pathSplit[pathSplit.size() - 1];
  auto tensorName = taco::util::split(filename, ".")[0];
  state.SetLabel(tensorName);

  Tensor<int64_t> frosttTensor, other;
  std::tie(frosttTensor, other) = inputCache.getUfuncInput(frosttTensorPath, Sparse, false /* countNNZ */, true /* includeThird */);
  Tensor<int64_t> third = inputCache.thirdTensor;

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", frosttTensor.getDimensions(), frosttTensor.getFormat());
    result.setAssembleWhileCompute(true);
    // We have to unfortunately perform this double nesting because for some reason
    // I get a TACO generated code compilation error trying to lift the ufunc operation
    // into lambda.
    switch (frosttTensor.getOrder()) {
      case 3: {
        IndexVar i, j, k;
        switch (op) {
          case XOR_AND: {
            result(i, j, k) = andOp(xorOp(frosttTensor(i, j, k), other(i, j, k)), third(i, j, k));
            break;
          }
          case XOR_OR: {
            result(i, j, k) = orOp(xorOp(frosttTensor(i, j, k), other(i, j, k)), third(i, j, k));
            break;
          }
          case XOR_XOR: {
            result(i, j, k) = nestedXorOp(frosttTensor(i, j, k), other(i, j, k), third(i, j, k));
            break;
          }
          default:
            state.SkipWithError("invalid fused op");
            return;
        }
        break;
      }
      case 4: {
        IndexVar i, j, k, l;
        switch (op) {
          case XOR_AND: {
            result(i, j, k, l) = andOp(xorOp(frosttTensor(i, j, k, l), other(i, j, k, l)), third(i, j, k, l));
            break;
          }
          case XOR_OR: {
            result(i, j, k, l) = orOp(xorOp(frosttTensor(i, j, k, l), other(i, j, k, l)), third(i, j, k, l));
            break;
          }
          case XOR_XOR: {
            result(i, j, k, l) = nestedXorOp(frosttTensor(i, j, k, l), other(i, j, k, l), third(i, j, k, l));
            break;
          }
          default:
            state.SkipWithError("invalid fused op");
            return;
        }
        break;
      }
      case 5: {
        IndexVar i, j, k, l, m;
        switch (op) {
          case XOR_AND: {
            result(i, j, k, l, m) = andOp(xorOp(frosttTensor(i, j, k, l, m), other(i, j, k, l, m)), third(i, j, k, l, m));
            break;
          }
          case XOR_OR: {
            result(i, j, k, l, m) = orOp(xorOp(frosttTensor(i, j, k, l, m), other(i, j, k, l, m)), third(i, j, k, l, m));
            break;
          }
          case XOR_XOR: {
            result(i, j, k, l, m) = nestedXorOp(frosttTensor(i, j, k, l, m), other(i, j, k, l, m), third(i, j, k, l, m));
            break;
          }
          default:
            state.SkipWithError("invalid fused op");
            return;
        }
        break;
      }
      default:
        state.SkipWithError("invalid tensor dimension");
        return;
    }
    result.compile();
    state.ResumeTiming();

    result.compute();
    
  }
}

#define DECLARE_FROSTT_FUSED_UFUNC_BENCH(name, path) \
  TACO_BENCH_ARGS(bench_frostt_ufunc_fused, name/xorAndFused, path, XOR_AND); \
  TACO_BENCH_ARGS(bench_frostt_ufunc_fused, name/xorOrFused, path, XOR_OR); \
//  TACO_BENCH_ARGS(bench_frostt_ufunc_fused, name/xorXorFused, path, XOR_XOR); \

FOREACH_FROSTT_TENSOR(DECLARE_FROSTT_FUSED_UFUNC_BENCH)

struct SuiteSparseTensors {
 SuiteSparseTensors() {
   auto ssTensorPath = getTacoTensorPath();
   ssTensorPath += "suitesparse/";
   if (std::experimental::filesystem::exists(ssTensorPath)) {
     for (auto& entry : std::experimental::filesystem::directory_iterator(ssTensorPath)) {
       std::string f(entry.path());
       // Check that the filename ends with .mtx.
       if (f.compare(f.size() - 4, 4, ".mtx") == 0) {
         this->tensors.push_back(entry.path());
       }
     }
   }
 }

 std::vector<std::string> tensors;
};
SuiteSparseTensors ssTensors;

static void bench_suitesparse_ufunc(benchmark::State& state, Func op) {
  // Counters must be present in every run to get reported to the CSV.
  state.counters["dimx"] = 0;
  state.counters["dimy"] = 0;
  state.counters["nnz"] = 0;

  auto tensorPath = getEnvVar("SUITESPARSE_TENSOR_PATH");
  if (tensorPath == "") {
    state.error_occurred();
    return;
  }

  auto pathSplit = taco::util::split(tensorPath, "/");
  auto filename = pathSplit[pathSplit.size() - 1];
  auto tensorName = taco::util::split(filename, ".")[0];
  state.SetLabel(tensorName);

  taco::Tensor<int64_t> ssTensor, other;
  try {
    std::tie(ssTensor, other) = inputCache.getUfuncInput(tensorPath, CSR, true /* countNNZ */);
  } catch (TacoException& e) {
    // Counters don't show up in the generated CSV if we used SkipWithError, so
    // just add in the label that this run is skipped.	  
    state.SetLabel(tensorName+"/SKIPPED-FAILED-READ");
    return;
  }

  state.counters["dimx"] = ssTensor.getDimension(0);
  state.counters["dimy"] = ssTensor.getDimension(1);
  state.counters["nnz"] = inputCache.nnz;

  for (auto _ : state) {
    state.PauseTiming();
    Tensor<int64_t> result("result", ssTensor.getDimensions(), ssTensor.getFormat());
    result.setAssembleWhileCompute(true);
    IndexVar i, j;
    result(i, j) = op(ssTensor(i, j), other(i, j));
    result.compile();
    state.ResumeTiming();

    result.compute();

    state.PauseTiming();
    if (auto validationPath = getValidationOutputPath(); validationPath != "") {
      auto key = ufuncBenchKey(tensorName, op.getName());
      auto outpath = validationPath + key + ".tns";
      taco::write(outpath, result.removeExplicitZeros(result.getFormat()));
    }
  }
}

TACO_BENCH_ARGS(bench_suitesparse_ufunc, xor, xorOp);
TACO_BENCH_ARGS(bench_suitesparse_ufunc, ldExp, ldExp);
TACO_BENCH_ARGS(bench_suitesparse_ufunc, rightShift, rightShift);
