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
#include <cmath>
#include <omp.h>

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

taco_tensor_t* to_taco_tensor(GrB_Matrix* mat) {
  GrB_Type type;
  GrB_Index M, N, posSize, crdSize, valsSize;
  GrB_Index* pos;
  GrB_Index* crd;
  void* vals;
  bool jumbled;
  GxB_Matrix_export_CSR(mat, &type, &M, &N, &pos, &crd, &vals, &posSize, &crdSize, &valsSize, &jumbled, NULL);

  auto csrt = new taco_tensor_t;
  csrt->dimensions = new int32_t[2];
  csrt->dimensions[0] = M;
  csrt->dimensions[1] = N;
  csrt->indices = new uint8_t**[2];
  csrt->indices[1] = new uint8_t*[2];

  csrt->indices[1][0] = (uint8_t*)pos;
  csrt->indices[1][1] = (uint8_t*)crd;
  csrt->vals = (uint8_t*)vals;

  return csrt;
}

taco_tensor_t* to_taco_tensor(GrB_Vector* vec) {
  GrB_Type type;
  GrB_Index N, valsSize;
  void* vals;
  GxB_Vector_export_Full(vec, &type, &N, &vals, &valsSize, NULL);

  auto vect = new taco_tensor_t;
  vect->dimensions = new int32_t[1];
  vect->dimensions[0] = N;
  vect->vals = (uint8_t*)vals;

  return vect;
}

taco_tensor_t indices_to_taco_tensor(GrB_Index* indices, GrB_Index size) {
  GrB_Index* pos = new GrB_Index[2];
  pos[0] = 0;
  pos[1] = size;

  taco_tensor_t ind;
  ind.indices = new uint8_t**[1];
  ind.indices[0] = new uint8_t*[2];
  ind.indices[0][0] = (uint8_t*)pos;
  ind.indices[0][1] = (uint8_t*)indices;

  return ind;
}

taco_tensor_t new_vec_taco_tensor(GrB_Index N) {
  taco_tensor_t vec;
  vec.dimensions = new int32_t[1];
  vec.dimensions[0] = N;
  vec.vals = nullptr;

  return vec;
}

taco_tensor_t new_mat_taco_tensor(GrB_Index M, GrB_Index N) {
  taco_tensor_t mat;
  mat.dimensions = new int32_t[2];
  mat.dimensions[0] = M;
  mat.dimensions[1] = N;
  mat.indices = new uint8_t**[2];
  mat.indices[1] = new uint8_t*[2];
  mat.indices[1][0] = nullptr;
  mat.indices[1][1] = nullptr;
  mat.vals = nullptr;

  return mat;
}

void free_mat_taco_tensor(taco_tensor_t mat) {
  free(mat.indices[1][0]);
  free(mat.indices[1][1]);
  free(mat.vals);
}

Format dv({Dense});
int nthreads = 12;

struct GraphBLASFixture {
  GraphBLASFixture() {
    const auto path = "/data/scratch/s3chou/formats-bench/data/pwtk.mtx";
    //const auto path = "/data/scratch/s3chou/formats-bench/data/webbase_1M.mtx";
    //const auto path = "/data/scratch/s3chou/formats-bench/data/coPapersDBLP/coPapersDBLP.mtx";
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
      //A_trop_taco.insert(c.first.toVector(), c.second);
    }
    GrB_Matrix_build_FP64(A_trop_gb, I.data(), J.data(), V.data(), V.size(), GrB_PLUS_FP64);
    //A_trop_taco.pack();
  
    GrB_Vector_new(&x_trop_gb, GrB_FP64, N);
    GrB_Vector_assign_FP64(x_trop_gb, NULL, NULL, 0, GrB_ALL, N, NULL);
  
    //x_trop_taco = Tensor<double>({T.getDimension(1)}, dv, std::numeric_limits<double>::infinity());
    //for (int i = 0; i < T.getDimension(1); ++i) {
    //  x_trop_taco.insert({i}, 0.0);
    //}
    //x_trop_taco.pack();

    GrB_Index stride = (GrB_Index)std::sqrt(T.getDimension(0));
    for (GrB_Index i = 0; i < T.getDimension(0); i += stride) {
      indices.push_back(i);
    }
    indices_taco = indices_to_taco_tensor(indices.data(), indices.size());
  }

  GrB_Matrix A_trop_gb = nullptr;
  GrB_Vector x_trop_gb = nullptr;
  Tensor<double> A_trop_taco;
  Tensor<double> x_trop_taco;
  taco_tensor_t* A_trop_taco_t = nullptr;
  taco_tensor_t* x_trop_taco_t = nullptr;
  std::vector<GrB_Index> indices;
  taco_tensor_t indices_taco;
};
GraphBLASFixture fixture;

static void bench_mxv_suitesparse(benchmark::State& state) {
  GrB_init(GrB_BLOCKING);
  GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
  GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
  GxB_Global_Option_set(GxB_NTHREADS, nthreads);

  GrB_Descriptor desc;
  //GrB_Descriptor_set (desc, GrB_OUTP, GrB_REPLACE);
  
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

static void bench_mxm_suitesparse(benchmark::State& state) {
  GrB_init(GrB_BLOCKING);
  GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
  GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
  GxB_Global_Option_set(GxB_NTHREADS, nthreads);

  GrB_Descriptor desc;
  GrB_Descriptor_set (desc, GrB_OUTP, GrB_REPLACE);
  
  GrB_Matrix C = NULL;
  for (auto _ : state) {
    state.PauseTiming();

    GrB_Matrix_free(&C);

    state.ResumeTiming();

    GrB_Matrix_new(&C, GrB_FP64, fixture.A_trop_taco.getDimension(0), fixture.A_trop_taco.getDimension(1));
    GrB_mxm(C, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP64, fixture.A_trop_gb, fixture.A_trop_gb, desc);
  }
  GrB_Matrix_free(&C);
}

static void bench_extract_suitesparse(benchmark::State& state) {
  GrB_init(GrB_BLOCKING);
  GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
  GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
  GxB_Global_Option_set(GxB_NTHREADS, nthreads);

  GrB_Descriptor desc;
  GrB_Descriptor_set (desc, GrB_OUTP, GrB_REPLACE);

  GrB_Index* indices = fixture.indices.data();
  GrB_Index size = fixture.indices.size();
  
  GrB_Matrix C = NULL;
  for (auto _ : state) {
    state.PauseTiming();

    GrB_Matrix_free(&C);

    state.ResumeTiming();

    //GrB_Matrix_new(&C, GrB_FP64, fixture.A_trop_taco.getDimension(0), fixture.A_trop_taco.getDimension(1));
    GrB_Matrix_new(&C, GrB_FP64, fixture.indices.size(), fixture.indices.size());
    GrB_Matrix_extract(C, NULL, NULL, fixture.A_trop_gb, indices, size, indices, size, desc);
  }
  //GrB_Index nnz;
  //GrB_Matrix_nvals(&nnz, C);
  //std::cout << "nnz: " << nnz << std::endl;
  GrB_Matrix_free(&C);
}

#define restrict __restrict__

int taco_mxv_trop(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x) {
  GrB_Index y1_dimension = (GrB_Index)(y->dimensions[0]);
  double* restrict y_vals = (double*)(y->vals);
  GrB_Index A1_dimension = (GrB_Index)(A->dimensions[0]);
  GrB_Index* restrict A2_pos = (GrB_Index*)(A->indices[1][0]);
  GrB_Index* restrict A2_crd = (GrB_Index*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  GrB_Index x1_dimension = (GrB_Index)(x->dimensions[0]);
  double* restrict x_vals = (double*)(x->vals);

  //y_vals = (double*)calloc(y1_dimension, sizeof(double));
  y_vals = (double*)malloc(sizeof(double) * y1_dimension);

  //#pragma omp parallel for schedule(static) num_threads(nthreads)
  #pragma omp parallel for schedule(dynamic, 256) num_threads(nthreads)
  for (GrB_Index i = 0; i < x1_dimension; i++) {
    //if (!(x_vals[i] != INFINITY)) {
      double tj_val = INFINITY;
      //double tj_val = 0.0;
      for (GrB_Index jA = A2_pos[i]; jA < A2_pos[(i + 1)]; jA++) {
        GrB_Index j = A2_crd[jA];
        tj_val = fmin(tj_val,A_vals[jA] + x_vals[j]);
      }
      y_vals[i] = tj_val;
    //}
  }

  y->vals = (uint8_t*)y_vals;
  return 0;
}

int cmp(const void *a, const void *b) {
  return *((const int*)a) - *((const int*)b);
}

int taco_mxm_trop(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {
  GrB_Index A1_dimension = (GrB_Index)(A->dimensions[0]);
  GrB_Index* restrict A2_pos = (GrB_Index*)(A->indices[1][0]);
  GrB_Index* restrict A2_crd = (GrB_Index*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  GrB_Index B1_dimension = (GrB_Index)(B->dimensions[0]);
  GrB_Index* restrict B2_pos = (GrB_Index*)(B->indices[1][0]);
  GrB_Index* restrict B2_crd = (GrB_Index*)(B->indices[1][1]);
  double* restrict B_vals = (double*)(B->vals);
  GrB_Index C1_dimension = (GrB_Index)(C->dimensions[0]);
  GrB_Index C2_dimension = (GrB_Index)(C->dimensions[1]);
  GrB_Index* restrict C2_pos = (GrB_Index*)(C->indices[1][0]);
  GrB_Index* restrict C2_crd = (GrB_Index*)(C->indices[1][1]);
  double* restrict C_vals = (double*)(C->vals);

  GrB_Index* restrict A2_nnz = 0;
  A2_nnz = (GrB_Index*)malloc(sizeof(GrB_Index) * B1_dimension);

  GrB_Index* restrict qw_index_list_all = 0;
  qw_index_list_all = (GrB_Index*)malloc(sizeof(GrB_Index) * C2_dimension * omp_get_max_threads());
  bool* restrict qw_already_set_all = (bool*)calloc(C2_dimension * omp_get_max_threads(), sizeof(bool));
  #pragma omp parallel for schedule(dynamic, 256) num_threads(nthreads)
  for (GrB_Index qi = 0; qi < B1_dimension; qi++) {
    GrB_Index qw_index_list_size = 0;
    GrB_Index* qw_index_list = qw_index_list_all + (C2_dimension * omp_get_thread_num());
    bool* qw_already_set = qw_already_set_all + (C2_dimension * omp_get_thread_num());
    for (GrB_Index qkB = B2_pos[qi]; qkB < B2_pos[(qi + 1)]; qkB++) {
      GrB_Index qk = B2_crd[qkB];
      for (GrB_Index qjC = C2_pos[qk]; qjC < C2_pos[(qk + 1)]; qjC++) {
        GrB_Index qj = C2_crd[qjC];
        if (!qw_already_set[qj]) {
          qw_index_list[qw_index_list_size] = qj;
          qw_already_set[qj] = 1;
          qw_index_list_size++;
        }
      }
    }
    GrB_Index tqjA2_nnz_val = 0;
    for (GrB_Index qw_index_locator = 0; qw_index_locator < qw_index_list_size; qw_index_locator++) {
      GrB_Index qj = qw_index_list[qw_index_locator];
      tqjA2_nnz_val += (GrB_Index)1;
      qw_already_set[qj] = 0;
    }
    A2_nnz[qi] = tqjA2_nnz_val;
  }
  free(qw_index_list_all);
  free(qw_already_set_all);

  A2_pos = (GrB_Index*)malloc(sizeof(GrB_Index) * (A1_dimension + 1));
  A2_pos[0] = 0;
  for (GrB_Index i = 0; i < A1_dimension; i++) {
    A2_pos[i + 1] = A2_pos[i] + A2_nnz[i];
  }
  A2_crd = (GrB_Index*)malloc(sizeof(GrB_Index) * A2_pos[A1_dimension]);
  A_vals = (double*)malloc(sizeof(double) * A2_pos[A1_dimension]);

  double* restrict w_all = 0;
  GrB_Index* restrict w_index_list_all = 0;
  w_index_list_all = (GrB_Index*)malloc(sizeof(GrB_Index) * C2_dimension * omp_get_max_threads());
  bool* restrict w_already_set_all = (bool*)calloc(C2_dimension * omp_get_max_threads(), sizeof(bool));
  w_all = (double*)malloc(sizeof(double) * C2_dimension * omp_get_max_threads());
  #pragma omp parallel for schedule(dynamic, 256) num_threads(nthreads)
  for (GrB_Index i = 0; i < B1_dimension; i++) {
    GrB_Index w_index_list_size = 0;
    GrB_Index* w_index_list = w_index_list_all + (C2_dimension * omp_get_thread_num());
    bool* w_already_set = w_already_set_all + (C2_dimension * omp_get_thread_num());
    double* w = w_all + (C2_dimension * omp_get_thread_num());
    for (GrB_Index kB = B2_pos[i]; kB < B2_pos[(i + 1)]; kB++) {
      GrB_Index k = B2_crd[kB];
      for (GrB_Index jC = C2_pos[k]; jC < C2_pos[(k + 1)]; jC++) {
        GrB_Index j = C2_crd[jC];
        if (!w_already_set[j]) {
          w[j] = B_vals[kB] + C_vals[jC];
          w_index_list[w_index_list_size] = j;
          w_already_set[j] = 1;
          w_index_list_size++;
        }
        else {
          w[j] = fmin(w[j], B_vals[kB] * C_vals[jC]);
        }
      }
    }
    //qsort(w_index_list, w_index_list_size, sizeof(GrB_Index), cmp);

    for (GrB_Index w_index_locator = 0; w_index_locator < w_index_list_size; w_index_locator++) {
      GrB_Index j = w_index_list[w_index_locator];
      GrB_Index pA2 = A2_pos[i];
      A2_pos[i] = A2_pos[i] + 1;
      A2_crd[pA2] = j;
      A_vals[pA2] = w[j];
      w_already_set[j] = 0;
    }
  }
  free(w_index_list_all);
  free(w_already_set_all);
  free(w_all);

  for (GrB_Index p = 0; p < A1_dimension; p++) {
    A2_pos[A1_dimension - p] = A2_pos[((A1_dimension - p) - 1)];
  }
  A2_pos[0] = 0;

  free(A2_nnz);

  A->indices[1][0] = (uint8_t*)(A2_pos);
  A->indices[1][1] = (uint8_t*)(A2_crd);
  A->vals = (uint8_t*)A_vals;
  return 0;
}

#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))

int taco_extract_trop(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *I, taco_tensor_t *J) {
  GrB_Index A1_dimension = (GrB_Index)(A->dimensions[0]);
  GrB_Index* restrict A2_pos = (GrB_Index*)(A->indices[1][0]);
  GrB_Index* restrict A2_crd = (GrB_Index*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  GrB_Index B1_dimension = (GrB_Index)(B->dimensions[0]);
  GrB_Index* restrict B2_pos = (GrB_Index*)(B->indices[1][0]);
  GrB_Index* restrict B2_crd = (GrB_Index*)(B->indices[1][1]);
  double* restrict B_vals = (double*)(B->vals);
  GrB_Index* restrict I1_pos = (GrB_Index*)(I->indices[0][0]);
  GrB_Index* restrict I1_crd = (GrB_Index*)(I->indices[0][1]);
  GrB_Index* restrict J1_pos = (GrB_Index*)(J->indices[0][0]);
  GrB_Index* restrict J1_crd = (GrB_Index*)(J->indices[0][1]);

  GrB_Index* restrict A2_nnz = 0;
  A2_nnz = (GrB_Index*)malloc(sizeof(GrB_Index) * A1_dimension);
  //A2_nnz = (GrB_Index*)malloc(sizeof(GrB_Index) * 4);

  //#pragma omp parallel for schedule(static) num_threads(nthreads)
  #pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads)
  for (GrB_Index qi = 0; qi < A1_dimension; qi++) {
  //for (GrB_Index qi = 0; qi < 4; qi++) {
    GrB_Index qiB = I1_crd[qi];
    GrB_Index tqjA2_nnz_val = 0;
    GrB_Index qjB = B2_pos[qiB];
    GrB_Index pB2_end = B2_pos[(qiB + 1)];
    GrB_Index qjJ_filter = J1_pos[0];
    GrB_Index pJ1_end = J1_pos[1];

    while (qjB < pB2_end && qjJ_filter < pJ1_end) {
      GrB_Index qjB0 = B2_crd[qjB];
      GrB_Index qjJ_filter0 = J1_crd[qjJ_filter];
      GrB_Index setMatch = TACO_MIN(qjB0,qjJ_filter0);
      if (qjB0 == setMatch && qjJ_filter0 == setMatch) {
        qjB0 = qjJ_filter;
        qjJ_filter0 = qjJ_filter;
      }
      else {
        qjB += (GrB_Index)(qjB0 == setMatch);
        qjJ_filter += (GrB_Index)(qjJ_filter0 == setMatch);
        continue;
      }
      GrB_Index qj = TACO_MIN(qjB0,qjJ_filter0);
      if (qjB0 == qj && qjJ_filter0 == qj) {
        tqjA2_nnz_val += (GrB_Index)1;
      }
      qjB += (GrB_Index)(qjB0 == qj);
      qjJ_filter += (GrB_Index)(qjJ_filter0 == qj);
    }
    A2_nnz[qi] = tqjA2_nnz_val;
  }

  A2_pos = (GrB_Index*)malloc(sizeof(GrB_Index) * (A1_dimension + 1));
  A2_pos[0] = 0;
  for (GrB_Index i = 0; i < A1_dimension; i++) {
    A2_pos[i + 1] = A2_pos[i] + A2_nnz[i];
  }
  A2_crd = (GrB_Index*)malloc(sizeof(GrB_Index) * A2_pos[A1_dimension]);
  A_vals = (double*)malloc(sizeof(double) * A2_pos[A1_dimension]);

  //#pragma omp parallel for schedule(static) num_threads(nthreads)
  #pragma omp parallel for schedule(dynamic, 16) num_threads(nthreads)
  for (GrB_Index i = 0; i < A1_dimension; i++) {
  //for (GrB_Index i = 0; i < 4; i++) {
    GrB_Index iB = I1_crd[i];

    GrB_Index jB = B2_pos[iB];
    GrB_Index pB2_end0 = B2_pos[(iB + 1)];
    GrB_Index jJ_filter = J1_pos[0];
    GrB_Index pJ1_end0 = J1_pos[1];

    while (jB < pB2_end0 && jJ_filter < pJ1_end0) {
      GrB_Index jB0 = B2_crd[jB];
      GrB_Index jJ_filter0 = J1_crd[jJ_filter];
      GrB_Index setMatch0 = TACO_MIN(jB0,jJ_filter0);
      if (jB0 == setMatch0 && jJ_filter0 == setMatch0) {
        jB0 = jJ_filter;
        jJ_filter0 = jJ_filter;
      }
      else {
        jB += (GrB_Index)(jB0 == setMatch0);
        jJ_filter += (GrB_Index)(jJ_filter0 == setMatch0);
        continue;
      }
      GrB_Index j = TACO_MIN(jB0,jJ_filter0);
      if (jB0 == j && jJ_filter0 == j) {
        GrB_Index pA2 = A2_pos[i];
        A2_pos[i] = A2_pos[i] + 1;
        A2_crd[pA2] = j;
        A_vals[pA2] = B_vals[jB];
      }
      jB += (GrB_Index)(jB0 == j);
      jJ_filter += (GrB_Index)(jJ_filter0 == j);
    }

    //A2_pos[iA + 1] = jA - jA;
  }

  for (GrB_Index p = 0; p < A1_dimension; p++) {
    A2_pos[A1_dimension - p] = A2_pos[((A1_dimension - p) - 1)];
  }
  A2_pos[0] = 0;

  free(A2_nnz);

  A->indices[1][0] = (uint8_t*)(A2_pos);
  A->indices[1][1] = (uint8_t*)(A2_crd);
  A->vals = (uint8_t*)A_vals;
  return 0;
}

static void bench_mxv_taco(benchmark::State& state) {
#if 0
  taco_set_num_threads(nthreads);
  for (auto _ : state) {
    state.PauseTiming();

    IndexVar i, j;
    Tensor<double> y({fixture.A_trop_taco.getDimension(0)}, dv, std::numeric_limits<double>::infinity());
    y(i) = Reduction(MinOp(), j, AddOp(fixture.A_trop_taco(i,j), fixture.x_trop_taco(j)));
    //y(i) = MaskOp(Reduction(MinOp(), j, AddOp(fixture.A_trop_taco(i,j), fixture.x_trop_taco(j))), fixture.x_trop_taco(i));
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
#else
  if (!fixture.A_trop_taco_t) {
    fixture.A_trop_taco_t = to_taco_tensor(&fixture.A_trop_gb);
  }
  if (!fixture.x_trop_taco_t) {
    fixture.x_trop_taco_t = to_taco_tensor(&fixture.x_trop_gb);
  }
  taco_tensor_t y = new_vec_taco_tensor(fixture.A_trop_taco.getDimension(0));
  for (auto _ : state) {
    state.PauseTiming();

    free(y.vals);

    state.ResumeTiming();

    taco_mxv_trop(&y, fixture.A_trop_taco_t, fixture.x_trop_taco_t);
  }
  free(y.vals);
#endif
}

static void bench_mxm_taco(benchmark::State& state) {
  if (!fixture.A_trop_taco_t) {
    fixture.A_trop_taco_t = to_taco_tensor(&fixture.A_trop_gb);
  }
  taco_tensor_t C = new_mat_taco_tensor(fixture.A_trop_taco.getDimension(0), fixture.A_trop_taco.getDimension(1));
  for (auto _ : state) {
    state.PauseTiming();

    free_mat_taco_tensor(C);

    state.ResumeTiming();

    taco_mxm_trop(&C, fixture.A_trop_taco_t, fixture.A_trop_taco_t);
  }
  free_mat_taco_tensor(C);
}

static void bench_extract_taco(benchmark::State& state) {
  if (!fixture.A_trop_taco_t) {
    fixture.A_trop_taco_t = to_taco_tensor(&fixture.A_trop_gb);
  }
  taco_tensor_t B = new_mat_taco_tensor(fixture.indices.size(), fixture.indices.size());
  for (auto _ : state) {
    state.PauseTiming();

    free_mat_taco_tensor(B);

    state.ResumeTiming();

    taco_extract_trop(&B, fixture.A_trop_taco_t, &fixture.indices_taco, &fixture.indices_taco);
  }
  //std::cout << ((GrB_Index*)(B.indices[1][0]))[B.dimensions[0]] << std::endl;
  free_mat_taco_tensor(B);
}

GRAPHBLAS_BENCH(bench_mxv_suitesparse, 1000);
GRAPHBLAS_BENCH(bench_mxm_suitesparse, 25);
GRAPHBLAS_BENCH(bench_extract_suitesparse, 1000);
GRAPHBLAS_BENCH(bench_mxv_taco, 1000);
GRAPHBLAS_BENCH(bench_mxm_taco, 25);
GRAPHBLAS_BENCH(bench_extract_taco, 1000);

#endif
