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

#include <fstream>
#include <vector>
#include <limits>
#include <cmath>
#include <map>
#include <omp.h>

using namespace taco;

taco_tensor_t* to_csr_taco_tensor(GrB_Matrix* mat) {
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

taco_tensor_t* to_bitmap_taco_tensor(GrB_Vector* vec) {
  GrB_Type type;
  GrB_Index N, valsSize, validSize, nvals;
  void* vals;
  int8_t* valid;
  GxB_Vector_export_Bitmap(vec, &type, &N, &valid, &vals, &validSize, &valsSize, &nvals, NULL);

  auto vect = new taco_tensor_t;
  vect->dimensions = new int32_t[1];
  vect->dimensions[0] = N;
  vect->indices = new uint8_t**[1];
  vect->indices[0] = new uint8_t*[1];

  vect->indices[0][0] = (uint8_t*)valid;
  vect->vals = (uint8_t*)vals;

  return vect;
}

taco_tensor_t* to_dense_taco_tensor(GrB_Vector* vec) {
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

#if 0
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
#endif

taco_tensor_t new_bitmap_taco_tensor(GrB_Index N) {
  taco_tensor_t vec;
  vec.dimensions = new int32_t[1];
  vec.dimensions[0] = N;
  vec.indices = new uint8_t**[1];
  vec.indices[0] = new uint8_t*[1];

  vec.indices[0][0] = nullptr;
  vec.vals = nullptr;

  return vec;
}

taco_tensor_t new_csr_taco_tensor(GrB_Index M, GrB_Index N) {
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

void free_bitmap_taco_tensor(taco_tensor_t vec) {
  free(vec.indices[0][0]);
  free(vec.vals);
}

void free_csr_taco_tensor(taco_tensor_t mat) {
  free(mat.indices[1][0]);
  free(mat.indices[1][1]);
  free(mat.vals);
}

bool is_bitmap_vector(GrB_Vector* vec) {
  int sparsity;
  GxB_Vector_Option_get(*vec, GxB_SPARSITY_STATUS, &sparsity);
  return (sparsity == GxB_BITMAP);
}

bool is_dense_vector(GrB_Vector* vec) {
  int sparsity;
  GxB_Vector_Option_get(*vec, GxB_SPARSITY_STATUS, &sparsity);
  return (sparsity == GxB_FULL);
}

bool is_csr_matrix(GrB_Matrix* mat) {
  int sparsity;
  GxB_Format_Value fmt;
  GxB_Matrix_Option_get(*mat, GxB_SPARSITY_STATUS, &sparsity);
  GxB_Matrix_Option_get(*mat, GxB_FORMAT, &fmt);
  return (sparsity == GxB_SPARSE && fmt == GxB_BY_ROW);
}

struct BitmapArrays {
  GrB_Index m;
  int8_t* valid = nullptr;
  void*   vals  = nullptr;
};

BitmapArrays get_bitmap_arrays(const taco_tensor_t vector) {
  BitmapArrays vec;

  vec.m = vector.dimensions[0];
  vec.valid = (int8_t*)vector.indices[0][0];
  vec.vals = vector.vals;

  return vec;
}

double compare_double_bitmap(BitmapArrays a, BitmapArrays b) {
  if (a.m != b.m) {
    return std::numeric_limits<double>::infinity();
  }

  double* avals = (double*)a.vals;
  double* bvals = (double*)b.vals;

  double ret = 0.0;
  for (int i = 0 ; i < a.m; ++i) {
    bool avalid = a.valid[i] && !std::isinf(avals[i]);
    bool bvalid = b.valid[i] && !std::isinf(bvals[i]);
    if (avalid != bvalid) {
      return std::numeric_limits<double>::infinity();
    }
    if (avalid) {
      if (avals[i] != 0.0) {
        const double diff = std::abs(bvals[i] / avals[i] - 1.0);
        if (diff > ret) {
          ret = diff;
        }
      } else if (bvals[i] != 0.0) {
        return std::numeric_limits<double>::infinity();
      }
    }
  }

  return ret;
}

double compare_bool_bitmap(BitmapArrays a, BitmapArrays b) {
  if (a.m != b.m) {
    return 1.0;
  }

  bool* avals = (bool*)a.vals;
  bool* bvals = (bool*)b.vals;

  for (int i = 0 ; i < a.m; ++i) {
    bool avalid = a.valid[i] && avals[i];
    bool bvalid = b.valid[i] && bvals[i];
    if (avalid != bvalid) {
      return 1.0;
    }
  }

  return 0.0;
}

struct CSRArrays {
  GrB_Index  m, n;
  GrB_Index* pos  = nullptr;
  GrB_Index* crd  = nullptr;
  void*      vals = nullptr;
};

CSRArrays get_csr_arrays(const taco_tensor_t matrix) {
  CSRArrays csr;

  csr.m = matrix.dimensions[0];
  csr.n = matrix.dimensions[1];
  csr.pos = (GrB_Index*)matrix.indices[1][0];
  csr.crd = (GrB_Index*)matrix.indices[1][1];
  csr.vals = matrix.vals;

  return csr;
}

double compare_double_csr(CSRArrays a, CSRArrays b) {
  //std::cout << a.m << " " << b.m << " " << a.n << " " << b.n << std::endl;
  if (a.m != b.m || a.n != b.n) {
    return std::numeric_limits<double>::infinity();
  }

  double* avals = (double*)a.vals;
  double* bvals = (double*)b.vals;

  double ret = 0.0;
  for (int i = 0; i < a.m; ++i) {
    int pA = a.pos[i];
    int pB = b.pos[i];
    while (pA < a.pos[i + 1] && pB < b.pos[i + 1]) {
      while (pA < a.pos[i + 1] && std::isinf(avals[pA])) pA++;
      while (pB < b.pos[i + 1] && std::isinf(bvals[pB])) pB++;
      if (pA < a.pos[i + 1] && pB < b.pos[i + 1]) {
        //std::cout << a.crd[pA] << " " << b.crd[pB] << " " << avals[pA] << " " << bvals[pB] << std::endl;
        if (a.crd[pA] != b.crd[pB]) {
          return std::numeric_limits<double>::infinity();
        } else if (avals[pA] != 0.0) {
          const double diff = std::abs(bvals[pB] / avals[pA] - 1.0);
          if (diff > ret) {
            ret = diff;
            //std::cout << i << " " << a.crd[pA] << " " << b.crd[pB] << " " << avals[pA] << " " << bvals[pB] << std::endl;
          }
        } else if (bvals[pB] != 0.0) {
          return std::numeric_limits<double>::infinity();
        }
        pA++;
        pB++;
      }
    }
    while (pA < a.pos[i + 1] && std::isinf(avals[pA])) pA++;
    while (pB < b.pos[i + 1] && std::isinf(bvals[pB])) pB++;
    if (pA != a.pos[i + 1] || pB != b.pos[i + 1]) {
      return std::numeric_limits<double>::infinity();
    }
  }

  return ret;
}

double compare_bool_csr(CSRArrays a, CSRArrays b) {
  //std::cout << a.m << " " << b.m << " " << a.n << " " << b.n << std::endl;
  if (a.m != b.m || a.n != b.n) {
    return 1.0;
  }

  bool* avals = (bool*)a.vals;
  bool* bvals = (bool*)b.vals;

  for (int i = 0; i < a.m; ++i) {
    int pA = a.pos[i];
    int pB = b.pos[i];
    while (pA < a.pos[i + 1] && pB < b.pos[i + 1]) {
      while (pA < a.pos[i + 1] && !avals[pA]) pA++;
      while (pB < b.pos[i + 1] && !bvals[pB]) pB++;
      if (pA < a.pos[i + 1] && pB < b.pos[i + 1]) {
        //std::cout << a.crd[pA] << " " << b.crd[pB] << " " << avals[pA] << " " << bvals[pB] << std::endl;
        if (a.crd[pA] != b.crd[pB]) {
          return 1.0;
        }
        pA++;
        pB++;
      }
    }
    while (pA < a.pos[i + 1] && !avals[pA]) pA++;
    while (pB < b.pos[i + 1] && !bvals[pB]) pB++;
    if (pA != a.pos[i + 1] || pB != b.pos[i + 1]) {
      return 1.0;
    }
  }

  return 0.0;
}
const int nthreads = 12;

struct GraphBLASFixture {
  GraphBLASFixture() {
    //const auto path = "/data/scratch/s3chou/formats-bench/data/pwtk.mtx";
    const auto path = "/data/scratch/s3chou/formats-bench/data/webbase_1M.mtx";
    //const auto path = "/data/scratch/s3chou/formats-bench/data/coPapersDBLP/coPapersDBLP.mtx";
    //const auto path = "/data/scratch/changwan/florida_all/soc-LiveJournal1/soc-LiveJournal1.mtx";
    //const auto path = "/data/scratch/changwan/florida_all/com-LiveJournal/com-LiveJournal.mtx";
    //const auto path = "/data/scratch/changwan/florida_all/indochina-2004/indochina-2004.mtx";

    // TODO: Only run for square matrices

    //double bsw[GxB_NBITMAP_SWITCH] = {0};
    //GrB_init(GrB_BLOCKING);
    GrB_init(GrB_NONBLOCKING);
    //GxB_Global_Option_set(GxB_HYPER_SWITCH, 1.0);
    //GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
    GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
    GxB_Global_Option_set(GxB_NTHREADS, nthreads);
    //GxB_Global_Option_set(GxB_BITMAP_SWITCH, bsw);
    //GxB_Global_Option_set(GxB_BURBLE, 1);

    read_matrix(path);
  
    if (is_bool) {
      GrB_Vector_new(&x_gb, GrB_BOOL, N);
    } else {
      GrB_Vector_new(&x_gb, GrB_FP64, N);
    }
    GxB_Vector_Option_set(x_gb, GxB_SPARSITY_CONTROL, GxB_BITMAP);
    //GrB_Vector_assign_FP64(x_gb, NULL, NULL, 1.0, GrB_ALL, N, NULL);
    for (GrB_Index i = 0; i < M; i += 4) {
      if (is_bool) {
        GrB_Vector_setElement_BOOL(x_gb, 1, i);
      } else {
        GrB_Vector_setElement_FP64(x_gb, (double)i, i);
      }
    }
    GrB_Vector_wait(&x_gb);
    taco_uassert(is_bitmap_vector(&x_gb)) << "x is not bitmap";
  
    GrB_Vector_new(&m_gb, GrB_BOOL, M);
    GxB_Vector_Option_set(m_gb, GxB_SPARSITY_CONTROL, GxB_FULL);
    GrB_Vector_assign_BOOL(m_gb, NULL, NULL, true, GrB_ALL, N, NULL);
    for (GrB_Index i = 0; i < M; i += 4) {
      GrB_Vector_setElement_BOOL(m_gb, false, i);
    }
    GrB_Vector_wait(&m_gb);
    taco_uassert(is_dense_vector(&m_gb)) << "m is not dense";
  
    //GrB_Index stride = (GrB_Index)std::sqrt(M);
    //for (GrB_Index i = 0; i < M; i += stride) {
    //  indices.push_back(i);
    //}
    //indices_taco = indices_to_taco_tensor(indices.data(), indices.size());
  }

  void read_matrix(const std::string& matrix_path) {
    std::fstream stream;
    stream.open(matrix_path, std::fstream::in);
    if (!stream) {
      stream.close();
      return;
    }
  
    std::string line;
    std::getline(stream, line);
  
    // Read Header
    std::stringstream lineStream(line);
    std::string head, type, formats, field, symmetry;
    lineStream >> head >> type >> formats >> field >> symmetry;
    assert(head=="%%MatrixMarket");
    // type = [matrix tensor]
    // formats = [coordinate array]
    assert((type == "matrix") || (type == "tensor"));
  
    // field = [real integer complex pattern]
    bool isreal = false;
    bool isint = false;
    if (field == "complex") {
      stream.close();
      return;
    } else if (field == "real") {
      isreal = true;
    } else if (field == "integer") {
      isint = true;
    }
  
    // symmetry = [general symmetric skew-symmetric Hermitian]
    if ((symmetry != "general") && (symmetry != "symmetric") && 
        (symmetry != "skew-symmetric")) {
      stream.close();
      return;
    }
  
    const bool symm = ((symmetry == "symmetric") || 
                       (symmetry == "skew-symmetric"));
    const bool skew = (symmetry == "skew-symmetric");
  
    std::getline(stream, line);
  
    // Skip comments at the top of the file
    std::string token;
    do {
      std::stringstream lineStream(line);
      lineStream >> token;
      if (token[0] != '%') {
        break;
      }
    } while (std::getline(stream, line));
  
    // The first non-comment line is the header with dimensions
    std::vector<GrB_Index> dimensions;
    char* linePtr = (char*)line.data();
    while (auto dimension = std::strtoull(linePtr, &linePtr, 10)) {
      dimensions.push_back(dimension);
    }
  
    assert(dimensions.size() == 3);
    nnz = dimensions[2];
  
    GrB_Index* rows = (GrB_Index*)malloc(sizeof(GrB_Index) * nnz * (1 + symm));
    GrB_Index* cols = (GrB_Index*)malloc(sizeof(GrB_Index) * nnz * (1 + symm));
    double* fvals = !is_bool ? (double*)malloc(sizeof(double) * nnz * (1 + symm)) : nullptr;
    bool* bvals = is_bool ? (bool*)malloc(sizeof(bool) * nnz * (1 + symm)) : nullptr;
  
    for (nnz = 0; std::getline(stream, line); nnz++) {
      //if (nnz % 10000000 == 0) std::cout << nnz << std::endl;
      linePtr = (char*)line.data();
  
      const GrB_Index i = strtoull(linePtr, &linePtr, 10) - 1;
      const GrB_Index j = strtoull(linePtr, &linePtr, 10) - 1;

      double fval = 1.0;
      bool bval = true;;
      if (isreal) {
        fval = strtod(linePtr, &linePtr);
        bval = (fval != 0.0);
      } else if (isint) {
        fval = strtoll(linePtr, &linePtr, 10);
        bval = (fval != 0.0);
      }
  
      rows[nnz] = i;
      cols[nnz] = j;
      if (fvals) {
        fvals[nnz] = fval;
      }
      if (bvals) {
        bvals[nnz] = bval;
      }
  
      if (symm && i != j) {
        nnz++;
  
        if (skew) {
          fval = -1.0 * fval;
        }
  
        rows[nnz] = j;
        cols[nnz] = i;
        if (fvals) {
          fvals[nnz] = fval;
        }
        if (bvals) {
          bvals[nnz] = bval;
        }
      }
    }
  
    stream.close();
  
    GrB_Matrix_new(&A_gb, is_bool ? GrB_BOOL : GrB_FP64, dimensions[0], dimensions[1]);
    GxB_Matrix_Option_set(A_gb, GxB_SPARSITY_CONTROL, GxB_SPARSE);
    if (is_bool) {
      GrB_Matrix_build_BOOL(A_gb, rows, cols, bvals, nnz, GrB_LOR);
    } else {
      GrB_Matrix_build_FP64(A_gb, rows, cols, fvals, nnz, GrB_PLUS_FP64);
    }
    GrB_Matrix_wait(&A_gb);
    taco_uassert(is_csr_matrix(&A_gb)) << "A is not CSR";

    free(rows);
    free(cols);
    free(fvals);
    free(bvals);

    M = dimensions[0];
    N = dimensions[1];
  }

  bool is_bool = true;
  bool validate = false;
  GrB_Index M, N, nnz;
  GrB_Matrix A_gb = nullptr;
  GrB_Matrix C_gb = nullptr;
  GrB_Vector x_gb = nullptr;
  GrB_Vector m_gb = nullptr;
  GrB_Vector y_gb = nullptr;
  taco_tensor_t* A_taco_t = nullptr;
  taco_tensor_t* x_taco_t = nullptr;
  taco_tensor_t* m_taco_t = nullptr;
  //std::vector<GrB_Index> indices;
  //taco_tensor_t indices_taco;
};
GraphBLASFixture fixture;

static void bench_mxv_suitesparse(benchmark::State& state) {
  GrB_Descriptor desc;
  GrB_Descriptor_new(&desc);
  GrB_Descriptor_set(desc, GrB_MASK, GrB_COMP);
  GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
  
  for (auto _ : state) {
    state.PauseTiming();

    GrB_Vector_free(&fixture.y_gb);

    GrB_Vector_new(&fixture.y_gb, fixture.is_bool ? GrB_BOOL : GrB_FP64, fixture.M);
    GxB_Vector_Option_set(fixture.y_gb, GxB_SPARSITY_CONTROL, GxB_BITMAP);

    state.ResumeTiming();

    if (fixture.is_bool) {
      GrB_mxv(fixture.y_gb, fixture.m_gb, NULL, GrB_LOR_LAND_SEMIRING_BOOL, fixture.A_gb, fixture.x_gb, desc);
    } else {
      GrB_mxv(fixture.y_gb, fixture.m_gb, NULL, GrB_MIN_PLUS_SEMIRING_FP64, fixture.A_gb, fixture.x_gb, desc);
    }
  }
  taco_uassert(is_bitmap_vector(&fixture.y_gb)) << "y is not bitmap";
  if (!fixture.validate) {
    GrB_Vector_free(&fixture.y_gb);
  }
}

static void bench_mxm_suitesparse(benchmark::State& state) {
  GrB_Descriptor desc;
  GrB_Descriptor_new(&desc);
  GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);
  GrB_Descriptor_set(desc, GxB_AxB_METHOD, GxB_AxB_GUSTAVSON);
  
  for (auto _ : state) {
    state.PauseTiming();

    GrB_Matrix_free(&fixture.C_gb);

    GrB_Matrix_new(&fixture.C_gb, fixture.is_bool ? GrB_BOOL : GrB_FP64, fixture.M, fixture.N);
    GxB_Matrix_Option_set(fixture.C_gb, GxB_SPARSITY_CONTROL, GxB_SPARSE);
    
    state.ResumeTiming();

    if (fixture.is_bool) {
      GrB_mxm(fixture.C_gb, NULL, NULL, GrB_LOR_LAND_SEMIRING_BOOL, fixture.A_gb, fixture.A_gb, desc);
    } else {
      GrB_mxm(fixture.C_gb, NULL, NULL, GrB_MIN_PLUS_SEMIRING_FP64, fixture.A_gb, fixture.A_gb, desc);
    }
    //GrB_Matrix_wait(&fixture.C_gb);
  }
  taco_uassert(is_csr_matrix(&fixture.C_gb)) << "C is not CSR";
  if (!fixture.validate) {
    GrB_Matrix_free(&fixture.C_gb);
  }
}

#if 0
static void bench_extract_suitesparse(benchmark::State& state) {
  //GrB_init(GrB_BLOCKING);
  //GxB_Global_Option_set(GxB_HYPER_SWITCH, GxB_NEVER_HYPER);
  //GxB_Global_Option_set(GxB_FORMAT, GxB_BY_ROW);
  //GxB_Global_Option_set(GxB_NTHREADS, nthreads);
  GrB_Descriptor desc;
  GrB_Descriptor_new(&desc);
  GrB_Descriptor_set(desc, GrB_OUTP, GrB_REPLACE);

  GrB_Index* indices = fixture.indices.data();
  GrB_Index size = fixture.indices.size();
  
  GrB_Matrix C = NULL;
  for (auto _ : state) {
    state.PauseTiming();

    GrB_Matrix_free(&C);

    state.ResumeTiming();

    //GrB_Matrix_new(&C, GrB_FP64, fixture.A_taco.getDimension(0), fixture.A_taco.getDimension(1));
    GrB_Matrix_new(&C, GrB_FP64, fixture.indices.size(), fixture.indices.size());
    GrB_Matrix_extract(C, NULL, NULL, fixture.A_gb, indices, size, indices, size, desc);
  }
  //GrB_Index nnz;
  //GrB_Matrix_nvals(&nnz, C);
  //std::cout << "nnz: " << nnz << std::endl;
  GrB_Matrix_free(&C);
}
#endif

#define restrict __restrict__

int taco_mxv_trop(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x, taco_tensor_t *m) {
#if 0
  GrB_Index y1_dimension = (GrB_Index)(y->dimensions[0]);
  double* restrict y_vals = (double*)(y->vals);
  GrB_Index A1_dimension = (GrB_Index)(A->dimensions[0]);
  GrB_Index* restrict A2_pos = (GrB_Index*)(A->indices[1][0]);
  GrB_Index* restrict A2_crd = (GrB_Index*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  GrB_Index x1_dimension = (GrB_Index)(x->dimensions[0]);
  double* restrict x_vals = (double*)(x->vals);
  GrB_Index m1_dimension = (GrB_Index)(m->dimensions[0]);
  bool* restrict m_vals = (bool*)(m->vals);

  //y_vals = (double*)calloc(y1_dimension, sizeof(double));
  y_vals = (double*)malloc(sizeof(double) * y1_dimension);

  //#pragma omp parallel for schedule(static) num_threads(nthreads)
  #pragma omp parallel for schedule(dynamic, 256) num_threads(nthreads)
  for (GrB_Index i = 0; i < x1_dimension; i++) {
    if (!(m_vals[i] != 0)) {
      double tj_val = INFINITY;
      //double tj_val = 0.0;
      for (GrB_Index jA = A2_pos[i]; jA < A2_pos[(i + 1)]; jA++) {
        GrB_Index j = A2_crd[jA];
        tj_val = fmin(tj_val,A_vals[jA] + x_vals[j]);
      }
      y_vals[i] = tj_val;
    }
    else {
      y_vals[i] = INFINITY;
    }
  }

  y->vals = (uint8_t*)y_vals;
  return 0;
#else
  GrB_Index y1_dimension = (GrB_Index)(y->dimensions[0]);
  int8_t* restrict y1_valid = (int8_t*)(y->indices[0][0]);
  double* restrict y_vals = (double*)(y->vals);
  GrB_Index A1_dimension = (GrB_Index)(A->dimensions[0]);
  GrB_Index* restrict A2_pos = (GrB_Index*)(A->indices[1][0]);
  GrB_Index* restrict A2_crd = (GrB_Index*)(A->indices[1][1]);
  double* restrict A_vals = (double*)(A->vals);
  GrB_Index x1_dimension = (GrB_Index)(x->dimensions[0]);
  int8_t* restrict x1_valid = (int8_t*)(x->indices[0][0]);
  double* restrict x_vals = (double*)(x->vals);
  GrB_Index m1_dimension = (GrB_Index)(m->dimensions[0]);
  bool* restrict m_vals = (bool*)(m->vals);

  y1_valid = (int8_t*)calloc(1, sizeof(int8_t) * y1_dimension);
  int32_t y_capacity = y1_dimension;
  y_vals = (double*)malloc(sizeof(double) * y_capacity);

  #pragma omp parallel for schedule(dynamic, 256) num_threads(nthreads)
  for (GrB_Index i = 0; i < m1_dimension; i++) {
    if (!(m_vals[i] != 0)) {
      //double tj_val = 0.0;
      double tj_val = INFINITY;
      for (GrB_Index jA = A2_pos[i]; jA < A2_pos[(i + 1)]; jA++) {
        GrB_Index j = A2_crd[jA];
        if (x1_valid[j] == 1) {
          tj_val = fmin(tj_val,A_vals[jA] + x_vals[j]);
        }
      }
      y_vals[i] = tj_val;
      y1_valid[i] = 1;
    }
  }

  y->indices[0][0] = (uint8_t*)(y1_valid);
  y->vals = (uint8_t*)y_vals;
  return 0;
#endif
}

int taco_mxv_bool(taco_tensor_t *y, taco_tensor_t *A, taco_tensor_t *x, taco_tensor_t *m) {
  GrB_Index y1_dimension = (GrB_Index)(y->dimensions[0]);
  int8_t* restrict y1_valid = (int8_t*)(y->indices[0][0]);
  bool* restrict y_vals = (bool*)(y->vals);
  GrB_Index A1_dimension = (GrB_Index)(A->dimensions[0]);
  GrB_Index* restrict A2_pos = (GrB_Index*)(A->indices[1][0]);
  GrB_Index* restrict A2_crd = (GrB_Index*)(A->indices[1][1]);
  bool* restrict A_vals = (bool*)(A->vals);
  GrB_Index x1_dimension = (GrB_Index)(x->dimensions[0]);
  int8_t* restrict x1_valid = (int8_t*)(x->indices[0][0]);
  bool* restrict x_vals = (bool*)(x->vals);
  GrB_Index m1_dimension = (GrB_Index)(m->dimensions[0]);
  bool* restrict m_vals = (bool*)(m->vals);

  y1_valid = (int8_t*)calloc(1, sizeof(int8_t) * y1_dimension);
  int32_t y_capacity = y1_dimension;
  y_vals = (bool*)malloc(sizeof(bool) * y_capacity);

  #pragma omp parallel for schedule(dynamic, 256) num_threads(nthreads)
  for (GrB_Index i = 0; i < m1_dimension; i++) {
    if (!(m_vals[i] != 0)) {
      bool tj_val = 0;
      for (GrB_Index jA = A2_pos[i]; jA < A2_pos[(i + 1)]; jA++) {
        GrB_Index j = A2_crd[jA];
        if (x1_valid[j] == 1) {
          tj_val = tj_val || A_vals[jA];
          if (tj_val == 1) {
            break;
          }
        }
      }
      y_vals[i] = tj_val;
      y1_valid[i] = 1;
    }
  }

  y->indices[0][0] = (uint8_t*)(y1_valid);
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

#if 0
  GrB_Index* restrict A2_nnz = 0;
  A2_nnz = (GrB_Index*)malloc(sizeof(GrB_Index) * B1_dimension);

  GrB_Index* restrict qw_index_list_all = 0;
  qw_index_list_all = (GrB_Index*)malloc(sizeof(GrB_Index) * C2_dimension * omp_get_max_threads());
  bool* restrict qw_already_set_all = (bool*)calloc(C2_dimension * omp_get_max_threads(), sizeof(bool));
  #pragma omp parallel for schedule(dynamic, 128) num_threads(nthreads)
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
  std::cout << A2_pos[A1_dimension] << std::endl;
  A2_crd = (GrB_Index*)malloc(sizeof(GrB_Index) * A2_pos[A1_dimension]);
  A_vals = (double*)malloc(sizeof(double) * A2_pos[A1_dimension]);

  double* restrict w_all = 0;
  GrB_Index* restrict w_index_list_all = 0;
  w_index_list_all = (GrB_Index*)malloc(sizeof(GrB_Index) * C2_dimension * omp_get_max_threads());
  bool* restrict w_already_set_all = (bool*)calloc(C2_dimension * omp_get_max_threads(), sizeof(bool));
  w_all = (double*)malloc(sizeof(double) * C2_dimension * omp_get_max_threads());
  #pragma omp parallel for schedule(dynamic, 128) num_threads(nthreads)
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
          w[j] = fmin(w[j], B_vals[kB] + C_vals[jC]);
          //w[j] = w[j] + B_vals[kB] * C_vals[jC]);
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
#else
  GrB_Index* restrict A2_nnz = 0;
  A2_nnz = (GrB_Index*)malloc(sizeof(GrB_Index) * B1_dimension);

  GrB_Index* restrict qw_index_list_all = 0;
  qw_index_list_all = (GrB_Index*)malloc(sizeof(GrB_Index) * (C2_dimension * omp_get_max_threads()));
  bool* restrict qw_already_set_all = calloc((C2_dimension * omp_get_max_threads()), sizeof(bool));

  //#pragma omp parallel for schedule(runtime)
  #pragma omp parallel for schedule(dynamic, 128) num_threads(nthreads)
  for (GrB_Index qi = 0; qi < B1_dimension; qi++) {
    GrB_Index qw_index_list_all_size = 0;
    GrB_Index* restrict qw_index_list = qw_index_list_all + C2_dimension * omp_get_thread_num();
    //GrB_Index* restrict qw_index_list = qw_index_list_all + qw_index_list_all_size * omp_get_thread_num();
    bool* restrict qw_already_set = qw_already_set_all + C2_dimension * omp_get_thread_num();
    for (GrB_Index qkB = B2_pos[qi]; qkB < B2_pos[(qi + 1)]; qkB++) {
      GrB_Index qk = B2_crd[qkB];
      for (GrB_Index qjC = C2_pos[qk]; qjC < C2_pos[(qk + 1)]; qjC++) {
        GrB_Index qj = C2_crd[qjC];
        if (!qw_already_set[qj]) {
          qw_index_list[qw_index_list_all_size] = qj;
          qw_already_set[qj] = 1;
          qw_index_list_all_size++;
        }
      }
    }
    GrB_Index tqjA2_nnz_val = 0;
    for (GrB_Index qw_index_locator = 0; qw_index_locator < qw_index_list_all_size; qw_index_locator++) {
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
  //std::cout << A2_pos[A1_dimension] << std::endl;
  A2_crd = (GrB_Index*)malloc(sizeof(GrB_Index) * A2_pos[A1_dimension]);
  A_vals = (double*)malloc(sizeof(double) * A2_pos[A1_dimension]);

  double* restrict w_all = 0;
  GrB_Index* restrict w_index_list_all = 0;
  w_index_list_all = (GrB_Index*)malloc(sizeof(GrB_Index) * (C2_dimension * omp_get_max_threads()));
  bool* restrict w_already_set_all = calloc((C2_dimension * omp_get_max_threads()), sizeof(bool));
  w_all = (double*)malloc(sizeof(double) * (C2_dimension * omp_get_max_threads()));

  //#pragma omp parallel for schedule(runtime)
  #pragma omp parallel for schedule(dynamic, 128) num_threads(nthreads)
  for (GrB_Index i = 0; i < B1_dimension; i++) {
    GrB_Index w_index_list_all_size = 0;
    double* restrict w = w_all + C2_dimension * omp_get_thread_num();
    GrB_Index* restrict w_index_list = w_index_list_all + C2_dimension * omp_get_thread_num();
    //GrB_Index* restrict w_index_list = w_index_list_all + w_index_list_all_size * omp_get_thread_num();
    bool* restrict w_already_set = w_already_set_all + C2_dimension * omp_get_thread_num();
    for (GrB_Index kB = B2_pos[i]; kB < B2_pos[(i + 1)]; kB++) {
      GrB_Index k = B2_crd[kB];
      for (GrB_Index jC = C2_pos[k]; jC < C2_pos[(k + 1)]; jC++) {
        GrB_Index j = C2_crd[jC];
        if (!w_already_set[j]) {
          w[j] = B_vals[kB] + C_vals[jC];
          //w[j] = B_vals[kB] * C_vals[jC];
          w_index_list[w_index_list_all_size] = j;
          w_already_set[j] = 1;
          w_index_list_all_size++;
        }
        else {
          w[j] = fmin(w[j], B_vals[kB] + C_vals[jC]);
          //w[j] = w[j] + B_vals[kB] * C_vals[jC];
        }
      }
    }
    //qsort(w_index_list, w_index_list_all_size, sizeof(GrB_Index), cmp);

    for (GrB_Index w_index_locator = 0; w_index_locator < w_index_list_all_size; w_index_locator++) {
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
#endif

  A->indices[1][0] = (uint8_t*)(A2_pos);
  A->indices[1][1] = (uint8_t*)(A2_crd);
  A->vals = (uint8_t*)A_vals;
  return 0;
}

int taco_mxm_bool(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *C) {
  GrB_Index A1_dimension = (GrB_Index)(A->dimensions[0]);
  GrB_Index* restrict A2_pos = (GrB_Index*)(A->indices[1][0]);
  GrB_Index* restrict A2_crd = (GrB_Index*)(A->indices[1][1]);
  bool* restrict A_vals = (bool*)(A->vals);
  GrB_Index B1_dimension = (GrB_Index)(B->dimensions[0]);
  GrB_Index* restrict B2_pos = (GrB_Index*)(B->indices[1][0]);
  GrB_Index* restrict B2_crd = (GrB_Index*)(B->indices[1][1]);
  bool* restrict B_vals = (bool*)(B->vals);
  GrB_Index C1_dimension = (GrB_Index)(C->dimensions[0]);
  GrB_Index C2_dimension = (GrB_Index)(C->dimensions[1]);
  GrB_Index* restrict C2_pos = (GrB_Index*)(C->indices[1][0]);
  GrB_Index* restrict C2_crd = (GrB_Index*)(C->indices[1][1]);
  bool* restrict C_vals = (bool*)(C->vals);

  GrB_Index* restrict A2_nnz = 0;
  A2_nnz = (GrB_Index*)malloc(sizeof(GrB_Index) * B1_dimension);

  GrB_Index* restrict qw_index_list_all = 0;
  qw_index_list_all = (GrB_Index*)malloc(sizeof(GrB_Index) * (C2_dimension * omp_get_max_threads()));
  bool* restrict qw_already_set_all = calloc((C2_dimension * omp_get_max_threads()), sizeof(bool));

  //#pragma omp parallel for schedule(runtime)
  #pragma omp parallel for schedule(dynamic, 128) num_threads(nthreads)
  for (GrB_Index qi = 0; qi < B1_dimension; qi++) {
    GrB_Index qw_index_list_all_size = 0;
    GrB_Index* restrict qw_index_list = qw_index_list_all + C2_dimension * omp_get_thread_num();
    //GrB_Index* restrict qw_index_list = qw_index_list_all + qw_index_list_all_size * omp_get_thread_num();
    bool* restrict qw_already_set = qw_already_set_all + C2_dimension * omp_get_thread_num();
    for (GrB_Index qkB = B2_pos[qi]; qkB < B2_pos[(qi + 1)]; qkB++) {
      GrB_Index qk = B2_crd[qkB];
      for (GrB_Index qjC = C2_pos[qk]; qjC < C2_pos[(qk + 1)]; qjC++) {
        GrB_Index qj = C2_crd[qjC];
        if (!qw_already_set[qj]) {
          qw_index_list[qw_index_list_all_size] = qj;
          qw_already_set[qj] = 1;
          qw_index_list_all_size++;
        }
      }
    }
    GrB_Index tqjA2_nnz_val = 0;
    for (GrB_Index qw_index_locator = 0; qw_index_locator < qw_index_list_all_size; qw_index_locator++) {
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
  //std::cout << A2_pos[A1_dimension] << std::endl;
  A2_crd = (GrB_Index*)malloc(sizeof(GrB_Index) * A2_pos[A1_dimension]);
  A_vals = (bool*)malloc(sizeof(bool) * A2_pos[A1_dimension]);

  bool* restrict w_all = 0;
  GrB_Index* restrict w_index_list_all = 0;
  w_index_list_all = (GrB_Index*)malloc(sizeof(GrB_Index) * (C2_dimension * omp_get_max_threads()));
  bool* restrict w_already_set_all = calloc((C2_dimension * omp_get_max_threads()), sizeof(bool));
  w_all = (bool*)malloc(sizeof(bool) * (C2_dimension * omp_get_max_threads()));

  //#pragma omp parallel for schedule(runtime)
  #pragma omp parallel for schedule(dynamic, 128) num_threads(nthreads)
  for (GrB_Index i = 0; i < B1_dimension; i++) {
    GrB_Index w_index_list_all_size = 0;
    bool* restrict w = w_all + C2_dimension * omp_get_thread_num();
    GrB_Index* restrict w_index_list = w_index_list_all + C2_dimension * omp_get_thread_num();
    //GrB_Index* restrict w_index_list = w_index_list_all + w_index_list_all_size * omp_get_thread_num();
    bool* restrict w_already_set = w_already_set_all + C2_dimension * omp_get_thread_num();
    for (GrB_Index kB = B2_pos[i]; kB < B2_pos[(i + 1)]; kB++) {
      GrB_Index k = B2_crd[kB];
      for (GrB_Index jC = C2_pos[k]; jC < C2_pos[(k + 1)]; jC++) {
        GrB_Index j = C2_crd[jC];
        if (!w_already_set[j]) {
          w[j] = B_vals[kB] && C_vals[jC];
          //w[j] = B_vals[kB] * C_vals[jC];
          w_index_list[w_index_list_all_size] = j;
          w_already_set[j] = 1;
          w_index_list_all_size++;
        }
        else {
          w[j] = w[j] || B_vals[kB] && C_vals[jC];
          //w[j] = w[j] + B_vals[kB] * C_vals[jC];
        }
      }
    }
    //qsort(w_index_list, w_index_list_all_size, sizeof(GrB_Index), cmp);

    for (GrB_Index w_index_locator = 0; w_index_locator < w_index_list_all_size; w_index_locator++) {
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

#if 0
#define TACO_MIN(_a,_b) ((_a) < (_b) ? (_a) : (_b))

int taco_extract(taco_tensor_t *A, taco_tensor_t *B, taco_tensor_t *I, taco_tensor_t *J) {
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
#endif

ir::Expr orImpl(const std::vector<ir::Expr>& v) {
  return ir::Or::make(v[0], v[1]);
}
Func OrOp("or", orImpl, {Annihilator(true), Identity(false), Commutative(), Associative()});

ir::Expr andImpl(const std::vector<ir::Expr>& v) {
  return ir::And::make(v[0], v[1]);
}
Func AndOp("and", andImpl, {Annihilator(false), Identity(true), Commutative(), Associative()});

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

#if 0
ir::Expr selectImpl(const std::vector<ir::Expr>& v) {
  return v[1];
}
ir::Expr defaultImpl(const std::vector<ir::Expr>& v) {
  return v[2];
}
struct SelectAlgebra {
  IterationAlgebra operator()(const std::vector<IndexExpr>& r) {
    return Union(Intersect(Complement(r[0]), r[1]), Intersect(r[0], r[2]));
    return Union(Intersect(Complement(r[0]), r[1]), r[2]);
    return Intersect(Complement(r[0]), r[1]);
  }
};
#endif

class BitmapModeFormat : public ModeFormatImpl {
public:
  BitmapModeFormat() :
      ModeFormatImpl("dense", false, true, true, false, false, true, false,
                     false, true, true, false) {}

  ~BitmapModeFormat() override {}

  ModeFormat copy(std::vector<ModeFormat::Property> properties) const override {
    return ModeFormat(std::make_shared<BitmapModeFormat>());  
  }
  
  ModeFunction locate(ir::Expr parentPos, std::vector<ir::Expr> coords,
                      Mode mode) const override {
    ir::Expr pos = ir::Add::make(ir::Mul::make(parentPos, getWidth(mode)), coords.back());
    return ModeFunction(ir::Stmt(), {pos, ir::Eq::make(ir::Load::make(getValidArray(mode.getModePack()), pos), 1)});
  }

  ir::Stmt getInsertCoord(ir::Expr p, const std::vector<ir::Expr>& i, 
                          Mode mode) const override {
    return ir::Store::make(getValidArray(mode.getModePack()), p, 1);
  }

  ir::Expr getWidth(Mode mode) const override {
    return (mode.getSize().isFixed() && mode.getSize().getSize() < 16) ?
           (int)mode.getSize().getSize() : 
           getSizeArray(mode.getModePack());
  }

  //ir::Stmt getInsertInitCoords(ir::Expr pBegin, ir::Expr pEnd, 
  //                             Mode mode) const override;

  ir::Stmt getInsertInitLevel(ir::Expr szPrev, ir::Expr sz, 
                              Mode mode) const override {
    return ir::Allocate::make(getValidArray(mode.getModePack()), sz, false, ir::Expr(), true);
  }

  std::vector<ir::Expr> getArrays(ir::Expr tensor, int mode, 
                                  int level) const override {
    return {ir::GetProperty::make(tensor, ir::TensorProperty::Dimension, mode),
            ir::GetProperty::make(tensor, ir::TensorProperty::Indices,
                              level - 1, 0, util::toString(tensor) + 
                              std::to_string(level) + "_valid")};
  }
  
  ir::Expr getSizeArray(ModePack pack) const {
    return pack.getArray(0);
  }

  ir::Expr getValidArray(ModePack pack) const {
    return pack.getArray(1);
  }
};
ModeFormat Bitmap(std::make_shared<BitmapModeFormat>());

static void bench_mxv_taco(benchmark::State& state) {
#if 0
  //std::map<std::vector<int>, FuncBodyGenerator> def;
  //def[{1,0,0}] = selectImpl;
  //Func SelectOp("select", selectImpl, SelectAlgebra(), {{{0, 2}, defaultImpl}});
  //Func SelectOp("select", selectImpl);

  taco_set_num_threads(nthreads);
  for (auto _ : state) {
    state.PauseTiming();

    IndexVar i, j;
    Format dv({Dense});
    Format bmv({Bitmap});
    //Tensor<double> A("A", {fixture.M, fixture.N}, CSR, std::numeric_limits<double>::infinity());
    //Tensor<double> y("y", {fixture.M}, bmv, std::numeric_limits<double>::infinity());
    //Tensor<double> x("x", {fixture.N}, bmv, std::numeric_limits<double>::infinity());
    Tensor<bool> A("A", {fixture.M, fixture.N}, CSR);
    Tensor<bool> y("y", {fixture.M}, bmv);
    Tensor<bool> x("x", {fixture.N}, bmv);
    Tensor<bool> m("m", {fixture.M}, dv);
    y(i) = MaskOp(Reduction(OrOp(), j, AndOp(A(i,j), x(j))), m(i));
    //y(i) = MaskOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), m(i));
    //y(i) = x(i);
    //y(i) = SelectOp(m(i), Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
    //y(i) = MaskOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
    //y(i) = Reduction(MinOp(), j, AddOp(A(i,j), x(j)));
    //y(i) = MinOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));
    //y(i) = MinOp(MaskOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i)), x(i));
    //y(i) = MaskOp(MinOp(Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i)), x(i));
    //y(i) = MinOp(FilterOp(x(i)) * Reduction(MinOp(), j, AddOp(A(i,j), x(j))), x(i));

    auto stmt = y.getAssignment().concretize().parallelize(i, ParallelUnit::CPUThread, OutputRaceStrategy::NoRaces);
    y.compile(stmt, true);

    state.ResumeTiming();

    y.assemble();
    y.compute();
  }
  taco_set_num_threads(1);
#else
  if (!fixture.A_taco_t) {
    fixture.A_taco_t = to_csr_taco_tensor(&fixture.A_gb);
  }
  if (!fixture.x_taco_t) {
    fixture.x_taco_t = to_bitmap_taco_tensor(&fixture.x_gb);
  }
  if (!fixture.m_taco_t) {
    fixture.m_taco_t = to_dense_taco_tensor(&fixture.m_gb);
  }
  taco_tensor_t y = new_bitmap_taco_tensor(fixture.M);
  for (auto _ : state) {
    state.PauseTiming();

    free_bitmap_taco_tensor(y);

    state.ResumeTiming();

    if (fixture.is_bool) {
      taco_mxv_bool(&y, fixture.A_taco_t, fixture.x_taco_t, fixture.m_taco_t);
    } else {
      taco_mxv_trop(&y, fixture.A_taco_t, fixture.x_taco_t, fixture.m_taco_t);
    }
  }
  if (fixture.validate && fixture.y_gb) {
    auto y_gb = to_bitmap_taco_tensor(&fixture.y_gb);
    std::cout << "comparing mxv: ";
    if (fixture.is_bool) {
      std::cout << compare_bool_bitmap(get_bitmap_arrays(y), get_bitmap_arrays(*y_gb));
    } else {
      std::cout << compare_double_bitmap(get_bitmap_arrays(y), get_bitmap_arrays(*y_gb));
    }
    std::cout << std::endl;
    fixture.y_gb = nullptr;
  }
  free_bitmap_taco_tensor(y);
#endif
}

static void bench_mxm_taco(benchmark::State& state) {
  if (!fixture.A_taco_t) {
    fixture.A_taco_t = to_csr_taco_tensor(&fixture.A_gb);
  }
  taco_tensor_t C = new_csr_taco_tensor(fixture.M, fixture.N);
  for (auto _ : state) {
    state.PauseTiming();

    free_csr_taco_tensor(C);

    state.ResumeTiming();

    if (fixture.is_bool) {
      taco_mxm_bool(&C, fixture.A_taco_t, fixture.A_taco_t);
    } else {
      taco_mxm_trop(&C, fixture.A_taco_t, fixture.A_taco_t);
    }
  }
  if (fixture.validate && fixture.C_gb) {
    auto C_gb = to_csr_taco_tensor(&fixture.C_gb);
    std::cout << "comparing mxm: ";
    if (fixture.is_bool) {
      std::cout << compare_bool_csr(get_csr_arrays(C), get_csr_arrays(*C_gb));
    } else {
      std::cout << compare_double_csr(get_csr_arrays(C), get_csr_arrays(*C_gb));
    }
    std::cout << std::endl;
    fixture.C_gb = nullptr;
  }
  free_csr_taco_tensor(C);
}

#if 0
static void bench_extract_taco(benchmark::State& state) {
  if (!fixture.A_taco_t) {
    fixture.A_taco_t = to_csr_taco_tensor(&fixture.A_gb);
  }
  taco_tensor_t B = new_csr_taco_tensor(fixture.indices.size(), fixture.indices.size());
  for (auto _ : state) {
    state.PauseTiming();

    free_csr_taco_tensor(B);

    state.ResumeTiming();

    taco_extract(&B, fixture.A_taco_t, &fixture.indices_taco, &fixture.indices_taco);
  }
  //std::cout << ((GrB_Index*)(B.indices[1][0]))[B.dimensions[0]] << std::endl;
  free_csr_taco_tensor(B);
}
#endif

GRAPHBLAS_BENCH(bench_mxv_suitesparse, 1000);
GRAPHBLAS_BENCH(bench_mxm_suitesparse, 25);
//GRAPHBLAS_BENCH(bench_extract_suitesparse, 10);
GRAPHBLAS_BENCH(bench_mxv_taco, 1000);
GRAPHBLAS_BENCH(bench_mxm_taco, 25);
//GRAPHBLAS_BENCH(bench_extract_taco, 10);

#endif
