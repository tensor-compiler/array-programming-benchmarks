import numpy
from scipy.sparse import random, csr_matrix
import sparse
import pytest

# TODO (rohany): Ask hameer about this. pydata/sparse isn't happy when
#  given this ufunc to evaluate.
def myfunc(x, y):
    return x + y
myufunc = numpy.frompyfunc(myfunc, 2, 1, identity=0)

# @pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.skip(reason="Not testing this right now")
@pytest.mark.parametrize("dim", [250, 500, 750, 1000, 2500, 5000, 7500, 8000])
@pytest.mark.parametrize("ufunc", [numpy.logical_xor, numpy.logical_or, numpy.right_shift, numpy.ldexp])
def bench_ufunc_dense(tacoBench, dim, ufunc):
    A = numpy.random.randint(2, size=(dim, dim)).astype(numpy.uint32)
    B = numpy.random.randint(2, size=(dim, dim)).astype(numpy.uint32)
    def bench():
        C = ufunc(A, B)
    tacoBench(bench)

# TODO (rohany): We can parametrize this over the sparsity as well.
# @pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.skip(reason="Not testing this right now")
@pytest.mark.parametrize("dim", [250, 500, 750, 1000, 2500, 5000, 7500, 8000])
@pytest.mark.parametrize("ufunc", [numpy.logical_xor, numpy.logical_or, numpy.right_shift, numpy.ldexp])
def bench_pydata_ufunc_sparse(tacoBench, dim, ufunc):
    A = sparse.random((dim, dim), data_rvs=numpy.ones).astype(numpy.uint32)
    B = sparse.random((dim, dim), data_rvs=numpy.ones).astype(numpy.uint32)
    def bench():
        C = ufunc(A, B)
    tacoBench(bench)


def import_tensor(filename, dim):
    print(filename)
    with open(filename, 'r') as f:
        lines = f.readlines()
        
        count = 0
        indptr = 0
        indices = 0
        data = 0
        for line in lines:
            count += 1
            if count == 5:
                indptr = line
                indptr = indptr[3: -2]
            if count == 6:
                indices = line
                indices = indices[3: -2]
            if count == 7:
                data = line
                data = data[1: -2]
        print("indptr", indptr)
        indptr = numpy.fromstring(indptr, dtype=int, sep=",")
        indices = numpy.fromstring(indices, dtype=int, sep=",")
        data = numpy.fromstring(data, dtype=float, sep=",")
    return indptr, indices, data

def get_ufunc_str(ufunc):
    if ufunc == numpy.logical_xor:
        return "xor"
@pytest.mark.parametrize("dim", [250, 500, 750, 1000, 2500, 5000, 7500, 8000])
@pytest.mark.parametrize("ufunc", [numpy.logical_xor])
def bench_pydata_import_ufunc_sparse(tacoBench, dim, ufunc):
    filenameA = "./data/bench_ufunc_sparse_"
    filenameA += get_ufunc_str(ufunc)
    filenameA += "_" + str(dim) + "_" + "010000_A.txt"
    indptrA, indicesA, dataA = import_tensor(filenameA, dim)

    A = csr_matrix((dataA, indicesA, indptrA), shape=(dim, dim)).toarray()

    filenameB = "./data/bench_ufunc_sparse_"
    filenameB += get_ufunc_str(ufunc)
    filenameB += "_" + str(dim) + "_" + "010000_B.txt"
    indptrB, indicesB, dataB = import_tensor(filenameB, dim)
    B = csr_matrix((dataB, indicesB, indptrB), shape=(dim, dim)).toarray()
    def bench():
        C = ufunc(A, B)
        return C
    tacoBench(bench)
    print("Result", bench())
