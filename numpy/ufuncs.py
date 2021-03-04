import numpy
from scipy.sparse import random
import sparse
import pytest

# TODO (rohany): Ask hameer about this. pydata/sparse isn't happy when
#  given this ufunc to evaluate.
def myfunc(x, y):
    return x + y
myufunc = numpy.frompyfunc(myfunc, 2, 1, identity=0)

# @pytest.mark.parametrize("dim", [5000, 10000, 20000])
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
@pytest.mark.parametrize("dim", [250, 500, 750, 1000, 2500, 5000, 7500, 8000])
@pytest.mark.parametrize("ufunc", [numpy.logical_xor, numpy.logical_or, numpy.right_shift, numpy.ldexp])
def bench_pydata_ufunc_sparse(tacoBench, dim, ufunc):
    A = sparse.random((dim, dim), data_rvs=numpy.ones).astype(numpy.uint32)
    B = sparse.random((dim, dim), data_rvs=numpy.ones).astype(numpy.uint32)
    def bench():
        C = ufunc(A, B)
    tacoBench(bench)
