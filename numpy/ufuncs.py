import numpy
from scipy.sparse import random
import pytest

# TODO (rohany): We can parametrize this over the sparsity as well.
# @pytest.mark.parametrize("format", ['csr', 'csc'])
# @pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("dim", [250, 500, 750, 1000, 2500, 5000, 7500, 8000])
def bench_xor_sparse(tacoBench, dim):
    # TODO (rohany): It doesn't look like scipy.sparse works with
    #  logical_xor. pydata.sparse might however.
    # A = random(dim, dim, format="csr", data_rvs=numpy.ones)
    # B = random(dim, dim, format="csr", data_rvs=numpy.ones)
    A = numpy.random.randint(2, size=(dim, dim))
    B = numpy.random.randint(2, size=(dim, dim))
    def bench():
        C = numpy.logical_xor(A, B)
    tacoBench(bench)
