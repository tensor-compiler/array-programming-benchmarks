import numpy
from scipy.sparse import random
import sparse
import pytest

from util import RandomScipySparseTensorLoader, RandomPydataSparseTensorLoader

@pytest.mark.skip(reason="too slow right now")
def bench_add_window(tacoBench):
    dim = 10000
    matrix = random(dim, dim, format="csr").todense()
    def bench():
        x = matrix[1:(dim-1), 1:(dim-1)]
        res = x + x
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
def bench_add_sparse_window(tacoBench, dim, format):
    loader = RandomScipySparseTensorLoader(format)
    matrix = loader.random((dim, dim), 0.01)
    def bench():
        x = matrix[1:(dim-1), 1:(dim-1)] 
        res = x + x
        # Sanity check that this has a similar runtime as taco.
        # res = matrix + matrix
    tacoBench(bench)

# TODO (rohany): This is really slow (compared to scipy.sparse). Check with hameer
#  that this result makes sense.
@pytest.mark.parametrize("dim", [5000, 10000, 20000])
def bench_add_pydata_sparse_window(tacoBench, dim):
    loader = RandomPydataSparseTensorLoader()
    matrix = loader.random((dim, dim), 0.01)
    def bench():
        x = matrix[1:(dim-1), 1:(dim-1)] 
        res = x + x
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
def bench_add_sparse_strided_window(tacoBench, dim, format):
    matrix = random(dim, dim, format=format)
    def bench():
        x = matrix[1:(dim-1):4, 1:(dim-1):4] 
        res = x + x
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
def bench_add_multiple_sparse_windows(tacoBench, dim, format):
    matrix1 = random(dim, dim, format=format)
    matrix2 = random(dim, dim, format=format)
    def bench():
        res = matrix1[1:(dim-1), 1:(dim-1)] + matrix2[1:(dim-1), 1:(dim-1)] + matrix1[0:(dim-2), 0:(dim-2)]
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
def bench_add_sparse_strided_window(tacoBench, dim, format):
    matrix = random(dim, dim, format=format)
    def bench():
        x = matrix[1:(dim-1):2, 1:(dim-1):2] 
        res = x + x
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
def bench_add_sparse_index_set(tacoBench, dim, format):
    indexes = [i * 2 for i in range(0, dim//2)]
    matrix = random(dim, dim, format=format)
    def bench():
        x = matrix[:, indexes] 
        res = x + x
    tacoBench(bench)
