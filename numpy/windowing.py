import numpy
from scipy.sparse import random
import pytest

def test_add_window(benchmark):
    dim = 10000
    matrix = random(dim, dim, format="csr").todense()
    def bench():
        x = matrix[1:(dim-1), 1:(dim-1)] 
        res = x + x
    benchmark.pedantic(bench, iterations=10)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
def test_add_sparse_window(benchmark, dim):
    matrix = random(dim, dim, format="csr")
    def bench():
        x = matrix[1:(dim-1), 1:(dim-1)] 
        res = x + x
        # Sanity check that this has a similar runtime as taco.
        # res = matrix + matrix
    benchmark.pedantic(bench, iterations=10)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
def test_add_multiple_sparse_windows(benchmark, dim):
    matrix1 = random(dim, dim, format="csr")
    matrix2 = random(dim, dim, format="csr")
    def bench():
        res = matrix1[1:(dim-1), 1:(dim-1)] + matrix2[1:(dim-1), 1:(dim-1)] + matrix1[0:(dim-2), 0:(dim-2)]
    benchmark.pedantic(bench, iterations=10)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
def test_add_sparse_strided_window(benchmark, dim):
    matrix = random(dim, dim, format="csr")
    def bench():
        x = matrix[1:(dim-1):2, 1:(dim-1):2] 
        res = x + x
    benchmark.pedantic(bench, iterations=10)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
def test_add_sparse_index_set(benchmark, dim):
    indexes = [i * 2 for i in range(0, dim//2)]
    matrix = random(dim, dim, format="csr")
    def bench():
        x = matrix[:, indexes] 
        res = x + x
    benchmark.pedantic(bench, iterations=10)

