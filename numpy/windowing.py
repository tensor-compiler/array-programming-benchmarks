import numpy
from scipy.sparse import random
import pytest

def test_add_sparse_window(benchmark):
    dim = 10000
    matrix = random(dim, dim, format="csr")

    def bench():
        res = matrix[1:(dim-1), 1:(dim-1)] + matrix[1:(dim-1), 1:(dim-1)]
    benchmark.pedantic(bench, iterations=10)

def test_add_multiple_sparse_windows(benchmark):
    dim = 10000
    matrix1 = random(dim, dim, format="csr")
    matrix2 = random(dim, dim, format="csr")

    def bench():
        res = matrix1[1:(dim-1), 1:(dim-1)] + matrix2[1:(dim-1), 1:(dim-1)] + matrix1[0:(dim-2), 0:(dim-2)]
    benchmark.pedantic(bench, iterations=10)
