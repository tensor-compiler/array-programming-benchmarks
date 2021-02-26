import numpy
from scipy.sparse import random
import pytest

@pytest.mark.parametrize("dim", [250, 500, 750, 1000, 2500, 5000, 7500, 8000])
def bench_add_dense_threshold(tacoBench, dim):
    matrix1 = random(dim, dim, format="csr").todense()
    matrix2 = random(dim, dim, format="csr").todense()
    def bench():
        res = matrix1 + matrix2
    tacoBench(bench)
