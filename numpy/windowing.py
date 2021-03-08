import numpy
from scipy.sparse import random
import sparse
import pytest

from util import RandomScipySparseTensorLoader, RandomPydataSparseTensorLoader

# Want to run these windows so that they operate on
# * A small, constant size grid: 500x500
# * A constant fraction of the tensor size : 1/4 x 1/4
# * Almost the entire tensor (I'm unsure what the point of this comparison is)
# * No windowing (TACO should use a window so we can measure the overhead).
# These options map to the corresponding config values:
sizeConfigs = ["constant", "constant-fraction", "almost-whole", "no-windowing"]

def sliceTensor(tensor, dim, config):
    if config == "constant":
        return tensor[250:750, 250:750]
    elif config == "constant-fraction":
        size = dim//4
        start = dim//4
        return tensor[start:(start+size), start:(start+size)]
    elif config == "almost-whole":
        return tensor[1:(dim-1), 1:(dim-1)]
    elif config == "no-windowing":
        return tensor
    else:
        assert(False)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
@pytest.mark.parametrize("config", sizeConfigs)
def bench_add_sparse_window(tacoBench, dim, format, config):
    loader = RandomScipySparseTensorLoader(format)
    matrix = loader.random((dim, dim), 0.01)
    def bench():
        x = sliceTensor(matrix, dim, config)
        res = x + x
        # Sanity check that this has a similar runtime as taco.
        # res = matrix + matrix
    tacoBench(bench)

# TODO (rohany): This is really slow (compared to scipy.sparse). Check with hameer
#  that this result makes sense.
@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("config", sizeConfigs)
def bench_add_pydata_sparse_window(tacoBench, dim, config):
    loader = RandomPydataSparseTensorLoader()
    matrix = loader.random((dim, dim), 0.01)
    def bench():
        x = sliceTensor(matrix, dim, config)
        res = x + x
    tacoBench(bench)

# TODO (rohany): Parametrize the below tests by appropriate windowing config.

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
def bench_add_sparse_strided_window(tacoBench, dim, format):
    loader = ScipySparseTensorLoader(format)
    matrix = loader.random((dim, dim), 0.01)
    def bench():
        x = matrix[1:(dim-1):4, 1:(dim-1):4] 
        res = x + x
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
def bench_add_sparse_index_set(tacoBench, dim, format):
    indexes = [i * 2 for i in range(0, dim//2)]
    loader = ScipySparseTensorLoader(format)
    matrix = loader.random((dim, dim), 0.01)
    def bench():
        x = matrix[:, indexes] 
        res = x + x
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
def bench_add_pydata_sparse_strided_window(tacoBench, dim):
    loader = RandomPydataSparseTensorLoader()
    matrix = loader.random((dim, dim), 0.01)
    def bench():
        x = matrix[1:(dim-1):4, 1:(dim-1):4] 
        res = x + x
    tacoBench(bench)

# TODO (rohany): This is really slow (compared to scipy.sparse). Check with hameer
#  that this result makes sense.
@pytest.mark.parametrize("dim", [5000, 10000, 20000])
def bench_add_pydata_sparse_index_set(tacoBench, dim):
    loader = RandomPydataSparseTensorLoader()
    indexes = [i * 2 for i in range(0, dim//2)]
    matrix = loader.random((dim, dim), 0.01)
    def bench():
        x = matrix[:, indexes] 
        res = x + x
    tacoBench(bench)

# TODO (rohany): I don't know if we care about this benchmark.
@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
def bench_add_multiple_sparse_windows(tacoBench, dim, format):
    matrix1 = random(dim, dim, format=format)
    matrix2 = random(dim, dim, format=format)
    def bench():
        res = matrix1[1:(dim-1), 1:(dim-1)] + matrix2[1:(dim-1), 1:(dim-1)] + matrix1[0:(dim-2), 0:(dim-2)]
    tacoBench(bench)

@pytest.mark.skip(reason="too slow right now")
def bench_add_window(tacoBench):
    dim = 10000
    matrix = random(dim, dim, format="csr").todense()
    def bench():
        x = matrix[1:(dim-1), 1:(dim-1)]
        res = x + x
    tacoBench(bench)

