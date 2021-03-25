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
sizeConfigs = ["Constant", "ConstantFraction", "AlmostWhole", "Whole", "NoWindowing"]

def sliceTensor(tensor, dim, config):
    if config == "Constant":
        return tensor[250:750, 250:750]
    elif config == "ConstantFraction":
        size = dim//4
        start = dim//4
        return tensor[start:(start+size), start:(start+size)]
    elif config == "AlmostWhole":
        return tensor[1:(dim-1), 1:(dim-1)]
    elif config == "Whole":
        return tensor[0:dim, 0:dim]
    elif config == "NoWindowing":
        return tensor
    else:
        assert(False)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr'])
@pytest.mark.parametrize("config", sizeConfigs)
def bench_add_sparse_window(tacoBench, dim, format, config):
    loader = RandomScipySparseTensorLoader(format)
    matrix = loader.random((dim, dim), 0.01).astype('float64')
    matrix2 = loader.random((dim, dim), 0.01, variant=1).astype('float64')
    def bench():
        x = sliceTensor(matrix, dim, config)
        x2 = sliceTensor(matrix2, dim, config)
        res = x + x2
        # Sanity check that this has a similar runtime as taco.
        # res = matrix + matrix
    tacoBench(bench)

# TODO (rohany): This is really slow (compared to scipy.sparse). Check with hameer
#  that this result makes sense.
@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("config", sizeConfigs)
def bench_add_pydata_sparse_window(tacoBench, dim, config):
    loader = RandomPydataSparseTensorLoader()
    matrix = loader.random((dim, dim), 0.01).astype('float64')
    matrix2 = loader.random((dim, dim), 0.01, variant=1).astype('float64')
    def bench():
        x = sliceTensor(matrix, dim, config)
        x2 = sliceTensor(matrix2, dim, config)
        res = x + x2
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr'])
@pytest.mark.parametrize("strideWidth", [2, 4, 8])
def bench_add_sparse_strided_window(tacoBench, dim, format, strideWidth):
    loader = RandomScipySparseTensorLoader(format)
    matrix = loader.random((dim, dim), 0.01).astype('float64')
    matrix2 = loader.random((dim, dim), 0.01, variant=1).astype('float64')
    def bench():
        x = matrix[0:dim:strideWidth, 0:dim:strideWidth] 
        x2 = matrix2[0:dim:strideWidth, 0:dim:strideWidth] 
        res = x + x2
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
@pytest.mark.parametrize("fraction", [2, 4, 8])
@pytest.mark.skip(reason="not doing index sets")
def bench_add_sparse_index_set(tacoBench, dim, format, fraction):
    indexes = [i * fraction for i in range(0, dim//fraction)]
    loader = RandomScipySparseTensorLoader(format)
    matrix = loader.random((dim, dim), 0.01)
    matrix2 = loader.random((dim, dim), 0.01, variant=1)
    def bench():
        x = matrix[:, indexes] 
        x2 = matrix2[:, indexes] 
        res = x + x2
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("strideWidth", [2, 4, 8])
def bench_add_pydata_sparse_strided_window(tacoBench, dim, strideWidth):
    loader = RandomPydataSparseTensorLoader()
    matrix = loader.random((dim, dim), 0.01).astype('float64')
    matrix2 = loader.random((dim, dim), 0.01, variant=1).astype('float64')
    def bench():
        x = matrix[0:dim:strideWidth, 0:dim:strideWidth] 
        x2 = matrix2[0:dim:strideWidth, 0:dim:strideWidth] 
        res = x + x2
    tacoBench(bench)

# TODO (rohany): This is really slow (compared to scipy.sparse). Check with hameer
#  that this result makes sense.
@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("fraction", [2, 4, 8])
@pytest.mark.skip(reason="not doing index sets")
def bench_add_pydata_sparse_index_set(tacoBench, dim, fraction):
    loader = RandomPydataSparseTensorLoader()
    indexes = [i * fraction for i in range(0, dim//fraction)]
    matrix = loader.random((dim, dim), 0.01)
    matrix2 = loader.random((dim, dim), 0.01, variant=1)
    def bench():
        x = matrix[:, indexes] 
        x2 = matrix2[:, indexes] 
        res = x + x2
    tacoBench(bench)

@pytest.mark.parametrize("dim", [5000, 10000, 20000])
@pytest.mark.parametrize("format", ['csr', 'csc'])
@pytest.mark.skip(reason="not using currently")
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

