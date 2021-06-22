import numpy as np
from scipy.sparse import random, csr_matrix
import sparse
import pytest
import os
from util import MinMaxPydataSparseTensorLoader, MinMaxScipySparseTensorLoader

@pytest.mark.parametrize("dims", [1, 3, 5])
def bench_pydata_minmax(tacoBench, dims):
    loader = MinMaxPydataSparseTensorLoader()
    dims_list = [20] + [20] + [43 for ele in range(dims)]
    
    matrix = loader.tensor(dims_list)
    extra_info = dict()
    extra_info["nnz"] = matrix.nnz
    def bench():
        reduced = matrix
        for m in range(len(dims_list)):   
            if m % 2 == 0: 
                reduced = np.max(reduced, -1)
            else:
                reduced = np.min(reduced, -1)     
        return reduced
    tacoBench(bench, extra_info, True)

@pytest.mark.parametrize("dims", [1, 3, 5])
def bench_scipy_minmax(tacoBench, dims):
    loader = MinMaxScipySparseTensorLoader()
    dims_list = [20] + [20] + [43 for ele in range(dims)]
    
    matrix = loader.tensor(dims_list)
    extra_info = dict()
    extra_info["nnz"] = matrix.nnz
    def bench():
        reduced = matrix
        for m in range(len(dims_list)):   
            if m % 2 == 0: 
                reduced = reduced.min(-1)
            else:
                reduced = reduced.max(-1)     
        return reduced
    tacoBench(bench, extra_info, True)

@pytest.mark.skip(reason="Only to get matrix statistics")
@pytest.mark.parametrize("dims", [1, 3, 5])
def bench_minmax_statistics(tacoBench, dims):
    loader = MinMaxPydataSparseTensorLoader()
    dims_list = [20] + [20] + [43 for ele in range(dims)]
    matrix = loader.tensor(dims_list)

    extra_info = dict()
    extra_info["nnz"] = matrix.nnz

    def nop():
        return 0
    tacoBench(nop, extra_info)
