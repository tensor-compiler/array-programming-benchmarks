import numpy as np
from scipy.sparse import random, csr_matrix
import sparse
import pytest
import os
from util import MinMaxPydataSparseTensorLoader

@pytest.mark.parametrize("dims", [1, 3, 5])
def bench_pydata_minmax(tacoBench, dims):
    loader = MinMaxPydataSparseTensorLoader()
    dims_list = [20] + [20] + [43 for ele in range(dims)]
    #FIXME: loader.random is always between 0 and 1, need to be larger. multiply by some value and then store to tns file
    #TODO: matrix shouldn't be completely random. it should have blocks of dense values (to simulate pruning) 
    #       and not just sparse uniform sampling
    
    matrix = loader.tensor(dims_list)
    def bench():
        reduced = matrix
        for m in range(len(dims_list)):   
            if m % 2 == 0: 
                reduced = np.max(reduced, -1)
            else:
                reduced = np.min(reduced, -1)     
        return reduced
    tacoBench(bench)

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
