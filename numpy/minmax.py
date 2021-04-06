import numpy as np
from scipy.sparse import random, csr_matrix
import sparse
import pytest
import os
from util import TensorCollectionFROSTT, PydataTensorShifter, TensorCollectionSuiteSparse, ScipyTensorShifter, PydataMatrixMarketTensorLoader, ScipyMatrixMarketTensorLoader, VALIDATION_OUTPUT_PATH, PydataSparseTensorDumper, SuiteSparseTensor, safeCastPydataTensorToInts, RandomPydataSparseTensorLoader

@pytest.mark.parametrize("dims", [3, 5, 7])
def bench_pydata_minmax(tacoBench, dims):
    loader = RandomPydataSparseTensorLoader()
    dims_list = [20] + [20] + [43 for ele in range(dims)]
    #FIXME: loader.random is always between 0 and 1, need to be larger. multiply by some value and then store to tns file
    #TODO: matrix shouldn't be completely random. it should have blocks of dense values (to simulate pruning) 
    #       and not just sparse uniform sampling
    
    matrix = safeCastPydataTensorToInts(20*loader.random(dims_list, 0.10))
    print(matrix)
    def bench():
        reduced = matrix
        for m in range(len(dims_list)):   
            if m % 2 == 0: 
                reduced = np.max(reduced, -1)
            else:
                reduced = np.min(reduced, -1)     
            print(reduced)
            print(np.max(reduced))
        print(reduced)
        return reduced
    tacoBench(bench)
