import numpy as np
import cv2
import os
import pytest
import matplotlib.pyplot as plt 
import sparse
from util import ImagePydataSparseTensorLoader, safeCastPydataTensorToInts, plot_image



@pytest.mark.parametrize("num", list(range(1, 99))) 
@pytest.mark.parametrize("pt1", [0.5])
def bench_edge_detection_pydata(tacoBench, num, pt1, plot):
        loader = ImagePydataSparseTensorLoader()
        sparse_bin_img1 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1, 1))
        sparse_bin_img2 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1+0.05, 2))
        bin_img1 = loader.dense_image(num, pt1, 1) 
        bin_img2 = loader.dense_image(num, pt1 + 0.05, 2)
        if plot:
            print(sparse_bin_img1.shape)
            print(sparse_bin_img2.shape)

        def sparse_bench():
            sparse_xor_img = np.logical_xor(sparse_bin_img1, sparse_bin_img2).astype('int')
            return sparse_xor_img

        def dense_bench():
            xor_img = np.logical_xor(bin_img1, bin_img2).astype('int')
            return xor_img
        ret = tacoBench(sparse_bench)
        sparse_xor_img = sparse_bench()
        xor_img = dense_bench()
        
        assert(sparse_xor_img.nnz == np.sum(xor_img != 0))

        if plot:
            num_elements = float(np.prod(bin_img1.shape))
            print("Sparse xor NNZ = ", sparse_xor_img.nnz, "\t", "Dense xor NNZ = ", np.sum(xor_img != 0))
            print("Sparsity img 1 ", np.sum(bin_img1 != 0) / num_elements)
            print("Sparsity img 2 ", np.sum(bin_img2 != 0) / num_elements)
            print("Sparsity xor ", np.sum(xor_img != 0) / num_elements)
            sparse_xor_img = sparse_xor_img.todense()
            t1 = round(loader.max[num]*pt1, 2)
            t2 = round(loader.max[num]*(pt1 + 0.05), 2)
            plot_image(loader.img[num], bin_img1, bin_img2, xor_img, sparse_xor_img, t1, t2)

@pytest.mark.parametrize("num", list(range(1, 99))) 
@pytest.mark.parametrize("pt1", [0.5])
def bench_edge_detection_dense(tacoBench, num, pt1):
        loader = ImagePydataSparseTensorLoader()
        bin_img1 = loader.dense_image(num, pt1, 1) 
        bin_img2 = loader.dense_image(num, pt1 + 0.05, 2)

        def dense_bench():
            xor_img = np.logical_xor(bin_img1, bin_img2).astype('int')
            return xor_img
        tacoBench(dense_bench)

@pytest.mark.parametrize("num", list(range(1, 99))) 
@pytest.mark.parametrize("pt1", [0.5])
def bench_edge_detection_fused_pydata(tacoBench, num, pt1, plot):
        loader = ImagePydataSparseTensorLoader()
        sparse_bin_img1 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1, 1))
        sparse_bin_img2 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1+0.05, 2))
        sparse_bin_window = loader.sparse_window(num, 3)
        bin_img1 = loader.dense_image(num, pt1, 1) 
        bin_img2 = loader.dense_image(num, pt1 + 0.05, 2)
        bin_window = loader.dense_window(num)

        if plot:
            print(sparse_bin_img1.shape)
            print(sparse_bin_img2.shape)

        def sparse_bench():
            sbi1 = np.logical_and(sparse_bin_img1, sparse_bin_window)
            sbi2 = np.logical_and(sparse_bin_img2, sparse_bin_window)
            sparse_xor_img = np.logical_xor(sbi1, sbi2).astype('int')
            return sparse_xor_img

        def dense_bench():
            bi1 = np.logical_and(bin_img1, bin_window).astype('int')
            bi2 = np.logical_and(bin_img2, bin_window).astype('int')
            xor_img = np.logical_xor(bi1, bi2).astype('int')
            return xor_img
        ret = tacoBench(sparse_bench)
        sparse_xor_img = sparse_bench()
        xor_img = dense_bench()
        
        if plot:
            num_elements = float(np.prod(bin_img1.shape))
            print("Sparse xor NNZ = ", sparse_xor_img.nnz, "\t", "Dense xor NNZ = ", np.sum(xor_img != 0))
            print("Sparsity img 1 ", np.sum(bin_img1 != 0) / num_elements)
            print("Sparsity img 2 ", np.sum(bin_img2 != 0) / num_elements)
            print("Sparsity xor ", np.sum(xor_img != 0) / num_elements)
            sparse_xor_img = sparse_xor_img.todense()
            t1 = round(loader.max[num]*pt1, 2)
            t2 = round(loader.max[num]*(pt1 + 0.05), 2)
            plot_image(loader.img[num], bin_img1, bin_img2, xor_img, sparse_xor_img, t1, t2, bin_window)

        assert(sparse_xor_img.nnz == np.sum(xor_img != 0))

@pytest.mark.parametrize("num", list(range(1, 99))) 
@pytest.mark.parametrize("pt1", [0.5])
def bench_edge_detection_fused_dense(tacoBench, num, pt1):
        loader = ImagePydataSparseTensorLoader()
        bin_img1 = loader.dense_image(num, pt1, 1) 
        bin_img2 = loader.dense_image(num, pt1 + 0.05, 2)
        bin_window = loader.dense_window(num)

        def dense_bench():
            bi1 = np.logical_and(bin_img1, bin_window).astype('int')
            bi2 = np.logical_and(bin_img2, bin_window).astype('int')
            xor_img = np.logical_xor(bin_img1, bin_img2).astype('int')
            return xor_img
        tacoBench(dense_bench)

#TODO: Add in a benchmark that uses windowing for medical imaging as well. 

if __name__=="__main__":
    main()

