import numpy as np
import cv2
import os
import pytest
import sparse
from util import ImagePydataSparseTensorLoader, safeCastPydataTensorToInts, TnsFileDumper, plot_image 

# import matplotlib.pyplot as plt 

@pytest.mark.parametrize("num", list(range(1, 311))) 
@pytest.mark.parametrize("pt1", [0.75])
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
            #plot_image(loader.img[num], bin_img1, bin_img2, xor_img, sparse_xor_img, t1, t2)

@pytest.mark.parametrize("num", list(range(1, 311))) 
@pytest.mark.parametrize("pt1", [0.75])
def bench_edge_detection_dense(tacoBench, num, pt1):
        loader = ImagePydataSparseTensorLoader()
        bin_img1 = loader.dense_image(num, pt1, 1) 
        bin_img2 = loader.dense_image(num, pt1 + 0.05, 2)

        def dense_bench():
            xor_img = np.logical_xor(bin_img1, bin_img2).astype('int')
            return xor_img
        tacoBench(dense_bench)

@pytest.mark.parametrize("num", list(range(1, 311))) 
@pytest.mark.parametrize("pt1", [0.75])
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

@pytest.mark.parametrize("num", list(range(1, 311))) 
@pytest.mark.parametrize("pt1", [0.75])
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
@pytest.mark.parametrize("num", list(range(1, 311))) 
@pytest.mark.parametrize("pt1", [0.75])
@pytest.mark.parametrize("window_size", [0.45, 0.4, 0.35, 0.3])
def bench_edge_detection_window_pydata(tacoBench, num, pt1, window_size, plot):
        loader = ImagePydataSparseTensorLoader()
        sparse_bin_img1 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1, 1))
        sparse_bin_img2 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1+0.05, 2))
        bin_img1 = loader.dense_image(num, pt1, 1) 
        bin_img2 = loader.dense_image(num, pt1 + 0.05, 2)

        mid0 = int(bin_img1.shape[0]/2)
        mid1 = int(bin_img1.shape[1]/2) 

        win_len0 = int(window_size * bin_img1.shape[0])
        win_len1 = int(window_size * bin_img1.shape[1])

        if plot:
            print(sparse_bin_img1.shape)
            print(sparse_bin_img2.shape)

        def sparse_bench():
            swin1 = sparse_bin_img1[mid0 - win_len0:mid0 + win_len0, mid1 - win_len1:mid1 + win_len1]
            swin2 = sparse_bin_img2[mid0 - win_len0:mid0 + win_len0, mid1 - win_len1:mid1 + win_len1]
            sparse_xor_img = np.logical_xor(swin1, swin2).astype('int')
            return sparse_xor_img

        def dense_bench():
            win1 = bin_img1[mid0 - win_len0:mid0 + win_len0, mid1 - win_len1:mid1 + win_len1]
            win2 = bin_img2[mid0 - win_len0:mid0 + win_len0, mid1 - win_len1:mid1 + win_len1]
            xor_img = np.logical_xor(win1, win2).astype('int')
            return xor_img

        ret = tacoBench(sparse_bench)
        sparse_xor_img = sparse_bench()
        xor_img = dense_bench()
        
        if plot:
            print(sparse_xor_img)
            print("sparse img1 nnz =", sparse_bin_img1.nnz, "    ", np.sum(bin_img1 != 0))
            print("sparse img2 nnz =", sparse_bin_img2.nnz, "    ", np.sum(bin_img2 != 0))
            num_elements = float(np.prod(bin_img1.shape))
            print("Sparse xor NNZ = ", sparse_xor_img.nnz, "\t", "Dense xor NNZ = ", np.sum(xor_img != 0))
            print("Sparsity img 1 ", np.sum(bin_img1 != 0) / num_elements)
            print("Sparsity img 2 ", np.sum(bin_img2 != 0) / num_elements)
            print("Sparsity xor ", np.sum(xor_img != 0) / num_elements)
            sparse_xor_img = sparse_xor_img.todense()
            t1 = round(loader.max[num]*pt1, 2)
            t2 = round(loader.max[num]*(pt1 + 0.05), 2)
            print(xor_img)
            #plot_image(loader.img[num], bin_img1, bin_img2, xor_img, sparse_xor_img, t1, t2)

        assert(sparse_xor_img.nnz == np.sum(xor_img != 0))

@pytest.mark.parametrize("num", list(range(1, 311))) 
@pytest.mark.parametrize("pt1", [0.75])
@pytest.mark.parametrize("window_size", [0.45, 0.4, 0.35, 0.3])
def bench_edge_detection_window_dense(tacoBench, num, pt1, window_size):
        loader = ImagePydataSparseTensorLoader()
        bin_img1 = loader.dense_image(num, pt1, 1) 
        bin_img2 = loader.dense_image(num, pt1 + 0.05, 2)

        mid0 = int(bin_img1.shape[0]/2)
        mid1 = int(bin_img1.shape[1]/2) 

        win_len0 = int(window_size * bin_img1.shape[0])
        win_len1 = int(window_size * bin_img1.shape[1])

        def dense_bench():
            win1 = bin_img1[mid0 - win_len0:mid0 + win_len0, mid1 - win_len1:mid1 + win_len1]
            win2 = bin_img2[mid0 - win_len0:mid0 + win_len0, mid1 - win_len1:mid1 + win_len1]
            xor_img = np.logical_xor(win1, win2).astype('int')
            return xor_img

        tacoBench(dense_bench)

# USED FOR TESTING ITTERATION LATTICE CONSTRUCTION TACO CODE ONLY
def testOp(a, b, c):
    return np.logical_and(np.logical_not(np.logical_and(a, c).astype('int')).astype('int'), np.logical_not(np.logical_and(b, c).astype('int')).astype('int')).astype('int')

@pytest.mark.skip(reason="Used for verification only")
@pytest.mark.parametrize("num", list(range(1, 11))) 
@pytest.mark.parametrize("pt1", [0.5])
def bench_test_fused_pydata(tacoBench, num, pt1):
        loader = ImagePydataSparseTensorLoader()
        sparse_bin_img1 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1, 1))
        sparse_bin_img2 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1+0.05, 2))
        sparse_bin_window = loader.sparse_window(num, 3)
        bin_img1 = loader.dense_image(num, pt1, 1) 
        bin_img2 = loader.dense_image(num, pt1 + 0.05, 2)
        bin_window = loader.dense_window(num)

        def sparse_bench():
            return testOp(sparse_bin_img1, sparse_bin_img2, sparse_bin_window).astype('int')

        def dense_bench():
            return testOp(bin_img1, bin_img2, bin_window).astype('int')

        ret = tacoBench(sparse_bench)
        sparse_xor_img = sparse_bench()
        xor_img = dense_bench()

        # Write result to TNS file to see what's different
        shape = xor_img.shape
        result = sparse.COO.from_numpy(xor_img, fill_value=0)
        dok = sparse.DOK(result)
        TnsFileDumper().dump_dict_to_file(shape, dok.data, os.path.join("temp", "numpy-result-{}.tns".format(num)))
        
    
        num_elements = float(np.prod(bin_img1.shape))
        f = sparse_xor_img.fill_value
        print("shape1", sparse_bin_img1.shape)
        print("shape2", sparse_bin_img2.shape)
        print("sparse img1 nnz =", sparse_bin_img1.nnz, "    ", np.sum(bin_img1 != 0))
        print("sparse img2 nnz =", sparse_bin_img2.nnz, "    ", np.sum(bin_img2 != 0))
        print("sparse win nnz =", sparse_bin_window.nnz, "    ", np.sum(bin_window != 0))
        print("Total num elements", num_elements)
        print("Fill value", f)
        print("Sparse xor NNF = ", sparse_xor_img.nnz, "\t", "Dense xor NNF = ", np.sum(xor_img != int(f)))
        print("Dense xor NNZ = ", np.sum(xor_img != 0))
        assert(sparse_xor_img.nnz == np.sum(xor_img != 1))

@pytest.mark.skip(reason="for getting the input matrices statistics only")
@pytest.mark.parametrize("num", list(range(1, 311))) 
@pytest.mark.parametrize("pt1", [0.75])
def bench_edge_detection_statistics(tacoBench, num, pt1):
        loader = ImagePydataSparseTensorLoader()
        sparse_bin_img1 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1, 1))
        sparse_bin_img2 = safeCastPydataTensorToInts(loader.sparse_image(num, pt1+0.05, 2))
        sparse_bin_window = loader.sparse_window(num, 3)

        print(sparse_bin_img1.shape)
        print(sparse_bin_img2.shape)
        extra_info = dict()
        extra_info['nnz1'] = sparse_bin_img1.nnz
        extra_info['nnz2'] = sparse_bin_img2.nnz
        extra_info['nnz3'] = sparse_bin_window.nnz
        extra_info['dimx'] = sparse_bin_window.shape[0]
        extra_info['dimy'] = sparse_bin_window.shape[1]

        def sparse_bench():
            sbi1 = np.logical_and(sparse_bin_img1, sparse_bin_window)
            sbi2 = np.logical_and(sparse_bin_img2, sparse_bin_window)
            sparse_xor_img = np.logical_xor(sbi1, sbi2).astype('int')
            return sparse_xor_img

        tacoBench(sparse_bench, extra_info)

@pytest.mark.skip(reasoun="For image generation only")
@pytest.mark.parametrize("num", [42, 44, 50, 63, 92]) 
@pytest.mark.parametrize("pt1", [0.75])
def bench_edge_detection_plot(tacoBench, num, pt1, plot):
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

        def xor():
            xor_img = np.logical_xor(sparse_bin_img1, sparse_bin_img2).astype('int')
            return xor_img

        ret = tacoBench(sparse_bench)
        sparse_xor_img = sparse_bench()
        xor_img = xor()

        if plot:
            num_elements = float(np.prod(bin_img1.shape))
            print("Sparse xor NNZ = ", sparse_xor_img.nnz, "\t", "Dense xor NNZ = ", np.sum(xor_img != 0))
            print("Sparsity img 1 ", np.sum(bin_img1 != 0) / num_elements)
            print("Sparsity img 2 ", np.sum(bin_img2 != 0) / num_elements)
            print("Sparsity xor ", np.sum(xor_img != 0) / num_elements)
            sparse_xor_img = sparse_xor_img.todense()
            t1 = round(loader.max[num]*pt1, 2)
            t2 = round(loader.max[num]*(pt1 + 0.05), 2)
            #plot_image(loader.img[num], bin_img1, bin_img2, xor_img.todense(), sparse_xor_img, t1, t2, bin_window)


