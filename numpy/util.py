import scipy.sparse
import scipy.io
import sparse
import os
import glob
import numpy
import cv2

# Get the path to the directory holding random tensors. Error out
# if this isn't set.
TENSOR_PATH = os.environ['TACO_TENSOR_PATH']
# Get the validation path, if it exists.
VALIDATION_OUTPUT_PATH = os.getenv('VALIDATION_OUTPUT_PATH', None)

# TnsFileLoader loads a tensor stored in .tns format.
class TnsFileLoader:
    def __init__(self):
        pass

    def load(self, path):
        coordinates = []
        values = []
        dims = []
        first = True
        with open(path, 'r') as f:
            for line in f:
                data = line.split(' ')
                if first:
                    first = False
                    dims = [0] * (len(data) - 1)
                    for i in range(len(data) - 1):
                        coordinates.append([])

                for i in range(len(data) - 1):
                    coordinates[i].append(int(data[i]) - 1)
                    dims[i] = max(dims[i], coordinates[i][-1] + 1)
                # TODO (rohany): What if we want this to be an integer?
                values.append(float(data[-1]))
        return dims, coordinates, values

# TnsFileDumper dumps a dictionary of coordinates to values
# into a coordinate list tensor file.
class TnsFileDumper:
    def __init__(self):
        pass

    def dump_dict_to_file(self, shape, data, path, write_shape = False):
        # Sort the data so that the output is deterministic.
        sorted_data = sorted([list(coords) + [value] for coords, value in data.items()])
        with open(path, 'w+') as f:
            for line in sorted_data:
                coords = [str(elem + 1) for elem in line[:len(line) - 1]]
                strings = coords + [str(line[-1])]
                f.write(" ".join(strings))
                f.write("\n")
            if write_shape:
                shape_strings = [str(elem) for elem in shape] + ['0']
                f.write(" ".join(shape_strings))
                f.write("\n")

# ScipySparseTensorLoader loads a sparse tensor from a file into a
# scipy.sparse CSR matrix.
class ScipySparseTensorLoader:
    def __init__(self, format):
        self.loader = TnsFileLoader()
        self.format = format

    def load(self, path):
        dims, coords, values = self.loader.load(path)
        if self.format == "csr":
            return scipy.sparse.csr_matrix((values, (coords[0], coords[1])), shape=tuple(dims))
        elif self.format == "csc":
            return scipy.sparse.csc_matrix((values, (coords[0], coords[1])), shape=tuple(dims))
        else:
            assert(False)

# PydataSparseTensorLoader loads a sparse tensor from a file into
# a pydata.sparse tensor.
class PydataSparseTensorLoader:
    def __init__(self):
        self.loader = TnsFileLoader()
    
    def load(self, path):
        dims, coords, values = self.loader.load(path)
        return sparse.COO(coords, values, tuple(dims))

# PydataSparseTensorDumper dumps a sparse tensor to a the desired file.
class PydataSparseTensorDumper:
    def __init__(self):
        self.dumper = TnsFileDumper()

    def dump(self, tensor, path):
        self.dumper.dump_dict_to_file(tensor.shape, sparse.DOK(tensor).data, path)

# construct_random_tensor_key constructs a unique key that represents
# a random tensor parameterized by the chosen shape and sparsity.
# The key itself is formatted by the dimensions, followed by the
# sparsity. For example, a 250 by 250 tensor with sparsity 0.01
# would have a key of 250x250-0.01.tns.
def construct_random_tensor_key(shape, sparsity, variant):
    path = TENSOR_PATH
    dims = "x".join([str(dim) for dim in shape])
    if variant is None:
        key = "{}-{}.tns".format(dims, sparsity)
    else:
        key = "{}-{}-{}.tns".format(dims, sparsity, variant)
    return os.path.join(path, "random", key)

# RandomPydataSparseTensorLoader should be used to generate
# random pydata.sparse tensors. It caches the loaded tensors
# in the file system so that TACO benchmarks using tensors
# with the same parameters can use the exact same tensors.
class RandomPydataSparseTensorLoader:
    def __init__(self):
        self.loader = PydataSparseTensorLoader()

    def random(self, shape, sparsity, variant=None):
        key = construct_random_tensor_key(shape, sparsity, variant)
        # If a tensor with these properties exists already, then load it.
        if os.path.exists(key):
            return self.loader.load(key)
        else:
            # Otherwise, we must create a random tensor with the desired properties,
            # dump it to the output file, then return it.
            result = sparse.random(shape, density=sparsity)
            dok = sparse.DOK(result)
            TnsFileDumper().dump_dict_to_file(shape, dok.data, key)
            return result

# RandomScipySparseTensorLoader is the same as RandomPydataSparseTensorLoader
# but for scipy.sparse tensors.
class RandomScipySparseTensorLoader:
    def __init__(self, format):
        self.loader = ScipySparseTensorLoader(format)
        self.format = format

    def random(self, shape, sparsity, variant=None):
        assert(len(shape) == 2)
        key = construct_random_tensor_key(shape, sparsity, variant)
        # If a tensor with these properties exists already, then load it.
        if os.path.exists(key):
            return self.loader.load(key)
        else:
            # Otherwise, create and then dump a tensor.
            result = scipy.sparse.random(shape[0], shape[1], density=sparsity, format=self.format)
            dok = scipy.sparse.dok_matrix(result)
            TnsFileDumper().dump_dict_to_file(shape, dict(dok.items()), key)
            return result

# FROSTTTensor represents a tensor in the FROSTT dataset.
class FROSTTTensor:
    def __init__(self, path):
        self.path = path
        self.__name__ = self.__str__()

    def __str__(self):
        f = os.path.split(self.path)[1]
        return f.replace(".tns", "")

    def load(self):
        return PydataSparseTensorLoader().load(self.path)

# TensorCollectionFROSTT represents the set of all FROSTT tensors.
class TensorCollectionFROSTT:
    def __init__(self):
        data = os.path.join(TENSOR_PATH, "FROSTT")
        frostttensors = glob.glob(os.path.join(data, "*.tns"))
        self.tensors = [FROSTTTensor(t) for t in frostttensors]
        allowlist = ["nips", "uber-pickups", "chicago-crime", "enron", "nell-2", "vast"]
        self.tensors = [t for t in self.tensors if str(t) in allowlist]

    def getTensors(self):
        return self.tensors
    def getTensorNames(self):
        return [str(tensor) for tensor in self.getTensors()]
    def getTensorsAndNames(self):
        return [(str(tensor), tensor) for tensor in self.getTensors()]

# PydataTensorShifter shifts all elements in the last mode
# of the input pydata/sparse tensor by one.
class PydataTensorShifter:
    def __init__(self):
        pass

    def shiftLastMode(self, tensor):
        coords = tensor.coords
        data = tensor.data
        resultCoords = []
        for j in range(len(tensor.shape)):
            resultCoords.append([0] * len(data))
        resultValues = [0] * len(data)
        for i in range(len(data)):
            for j in range(len(tensor.shape)):
                resultCoords[j][i] = coords[j][i]
            # resultValues[i] = data[i]
            # TODO (rohany): Temporarily use a constant as the value.
            resultValues[i] = 2
            # For order 2 tensors, always shift the last coordinate. Otherwise, shift only coordinates
            # that have even last coordinates. This ensures that there is at least some overlap
            # between the original tensor and its shifted counter part.
            if len(tensor.shape) <= 2 or resultCoords[-1][i] % 2 == 0:
                resultCoords[-1][i] = (resultCoords[-1][i] + 1) % tensor.shape[-1]
        return sparse.COO(resultCoords, resultValues, tensor.shape)

# ScipyTensorShifter shifts all elements in the last mode
# of the input scipy/sparse tensor by one.
class ScipyTensorShifter:
    def __init__(self, format):
        self.format = format

    def shiftLastMode(self, tensor):
        dok = scipy.sparse.dok_matrix(tensor)
        result = scipy.sparse.dok_matrix(tensor.shape)
        for coord, val in dok.items():
            newCoord = list(coord[:])
            newCoord[-1] = (newCoord[-1] + 1) % tensor.shape[-1]
            # result[tuple(newCoord)] = val
            # TODO (rohany): Temporarily use a constant as the value.
            result[tuple(newCoord)] = 2
        if self.format == "csr":
            return scipy.sparse.csr_matrix(result)
        elif self.format == "csc":
            return scipy.sparse.csc_matrix(result)
        else:
            assert(False)

# ScipyMatrixMarketTensorLoader loads tensors in the matrix market format
# into scipy.sparse matrices.
class ScipyMatrixMarketTensorLoader:
    def __init__(self, format):
        self.format = format 

    def load(self, path):
        coo = scipy.io.mmread(path)
        if self.format == "csr":
            return scipy.sparse.csr_matrix(coo)
        elif self.format == "csc":
            return scipy.sparse.csc_matrix(coo)
        else:
            assert(False)

# PydataMatrixMarketTensorLoader loads tensors in the matrix market format
# into pydata.sparse matrices.
class PydataMatrixMarketTensorLoader:
    def __init__(self):
        pass

    def load(self, path):
        coo = scipy.io.mmread(path)
        return sparse.COO.from_scipy_sparse(coo)

# SuiteSparseTensor represents a tensor in the suitesparse collection.
class SuiteSparseTensor:
    def __init__(self, path):
        self.path = path
        self.__name__ = self.__str__()

    def __str__(self):
        f = os.path.split(self.path)[1]
        return f.replace(".mtx", "")

    def load(self, loader):
        return loader.load(self.path)

# TensorCollectionSuiteSparse represents the set of all downloaded
# SuiteSparse tensors.
class TensorCollectionSuiteSparse:
    def __init__(self):
        data = os.path.join(TENSOR_PATH, "suitesparse")
        sstensors = glob.glob(os.path.join(data, "*.mtx"))
        self.tensors = [SuiteSparseTensor(t) for t in sstensors]

    def getTensors(self):
        return self.tensors
    def getTensorNames(self):
        return [str(tensor) for tensor in self.getTensors()]
    def getTensorsAndNames(self):
        return [(str(tensor), tensor) for tensor in self.getTensors()]

# safeCastPydataTensorToInts casts a floating point tensor to integers
# in a way that preserves the sparsity pattern.
def safeCastPydataTensorToInts(tensor):
    data = numpy.zeros(len(tensor.data), dtype='int64')
    for i in range(len(data)):
        # If the cast would turn a value into 0, instead write a 1. This preserves
        # the sparsity pattern of the data.
        if int(tensor.data[i]) == 0:
            data[i] = 1
        else:
            data[i] = int(tensor.data[i])
    return sparse.COO(tensor.coords, data, tensor.shape)


###########################
# Imaging Benchmark Utils #
###########################

# load_image loads an image with the correct color format for the numpy/image.py 
# benchmark
def load_image(image_folder, num):
    if image_folder == 'no':
        image_folder = "./data/image/no"
    else:
        image_folder = "./data/image/yes"

    name = "image" + str(num) + '.'  
    file_names = [fn for fn in os.listdir(image_folder)
                  if fn.startswith(name)]
    path = os.path.join(image_folder, file_names[0])
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

# thresh thresholdes the given image by a threshold
def thresh(images, t=85):
    if len(images.shape) < 3:
        images = numpy.expand_dims(images, axis=0)
    thresh_imgs = []
    for i in range(images.shape[0]):
        img = images[i]
        ret, thresh_img = cv2.threshold(img, t, 255, cv2.THRESH_BINARY)
        thresh_imgs.append(thresh_img)

    thresh_imgs = numpy.stack(thresh_imgs, axis=0)        
    return thresh_imgs 

# construct_image_tensor_key constructs a unique key that represents
# an image tensor parameterized by the image number and threshold.
# The key itself is formatted by the image number, followed by the
# threshold. For example, image1.* image with threshold of 0.5
# would have a key of image1-0.05.tns.
def construct_image_tensor_key(num, pt, variant):
    path = TENSOR_PATH
    name = "image" + str(num)
    if variant is None:
        key = "{}-{}.tns".format(dims, pt)
    else:
        key = "{}-{}-{}.tns".format(name, pt, variant)
    return os.path.join(path, "image", "tensors", key)

# ImagePydataSparseTensorLoader is the same as RandomPydataSparseTensorLoader
# but for images loaded from memory and converted to sparse.COO tensors
class ImagePydataSparseTensorLoader:
    def __init__(self):
        self.loader = PydataSparseTensorLoader()
        self.img = dict()
        self.max = dict()
        self.shape = dict()

    def dense_image(self, num, pt, variant=None, path='no'):
        # Used for verification and baseline only.
        # Do not need to write to output file 
        if num not in self.img.keys():
            self.img[num] = load_image(path, num)
            self.max[num] = numpy.max(self.img[num])

        img = self.img[num]
        t = self.max[num]*pt
        bin_img = thresh(img, t)[0]
        self.shape[num] = bin_img.shape 
        return bin_img 

    def sparse_image(self, num, pt, variant=None, path='no'):
        key = construct_image_tensor_key(num, pt, variant)
        # If an image with these properties exists already, then load it.
        if os.path.exists(key):
            result = self.loader.load(key)
            self.shape[num] = result.shape
            return result
        else:
            # Otherwise, we must create load the image and preprocess it with the desired properties.
            # dump it to the output file, then return it. 
            bin_img = self.dense_image(num, pt, variant, path)
            result = sparse.COO.from_numpy(bin_img)
            dok = sparse.DOK(result)
            write_shape = bin_img.flat[-1] == 0
            TnsFileDumper().dump_dict_to_file(self.shape[num], dok.data, key, write_shape)
            return result

    # sparse_window and dense_window must be called after the image calls
    def sparse_window(self, num, variant=3):
        path = TENSOR_PATH
        key = "image"+str(num) + "-" + str(variant) + ".tns"
        key = os.path.join(path, "image", "tensors", key)

        shape = self.shape[num]

        if os.path.exists(key):
            return self.loader.load(key)
        else:
            result_np = self.dense_window(num)
            result = sparse.COO.from_numpy(result_np)
            dok = sparse.DOK(result)
            write_shape = result_np.flat[-1] == 0
            TnsFileDumper().dump_dict_to_file(shape, dok.data, key, write_shape)
            return result

    def dense_window(self, num):
        shape = self.shape[num]
        result_np = numpy.zeros(shape)
        m0 = int(shape[0] / 2)
        m1 = int(shape[1] / 2)
        dm0 = int(0.1*m0)
        dm1 = int(0.1*m1)
        result_np[m0+dm0:m0+3*dm0, m1+dm1:m1+3*dm1] = 1
        result_np[m0-3*dm0:m0-dm0, m1+dm1:m1+3*dm1] = 1
        result_np[m0-3*dm0:m0-dm0, m1-3*dm1:m1-dm1] = 1
        result_np[m0+dm0:m0+3*dm0, m1-3*dm1:m1-dm1] = 1
        return result_np 

# plot_image plots the given original, binned, xor, and sparse xor images
# for the numpy/image.py. Used for debugging only with the --plot flag
# def plot_image(img, img1, img2, xor_img, sparse_xor_img, t1, t2, window=None):
#     f, ax = plt.subplots(2, 3)
#     ax[0, 0].imshow(img1, 'gray')
#     ax[0, 0].title.set_text("Binned Image 1. t1 = " + str(t1))
# 
#     ax[0, 1].imshow(img2, 'gray')
#     ax[0, 1].title.set_text("Binned Image 2. t2 = " + str(t2))
# 
#     ax[1, 0].imshow(img, 'gray')
#     ax[1, 0].title.set_text("Saturdated Image")
# 
#     ax[1, 1].imshow(xor_img, 'gray')
#     ax[1, 1].title.set_text("XOR Image")
# 
#     ax[1, 2].imshow(sparse_xor_img, 'gray')
#     ax[1, 2].title.set_text("Sparse XOR Image")
# 
#     if window is not None:
#         ax[0, 2].imshow(window, 'gray')
#         ax[0, 2].title.set_text("Fused Window Image")
# 
#     f.tight_layout()
#     plt.show()

