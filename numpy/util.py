import scipy.sparse
import scipy.io
import sparse
import os
import glob

# Get the path to the directory holding random tensors. Error out
# if this isn't set.
TENSOR_PATH = os.environ['TACO_TENSOR_PATH']

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

    def dump_dict_to_file(self, shape, data, path):
        # Sort the data so that the output is deterministic.
        sorted_data = sorted([list(coords) + [value] for coords, value in data.items()])
        with open(path, 'w+') as f:
            for line in sorted_data:
                coords = [str(elem + 1) for elem in line[:len(line) - 1]]
                strings = coords + [str(line[-1])]
                f.write(" ".join(strings))
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

# construct_random_tensor_key constructs a unique key that represents
# a random tensor parameterized by the chosen shape and sparsity.
# The key itself is formatted by the dimensions, followed by the
# sparsity. For example, a 250 by 250 tensor with sparsity 0.01
# would have a key of 250x250-0.01.tns.
def construct_random_tensor_key(shape, sparsity):
    path = TENSOR_PATH
    dims = "x".join([str(dim) for dim in shape])
    key = "{}-{}.tns".format(dims, sparsity)
    return os.path.join(path, "random", key)

# RandomPydataSparseTensorLoader should be used to generate
# random pydata.sparse tensors. It caches the loaded tensors
# in the file system so that TACO benchmarks using tensors
# with the same parameters can use the exact same tensors.
class RandomPydataSparseTensorLoader:
    def __init__(self):
        self.loader = PydataSparseTensorLoader()

    def random(self, shape, sparsity):
        key = construct_random_tensor_key(shape, sparsity)
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

    def random(self, shape, sparsity):
        assert(len(shape) == 2)
        key = construct_random_tensor_key(shape, sparsity)
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

    def getTensors(self):
        return self.tensors
    def getTensorNames(self):
        return [str(tensor) for tensor in self.getTensors()]

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
            if tensor.shape[-1] <= 0 or resultCoords[-1][i] % 2 == 0:
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
