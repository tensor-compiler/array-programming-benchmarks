import scipy.sparse
import sparse
import os

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
                coords = [int(coord) - 1 for coord in data[:len(data) - 1]]
                # TODO (rohany): What if we want this to be an integer?
                value = float(data[-1])
                if first:
                    first = False
                    dims = [0] * len(coords)
                for i in range(len(coords)):
                    dims[i] = max(dims[i], coords[i] + 1)
                coordinates.append(coords)
                values.append(value)
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
        dok = scipy.sparse.dok_matrix(tuple(dims))
        for i in range(len(coords)):
            coord = coords[i]
            value = values[i]
            dok[tuple(coord)] = value
        # TODO (rohany): We could parametrize this by what kind of scipy.sparse
        #  matrix we want to make.
        if self.format == "csr":
            return scipy.sparse.csr_matrix(dok)
        elif self.format == "csc":
            return scipy.sparse.csc_matrix(dok)
        else:
            assert(False)

# PydataSparseTensorLoader loads a sparse tensor from a file into
# a pydata.sparse tensor.
class PydataSparseTensorLoader:
    def __init__(self):
        self.loader = TnsFileLoader()
    
    def load(self, path):
        dims, coords, values = self.loader.load(path)
        dok = sparse.DOK(tuple(dims))
        for i in range(len(coords)):
            coord = coords[i]
            value = values[i]
            dok[tuple(coord)] = value
        return sparse.COO(dok)

# construct_random_tensor_key constructs a unique key that represents
# a random tensor parameterized by the chosen shape and sparsity.
# The key itself is formatted by the dimensions, followed by the
# sparsity. For example, a 250 by 250 tensor with sparsity 0.01
# would have a key of 250x250-0.01.tns.
def construct_random_tensor_key(shape, sparsity):
    # Get the path to the directory holding random tensors. Error out
    # if this isn't set.
    path = os.environ['TACO_RANDOM_TENSOR_PATH']
    dims = "x".join([str(dim) for dim in shape])
    key = "{}-{}.tns".format(dims, sparsity)
    return os.path.join(path, key)

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
