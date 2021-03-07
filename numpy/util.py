import scipy.sparse
import sparse
import os

# CoordinateListFileLoader loads a file in coordinate list
# format into a list of the coordinates and a list of the values.
class CoordinateListFileLoader:
    def __init__(self):
        pass
    
    def load(self, path):
        dims = []
        entries = None
        coordinates = []
        values = []
        first = True
        with open(path, 'r') as f:
            for line in f:
                # Skip lines with %, as some downloaded files have these
                # at the header as comments.
                if line.startswith("%"):
                    continue
                data = line.split(' ')
                coords = [int(coord) for coord in data[:len(data) - 1]]
                # TODO (rohany): What if we want this to be an integer?
                value = float(data[-1])
                # If this is the first line being read, then the read
                # coordinates and values are actually the size of each
                # dimension and the number of non-zeros.
                if first:
                    dims = coords
                    entries = int(value)
                    first = False
                else:
                    coordinates.append(coords)
                    values.append(value)
        assert(len(coordinates) == entries)
        assert(len(values) == entries)
        return dims, coordinates, values

# CoordinateListFileDumper dumps a dictionary of coordinates to values
# into a coordinate list tensor file.
class CoordinateListFileDumper:
    def __init__(self):
        pass

    def dump_dict_to_file(self, shape, data, path):
        # Sort the data so that the output is deterministic.
        sorted_data = sorted([list(coords) + [value] for coords, value in data.items()])
        with open(path, 'w+') as f:
            # Write the metadata into the file.
            dims = list(shape) + [len(data)]
            f.write(" ".join([str(elem) for elem in dims]))
            f.write("\n")
            for line in sorted_data:
                strings = [str(elem) for elem in line]
                f.write(" ".join(strings))
                f.write("\n")

# ScipySparseTensorLoader loads a sparse tensor from a file into a
# scipy.sparse CSR matrix.
class ScipySparseTensorLoader:
    def __init__(self, format):
        self.loader = CoordinateListFileLoader()
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
        self.loader = CoordinateListFileLoader()
    
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
# would have a key of 250x250-0.01.tensor.
def construct_random_tensor_key(shape, sparsity):
    # Get the path to the directory holding random tensors. Error out
    # if this isn't set.
    path = os.environ['TACO_RANDOM_TENSOR_PATH']
    dims = "x".join([str(dim) for dim in shape])
    key = "{}-{}.tensor".format(dims, sparsity)
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
            CoordinateListFileDumper().dump_dict_to_file(shape, dok.data, key)
            return result

# RandomScipySparseTensorLoader is the same as RandomPydataSparseTensorLoader
# but for scipy.sparse tensors.
class RandomScipySparseTensorLoader:
    def __init__(self, format):
        self.loader = ScipySparseTensorLoader(format)

    def random(self, shape, sparsity):
        assert(len(shape) == 2)
        key = construct_random_tensor_key(shape, sparsity)
        # If a tensor with these properties exists already, then load it.
        if os.path.exists(key):
            return self.loader.load(key)
        else:
            # Otherwise, create and then dump a tensor.
            result = scipy.sparse.random(shape[0], shape[1], density=sparsity, format='csr')
            dok = scipy.sparse.dok_matrix(result)
            CoordinateListFileDumper().dump_dict_to_file(shape, dict(dok.items()), key)
            return result
