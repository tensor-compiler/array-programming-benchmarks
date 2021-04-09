import scipy.sparse
import scipy.io
import sparse
import os
import glob
import numpy
import cv2

# NEEDS TO BE COMMENTED OUT FOR LANKA
# import matplotlib.pyplot as plt

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

# PydataSparseTensorLoader loads a sparse tensor from a file into
# a pydata.sparse tensor.
class PydataSparseTensorLoader:
    def __init__(self):
        self.loader = TnsFileLoader()
    
    def load(self, path):
        dims, coords, values = self.loader.load(path)
        return sparse.COO(coords, values, tuple(dims))

# construct_minmax_tensor_key constructs a unique key that represents
# an image tensor parameterized by the tensor order
# The key itself is formatted by the string 'minmax', followed by the
# tensor order. For example, a parameter of 3  
# would have a key of minmax-3.tns.
def construct_minmax_tensor_key(dims, variant=None):
    path = TENSOR_PATH
    name = "minmax"
    if variant is None:
        key = "{}-{}.tns".format(name, len(dims))
    else:
        key = "{}-{}-{}.tns".format(name,len(dims), variant)
    return os.path.join(path, name, key)

def generate_crds_helper(shape, level, crds):
    sampling = 0.1
    num = 3
    std = 2
    last_layer_sampling = 0.4

    if level == len(shape) - 1:
        return crds
    else:
        result = []
        d = shape[level]
        for c in crds:
            # Get number of locations 
            num_locs = int(sampling*d)
            # Get location uniformly of where to sample around
            locs = numpy.random.rand(num_locs)*d

            # sample around each location using a normal distribution around that value with a std of 2
            for loc in locs:
                points = std * numpy.random.randn(num) + loc
                points = points.astype('int')
                points = numpy.clip(points, 0, d - 1)
                for p in points:
                    result.append(c+[p])

        return generate_crds_helper(shape, level + 1, result)

# RandomPydataSparseTensorLoader should be used to generate
# random pydata.sparse tensors. It caches the loaded tensors
# in the file system so that TACO benchmarks using tensors
# with the same parameters can use the exact same tensors.
class MinMaxPydataSparseTensorLoader:
    def __init__(self):
        self.loader = PydataSparseTensorLoader()

    def tensor(self, shape, variant=None):
        key = construct_minmax_tensor_key(shape)
        # If a tensor with these properties exists already, then load it.
        if os.path.exists(key):
            return self.loader.load(key)
        else:
            # Otherwise, we must create a random tensor with the desired properties,
            # dump it to the output file, then return it.
            crds = self.generate_crds(shape)
            values = dict()
            for c in crds:
                ind_list = numpy.random.rand(2)*shape[-1]
                ind_list = ind_list.astype('int')
                start = numpy.min(ind_list)
                stop = numpy.max(ind_list)
                for i in range(start, stop):
                    temp = tuple(c[1:] + [i])
                    values[temp] = int(20*numpy.random.rand())

            dok = sparse.DOK(shape, values)
            TnsFileDumper().dump_dict_to_file(shape, dok.data, key)
            result = dok.asformat('coo')
            return result

                
    def generate_crds(self, shape):
        return generate_crds_helper(shape, 0, [[0]])
