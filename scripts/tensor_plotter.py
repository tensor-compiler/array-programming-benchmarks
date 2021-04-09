import matplotlib.pyplot as plt
import numpy as np
from util import MinMaxPydataSparseTensorLoader
from mpl_toolkits.mplot3d import Axes3D

dims = 1
loader = MinMaxPydataSparseTensorLoader()
dims_list = [20] + [20] + [43]
matrix = loader.tensor(dims_list)
print(matrix)
matrix = matrix.todense()
print(matrix.shape)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
x,y,z = np.meshgrid(range(matrix.shape[0]), range(matrix.shape[1]), range(matrix.shape[2]))
ax.scatter(x,y,z, c=matrix.flat)

plt.show()
