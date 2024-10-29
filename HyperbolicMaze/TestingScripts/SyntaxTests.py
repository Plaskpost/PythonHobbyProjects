import math
import HyperbolicGrid
from Explorer import Explorer
import numpy as np
#import MiniMap

neighbors = ['LU', 'UR', 'RD', 'DL', 'UL', 'RU', 'DR', 'LD']
d = ["D", "R", "U", "L"]

larger_matrix = np.array([[[i, j] for j in range(10)] for i in range(10)])

i, j = 0, 0
sub_matrix = larger_matrix[i:(i+2), j:(j+2)]
vector_version = sub_matrix.reshape((4, 2))

reference_point_mat = np.array([[[0, 0], [0, 1]], [[1, 0], [1, 1]]])  # Scaled and translated unit square
reference_points = reference_point_mat.reshape((4, 2))

target_points = [[2, 2], [2, 4], [4, 2], [4, 4]]    # Target frame
point = [1., 0.5]  # Point in the reference frame


angles = np.array([-120, 120])

print(angles % 306)


class A:

    def __init__(self):
        self.i = 0

    def operation_1(self):
        print("A 1")
        self.operation_2()

    def operation_2(self):
        print("A 2")


class B(A):

    def __init__(self):
        super().__init__()

    def operation_2(self):
        print("B 2")


b = B()
b.operation_1()