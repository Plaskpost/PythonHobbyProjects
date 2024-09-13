import math
import HyperbolicGrid
from Explorer import Explorer
import numpy as np
import MiniMap

neighbors = ['LU', 'UR', 'RD', 'DL', 'UL', 'RU', 'DR', 'LD']
d = ["D", "R", "U", "L"]

point = np.array([0., -0.5])
vector = np.array([-1, -1])

print(MiniMap.rotate_vector_towards_origin(point, vector))
