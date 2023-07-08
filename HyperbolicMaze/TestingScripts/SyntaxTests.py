import numpy as np

a = np.array([[1, 2], [3, 4]])
b = np.array([10, 10])
a[1] = a[0]
a[0][0] = 0

print(a)
