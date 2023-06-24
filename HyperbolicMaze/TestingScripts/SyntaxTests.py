import numpy as np

a = np.array([0.9, 0.4])
c = np.array([4, 2])
b = np.minimum(a, 1-a)

print(b)

result = np.argmin([a, 1-a], axis=0)

print(result)
print()