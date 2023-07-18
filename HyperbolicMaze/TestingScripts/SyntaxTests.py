import numpy as np
import matplotlib.pyplot as plt

a = np.array([1, 2])
[x, y] = a

values = np.random.binomial(5, 0.5, 1000)

gauss_values = np.round(0.5 * np.random.randn(200) + 2.2).astype(int)

print(gauss_values)