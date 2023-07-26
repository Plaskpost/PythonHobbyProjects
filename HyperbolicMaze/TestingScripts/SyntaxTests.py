import numpy as np

journey = np.array([1, 2])
b = journey[:, np.newaxis]
c = np.hypot(3,4)
print(c)

class A:

    def __init__(self, v):
        self.v = v

    def set(self, nv):
        self.v = nv


def print_this(obj):
    print(obj.v)


a = A(3)
print_this(a.set(5))

