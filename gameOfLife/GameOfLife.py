import numpy as np


class GameOfLife:

    def __init__(self, N):
        self.board = np.zeros((N, N))
        self.kernel = np.ones((3, 3))

    def tick(self):
        pass