import numpy as np
from scipy.signal import convolve2d

class GameOfLife:

    def __init__(self, N):
        self.board = np.zeros((N, N))
        self.kernel = np.ones((3, 3))
        self.kernel[1][1] = 0

    def tick(self):
        live_neighbors = convolve2d(self.board, self.kernel, mode='same', boundary='wrap')
        alive_condition = np.logical_or(live_neighbors == 3, np.logical_and(self.board == 1, live_neighbors == 2))
        self.board = np.where(alive_condition, 1, 0)
