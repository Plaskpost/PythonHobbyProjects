import numpy as np
import pygame

from Rendering2D import Rendering


class Rendering3D(Rendering):

    def __init__(self, dynamic_maze, explorer):
        super().__init__("First person view", dynamic_maze, explorer)
        self.camera_span = (100, 80)  # Degrees
        self.camera_facing = (self.explorer.rotation, 0)  # Also degrees

    def update(self):
        screen_width = self.SCREEN_SIZE[0]
        dist_to_wall = -np.ones(screen_width)
        direction = self.camera_facing[0] - self.camera_span[0] / 2.0
        for col in range(screen_width):
            dist_to_wall[col] = self.compute_distance(self.explorer.pos, direction)
            direction += self.camera_span[0] / (screen_width-1)
        a = 0

            
    def compute_distance(self, position, direction):
        return direction
