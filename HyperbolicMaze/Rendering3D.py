import math

import numpy as np
import matplotlib.pyplot as plt
import pygame.display

from Rendering2D import Rendering


class Rendering3D(Rendering):

    def __init__(self, dynamic_maze, explorer):
        super().__init__("First person view", dynamic_maze, explorer)
        self.camera_span = (90, 60)  # Degrees
        self.vertical_scale = 50000
        self.camera_y_angle = 0

        self.wall_color = (255, 255, 255)
        self.edge_color = (0, 0, 0)
        self.floor_color = (100, 100, 100)
        self.background_color = (0, 0, 0)

    def update(self):
        screen_width = self.SCREEN_SIZE[0]
        dist_to_wall = -np.ones(screen_width)
        direction = self.explorer.rotation - self.camera_span[0] / 2.0
        edge_tolerance = 2 * self.camera_span[0] / screen_width

        # Fill dist_to_wall with distances
        for col in range(screen_width):
            dist_to_wall[col] = self.explorer.compute_distance(self.maze, direction)
            direction += self.camera_span[0] / (screen_width - 1)

        # Expand to 2D array
        image_matrix = np.zeros((self.SCREEN_SIZE[1], screen_width, 3), dtype=np.uint8)
        for col in range(screen_width):
            color = self.wall_color
            if 1 <= col < screen_width-1:
                a = (dist_to_wall[col] - dist_to_wall[col - 1])
                b = (dist_to_wall[col - 1] - dist_to_wall[col - 2])
                c = abs(a - b)
                if c > edge_tolerance:
                    color = self.edge_color

            self.add_wall_line(image_matrix, dist_to_wall[col], col, color)

        pixel_array = pygame.surfarray.make_surface(image_matrix.swapaxes(0, 1))
        self.screen.blit(pixel_array, (0, 0))
        pygame.display.flip()

    def add_wall_line(self, matrix, distance, column, color):
        height = self.SCREEN_SIZE[1]
        flipped_col_index = self.SCREEN_SIZE[0]-1 - column
        line_length = np.round(self.vertical_scale/distance).astype(int)
        # camera_shift = TODO: To be continued..
        start = max(0, (height - line_length) // 2)
        end = min(height-1, (height + line_length) // 2)

        matrix[start:end, flipped_col_index] = color
        if end < height-1:
            matrix[(end+1):, flipped_col_index] = self.floor_color

    def draw_overview_line(self, direction, distance, color):
        center = (self.SCREEN_SIZE[0] // 2, self.SCREEN_SIZE[1] // 2)
        end_point = (center[0] + distance * math.cos(math.radians(direction - self.explorer.rotation + 90)),
                     center[1] + distance * math.sin(math.radians(direction - self.explorer.rotation - 90)))
        pygame.draw.line(self.screen, color, center, end_point, 1)


