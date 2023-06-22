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
        self.edge_line_thickness = 2
        self.wall_thickness = 5  # *2
        self.wall_color = (255, 255, 255)
        self.edge_color = (0, 0, 0)
        self.floor_color = (100, 100, 100)
        self.background_color = (0, 0, 0)

        self.camera_y_angle = 0
        self.camera_shift = 0*self.camera_y_angle  # TODO: To be continued..

    def update(self):
        self.draw_background()
        self.draw_walls()
        self.write_debug_info()
        pygame.display.flip()

    def draw_background(self):
        screen_width = self.SCREEN_SIZE[0]
        screen_height = self.SCREEN_SIZE[1]
        self.screen.fill(self.background_color)
        polygon_points = [(0, screen_height / 2 + self.camera_shift), (0, screen_height),
                          (screen_width, screen_height), (screen_width, screen_height / 2 + self.camera_shift)]
        pygame.draw.polygon(self.screen, self.floor_color, polygon_points)

    def draw_walls(self):
        screen_width = self.SCREEN_SIZE[0]
        direction = self.explorer.rotation - self.camera_span[0] / 2.0
        edge_tolerance = 1.6 * self.camera_span[0] / screen_width
        polygon_points = [(0, 0), (0, 0), (0, 0), (0, 0)]

        distance = -1
        distance_front = self.explorer.compute_distance(self.maze, direction, False)
        polygon_points[0], polygon_points[1] = self.get_vertical_points(column=0, distance=distance_front)
        for col in range(screen_width):
            direction += self.camera_span[0] / (screen_width - 1)
            distance_back = distance
            distance = distance_front
            distance_front = self.explorer.compute_distance(self.maze, direction, False)

            if distance_back != -1:
                edge_value = abs((distance_front - distance) - (distance - distance_back))
                if edge_value > edge_tolerance:
                    polygon_points[3], polygon_points[2] = self.get_vertical_points(col - 1, distance_back)
                    pygame.draw.polygon(self.screen, self.edge_color, polygon_points, 3)
                    pygame.draw.polygon(self.screen, self.wall_color, polygon_points)
                    polygon_points[0], polygon_points[1] = self.get_vertical_points(col + 1, distance_front)
        polygon_points[3], polygon_points[2] = self.get_vertical_points(screen_width - 1, distance)
        pygame.draw.polygon(self.screen, self.edge_color, polygon_points, 2)
        pygame.draw.polygon(self.screen, self.wall_color, polygon_points)

    def get_vertical_points(self, column, distance):
        line_length = np.round(self.vertical_scale/distance).astype(int)
        start = self.camera_shift + (self.SCREEN_SIZE[1] - line_length) // 2
        end = self.camera_shift + (self.SCREEN_SIZE[1] + line_length) // 2

        return (self.col_invert(column), start), (self.col_invert(column), end)

    def col_invert(self, col):
        return self.SCREEN_SIZE[0]-1 - col




