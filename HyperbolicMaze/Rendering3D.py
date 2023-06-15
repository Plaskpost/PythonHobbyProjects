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
        self.wall_color = (255, 255, 255)
        self.edge_color = (0, 0, 0)
        self.floor_color = (100, 100, 100)
        self.background_color = (0, 0, 0)

        self.camera_y_angle = 0
        self.camera_shift = 0*self.camera_y_angle  # TODO: To be continued..



    def update(self):
        screen_width = self.SCREEN_SIZE[0]
        screen_height = self.SCREEN_SIZE[1]
        direction = self.explorer.rotation - self.camera_span[0] / 2.0
        edge_tolerance = 2 * self.camera_span[0] / screen_width

        # Draw background
        self.screen.fill(self.background_color)
        polygon_points = [(0, screen_height/2 + self.camera_shift), (0, screen_height),
                          (screen_width, screen_height), (screen_width, screen_height/2 + self.camera_shift)]
        pygame.draw.polygon(self.screen, self.floor_color, polygon_points)

        # Draw walls
        distance = -1
        distance_front = self.explorer.compute_distance(self.maze, direction)
        polygon_points[0], polygon_points[1] = self.get_vertical_points(column=0, distance=distance_front)
        for col in range(screen_width):
            direction += self.camera_span[0] / (screen_width - 1)
            distance_back = distance
            distance = distance_front
            distance_front = self.explorer.compute_distance(self.maze, direction)

            if distance_back != -1:
                edge_value = abs((distance_front - distance) - (distance - distance_back))
                if edge_value > edge_tolerance:
                    polygon_points[3], polygon_points[2] = self.get_vertical_points(col-1, distance_back)
                    pygame.draw.polygon(self.screen, self.wall_color, polygon_points)
                    edge_start, edge_end = self.get_vertical_points(col, distance)
                    pygame.draw.line(self.screen, self.edge_color, edge_start, edge_end, self.edge_line_thickness)
                    polygon_points[0], polygon_points[1] = self.get_vertical_points(col+1, distance_front)
        polygon_points[3], polygon_points[2] = self.get_vertical_points(screen_width-1, distance)
        pygame.draw.polygon(self.screen, self.wall_color, polygon_points)

        pygame.display.flip()

    def get_vertical_points(self, column, distance):
        line_length = np.round(self.vertical_scale/distance).astype(int)
        start = self.camera_shift + (self.SCREEN_SIZE[1] - line_length) // 2
        end = self.camera_shift + (self.SCREEN_SIZE[1] + line_length) // 2

        return (column, start), (column, end)

    def draw_overview_line(self, direction, distance, color):
        center = (self.SCREEN_SIZE[0] // 2, self.SCREEN_SIZE[1] // 2)
        end_point = (center[0] + distance * math.cos(math.radians(direction - self.explorer.rotation + 90)),
                     center[1] + distance * math.sin(math.radians(direction - self.explorer.rotation - 90)))
        pygame.draw.line(self.screen, color, center, end_point, 1)


