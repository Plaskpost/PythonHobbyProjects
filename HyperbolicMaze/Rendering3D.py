import math
import numpy as np
import matplotlib.pyplot as plt
import pygame.display
import config

from Rendering2D import Rendering
from Explorer import Ray


class Rendering3D(Rendering):
    r = 0
    phi = 1
    right = 0
    left = 1

    def __init__(self, dynamic_maze, explorer):
        super().__init__("First person view", dynamic_maze, explorer)
        self.vertical_scale = 50000
        self.edge_line_thickness = 1
        self.wall_color = (255, 255, 255)
        self.edge_color = (0, 0, 0)
        self.floor_color = (100, 100, 100)
        self.background_color = (0, 0, 0)

        self.camera_y_angle = 0
        self.camera_shift = 0*self.camera_y_angle  # To be continued..

        self.inner_corners, self.outer_corners = self.list_those_corners()  # [local_side_index][right=0, left=1]
        self.journey_steps = config.tile_size * np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

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
        left = self.explorer.rotation - self.camera_span[0] / 2.0
        right = self.explorer.rotation + self.camera_span[0] / 2.0
        wall_segment = np.array([[-1, -1], [-1, -1]])  # [[r_left, phi_left], [r_right, phi_right]]
        self.draw_walls_recursive(self.explorer, left, right, wall_segment, np.array([0, 0]))

    def draw_walls_recursive(self, probe, left_angle_limit, right_angle_limit, wall_segment, journey):
        prev = probe.local_index_to_previous

        # Now the tree walls from right to left
        for i in range(1, 4):
            local_index = (i + prev) % 4
            global_index = probe.global_index_to(local_index)
            wall_here = self.maze.check_wall_with_placement(probe.pos_tile, global_index)

            # Find the far (ahead in rotation) inner corner and compute its angle.
            inner_left = self.to_polar(self.inner_corners[local_index] + journey)

            # Make a cut in the corner behind if it is an inner or outer corner.
            if not (wall_here ^ self.maze.check_wall_with_placement(probe.pos_tile, (global_index-1) % 4)):
                self.join_cut(wall_segment)

            # Extend wall if there is a wall here.
            if wall_here:
                if self.extend(wall_segment, inner_left, left_angle_limit, right_angle_limit):
                    break  # Break the loop if we met the left angle limit.
            else:  # If we have an opening
                extra_step = self.outer_corners[local_index][self.left] - self.inner_corners[local_index]
                inner_right = self.to_polar(self.inner_corners[(local_index - 1) % 4] + journey)
                outer_right = self.to_polar(self.outer_corners[local_index][self.right] + extra_step + journey)
                right_options = [inner_right[self.phi], outer_right[self.phi], right_angle_limit]
                sorted_right_indices = np.flip(np.argsort(right_options))

                outer_left = self.to_polar(self.outer_corners[local_index][self.left] + extra_step + journey)
                left_options = [inner_left[self.phi], outer_left[self.phi], left_angle_limit]
                sorted_left_indices = np.argsort(left_options)
                new_left_angle = left_options[sorted_left_indices[0]]

                if sorted_right_indices[0] == 1:  # If outer_right is inmost this segment is visible.
                    if wall_segment[1] == -1:
                        wall_segment[1] = right_options[sorted_right_indices[1]]  # TODO: I hate to say it but this only gives phi.
                    self.extend(wall_segment, outer_right, left_angle_limit, right_angle_limit)

                # Recursion call.
                self.draw_walls_recursive(probe=probe.transfer_tile(self.maze, global_index, local_index),
                                          right_angle_limit=right_options[sorted_right_indices[0]],
                                          left_angle_limit=left_options[sorted_left_indices[0]],
                                          wall_segment=wall_segment, journey=(journey+self.journey_steps[local_index]))

                if sorted_left_indices[2] == 0:  # If inner_left is outermost this segment is visible.
                    if wall_segment[1] == -1:
                        wall_segment[1] = left_options[sorted_left_indices[1]]
                    self.extend(wall_segment, inner_left, left_angle_limit, right_angle_limit)

    # Assumes straight wall segment without corners between itself and the new point.
    # Returns True if we met the left limit.
    def extend(self, wall_segment, new_point, left_limit, right_limit):
        # TODO: Check both left_limit and right limit and call split_cut() accordingly.
        if right_limit > new_point[self.phi]:
            self.split_cut(wall_segment)
            return False
        elif new_point[self.phi] > left_limit:
            wall_segment[0] = np.array([-1, -1])  # TODO: Figure out a way to compute the new coordinates.
            self.split_cut(wall_segment)
            return True  # Only if left limit exceeded.
        else:
            wall_segment[0] = new_point
            return False

    def join_cut(self, wall_segment):
        self.draw_wall_segment(wall_segment)
        wall_segment[1] = wall_segment[0]

    def split_cut(self, wall_segment):
        self.draw_wall_segment(wall_segment)
        wall_segment[1] = -1

    def draw_wall_segment(self, wall_segment):
        polygon_points = [(0, 0), (0, 0), (0, 0), (0, 0)]
        polygon_points[0], polygon_points[1] = self.get_vertical_points(self.angle_to_column(wall_segment[0][self.phi]),
                                                                        wall_segment[0][self.r])
        polygon_points[3], polygon_points[2] = self.get_vertical_points(self.angle_to_column(wall_segment[1][self.phi]),
                                                                        wall_segment[1][self.r])
        pygame.draw.polygon(self.screen, self.edge_color, polygon_points, 3)
        pygame.draw.polygon(self.screen, self.wall_color, polygon_points)

    def to_polar(self, point):  # [r, phi]
        polar = np.array([0, 0])
        polar[0] = np.hypot(point[0] - self.explorer.pos[0], point[1] - self.explorer.pos[1])
        polar[1] = np.degrees(np.arctan2(point[1] - self.explorer.pos[1], point[0] - self.explorer.pos[0]))
        return polar

    def angle_to_column(self, angle):
        left_edge = self.explorer.rotation + self.camera_span//2
        right_edge = self.explorer.rotation - self.camera_span//2
        if right_edge > angle or angle > left_edge:
            raise ValueError("Angle outside visible span!")

        return (left_edge - angle) * self.SCREEN_SIZE[0] // self.camera_span

    def list_those_corners(self):
        s = config.tile_size
        w = config.wall_thickness
        inner = np.array([[s - w, w], [s - w, s - w], [w, s - w], [w, w]])
        outer = np.array([[[w, 0], [s - w, 0]], [[s, w], [s, s - w]], [[s - w, s], [w, s]], [[0, s - w], [0, w]]])
        return inner, outer

    def old_draw_walls(self):
        screen_width = self.SCREEN_SIZE[0]
        direction = self.explorer.rotation - self.camera_span[0] / 2.0
        edge_tolerance = 1.6 * self.camera_span[0] / screen_width
        polygon_points = [(0, 0), (0, 0), (0, 0), (0, 0)]

        distance = -1
        distance_front = Ray.shoot(self.maze, self.explorer, direction)
        polygon_points[0], polygon_points[1] = self.get_vertical_points(column=0, distance=distance_front)
        for col in range(screen_width):
            direction += self.camera_span[0] / (screen_width - 1)
            distance_back = distance
            distance = distance_front
            distance_front = Ray.shoot(self.maze, self.explorer, direction)

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
        line_length = np.round(self.vertical_scale/max(distance, 0.001)).astype(int)
        start = self.camera_shift + (self.SCREEN_SIZE[1] - line_length) // 2
        end = self.camera_shift + (self.SCREEN_SIZE[1] + line_length) // 2

        return (self.col_invert(column), start), (self.col_invert(column), end)

    def col_invert(self, col):
        return self.SCREEN_SIZE[0]-1 - col




