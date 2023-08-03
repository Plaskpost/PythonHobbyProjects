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

        self.wall_cut = True

    def update(self):
        self.draw_background()
        self.draw_walls()
        self.write_debug_info()
        pygame.display.flip()

    def draw_background(self):
        screen_width = config.SCREEN_SIZE[0]
        screen_height = config.SCREEN_SIZE[1]
        self.screen.fill(self.background_color)
        polygon_points = [(0, screen_height / 2 + self.camera_shift), (0, screen_height),
                          (screen_width, screen_height), (screen_width, screen_height / 2 + self.camera_shift)]
        pygame.draw.polygon(self.screen, self.floor_color, polygon_points)

    def draw_walls(self):
        left_limit = self.explorer.rotation + config.camera_span[0] / 2.0
        right_limit = self.explorer.rotation - config.camera_span[0] / 2.0
        wall_segment = np.array([[-1.0, -1.0], [-1.0, -1.0]])  # [[r_left, phi_left], [r_right, phi_right]]
        wall_segment[1] = self.to_polar(self.inner_corners[0])
        self.draw_walls_recursive(self.explorer.__copy__(), left_limit, right_limit, wall_segment, np.array([0, 0]))

    def draw_walls_recursive(self, probe, left_angle_limit, right_angle_limit, wall_segment, journey):
        prev = probe.local_index_to_previous

        # Now the tree walls from right to left
        for i in range(1, 4):
            local_index = (i + prev) % 4
            global_index = probe.global_index_to(local_index)
            wall_here = self.maze.check_wall_with_placement(probe.pos_tile, global_index)

            # Find the far (ahead in rotation) inner corner.
            inner_left = self.to_polar(self.inner_corners[local_index] + journey)

            # Make a cut in the corner behind if it is an inner or outer corner.
            if not (wall_here ^ self.maze.check_wall_with_placement(probe.pos_tile, (global_index-1) % 4)) and not self.wall_cut:
                self.join_cut(wall_segment)

            # Extend wall if there is a wall here.
            # We'll do the same if we're outside view to the right to be in phase when back.
            if wall_here or inner_left[self.phi] < right_angle_limit:
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

                if sorted_right_indices[0] == 1:  # If outer_right is inmost this segment is visible.
                    self.extend(wall_segment, outer_right, left_angle_limit, right_angle_limit)

                # Recursion call.
                self.draw_walls_recursive(probe=probe.transfer_tile(self.maze, global_index, local_index),
                                          right_angle_limit=right_options[sorted_right_indices[0]],
                                          left_angle_limit=left_options[sorted_left_indices[0]],
                                          wall_segment=wall_segment, journey=(journey+self.journey_steps[local_index]))

                if sorted_left_indices[2] == 0:  # If inner_left is outermost, this segment is visible.
                    if self.wall_cut:
                        wall_segment[1] = outer_left
                        self.wall_cut = False
                    self.extend(wall_segment, inner_left, left_angle_limit, right_angle_limit)  # Enough because extend will catch the left limit.

    # Assumes straight wall segment without corners between itself and the new point.
    # Returns True if we met the left limit.
    def extend(self, wall_segment, new_point, left_limit, right_limit):
        if right_limit > new_point[self.phi]:
            if not self.wall_cut:
                self.split_cut(wall_segment)
            wall_segment[0] = new_point
            return False
        elif new_point[self.phi] > left_limit:
            wall_segment[0] = self.find_on_line(wall_segment[0], new_point, left_limit)
            self.split_cut(wall_segment)
            return True  # Only if left limit exceeded.
        else:  # If new point within the span.
            if self.wall_cut:
                wall_segment[1] = self.find_on_line(wall_segment[0], new_point, right_limit)
                self.wall_cut = False
            wall_segment[0] = new_point
            return False

    def join_cut(self, wall_segment):
        self.draw_wall_segment(wall_segment)
        wall_segment[1] = wall_segment[0]

    def split_cut(self, wall_segment):
        self.draw_wall_segment(wall_segment)
        wall_segment[1] = [None, None]
        self.wall_cut = True

    def draw_wall_segment(self, wall_segment):
        polygon_points = [(0, 0), (0, 0), (0, 0), (0, 0)]
        polygon_points[0], polygon_points[1] = self.get_vertical_points(self.angle_to_column(wall_segment[0][self.phi]),
                                                                        wall_segment[0][self.r])
        polygon_points[3], polygon_points[2] = self.get_vertical_points(self.angle_to_column(wall_segment[1][self.phi]),
                                                                        wall_segment[1][self.r])
        pygame.draw.polygon(self.screen, self.edge_color, polygon_points, 3)
        pygame.draw.polygon(self.screen, self.wall_color, polygon_points)

        # TEMPORARY LINE
        pygame.display.flip()

    def to_polar(self, point):  # [r, phi]
        polar = np.array([0.0, 0.0])
        polar[0] = np.hypot(point[0] - self.explorer.pos[0], point[1] - self.explorer.pos[1])
        polar[1] = np.degrees(np.arctan2(point[1] - self.explorer.pos[1], point[0] - self.explorer.pos[0]))
        return polar

    # TODO: Make find_on_line() failproof.
    def find_on_line(self, point1, point2, angle3):
        a = point1[self.r]
        b = point2[self.r]
        gamma = math.radians(abs((point1[self.phi] - point2[self.phi]) % 360))
        gamma_bc = math.radians(abs((angle3 - point2[self.phi]) % 360))
        c = math.sqrt(a ** 2 + b ** 2 - 2 * a * b * math.cos(gamma))
        alpha = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        return math.sin(alpha) * b / math.sin(math.pi - alpha - gamma_bc)

    def angle_to_column(self, angle):
        left_edge = self.explorer.rotation + config.camera_span[0]//2
        right_edge = self.explorer.rotation - config.camera_span[0]//2
        if right_edge > angle or angle > left_edge:
            raise ValueError("Angle outside visible span!")

        return (left_edge - angle) * config.SCREEN_SIZE[0] // config.camera_span[0]

    def list_those_corners(self):
        s = config.tile_size
        w = config.wall_thickness
        inner = np.array([[s - w, w], [s - w, s - w], [w, s - w], [w, w]])
        outer = np.array([[[w, 0], [s - w, 0]], [[s, w], [s, s - w]], [[s - w, s], [w, s]], [[0, s - w], [0, w]]])
        return inner, outer

    def get_vertical_points(self, column, distance):
        if column < 0 or column > config.SCREEN_SIZE[0]:
            raise ValueError("Column value ", column, " outside range!")
        line_length = np.round(self.vertical_scale/max(distance, 0.001)).astype(int)
        start = self.camera_shift + (config.SCREEN_SIZE[1] - line_length) // 2
        end = self.camera_shift + (config.SCREEN_SIZE[1] + line_length) // 2

        return (self.col_invert(column), start), (self.col_invert(column), end)

    def col_invert(self, col):
        return config.SCREEN_SIZE[0]-1 - col




