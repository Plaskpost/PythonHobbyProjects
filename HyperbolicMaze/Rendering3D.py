import math

import numpy as np
import matplotlib.pyplot as plt
import pygame.display

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
        self.wall_thickness = self.explorer.wall_thickness  # *2
        self.wall_color = (255, 255, 255)
        self.edge_color = (0, 0, 0)
        self.floor_color = (100, 100, 100)
        self.background_color = (0, 0, 0)

        self.camera_y_angle = 0
        self.camera_shift = 0*self.camera_y_angle  # To be continued..

        self.inner_corners, self.outer_corners = self.list_those_corners() # [local_side_index][right=0, left=1]

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
        wall_segment = np.array([[-1, -1], [-1, -1]])  # [[x_left, y_left], [x_right, y_right]]
        self.draw_walls_recursive(self.explorer, left, right, wall_segment, np.array([0, 0]))

    def draw_walls_recursive(self, probe, left_angle_limit, right_angle_limit, wall_segment, journey):  # TODO: Journey is journey[:, np.newaxis] as standard.
        prev = probe.local_index_to_previous

        # Start by the first small corner segment to the right.
        first_right_corner = self.to_polar(self.inner_corners[0+prev] + journey)
        if first_right_corner[self.phi] > right_angle_limit:  # If first small segment visible
            if self.extend(wall_segment, first_right_corner, left_angle_limit):
                return

        # A cut condition here? # TODO: It's very well time to have a good system for when to call cut()

        # Now the tree walls from right to left
        for i in range(1, 4):
            local_index = (i + prev) % 4
            global_index = probe.global_index_to(local_index)
            wall_here = self.maze.check_wall_with_placement(probe.pos_tile, global_index)

            # Find the far (ahead in rotation) inner corner and compute its angle.
            far_corner = self.to_polar(self.inner_corners[local_index] + journey)

            if wall_here:
                if self.extend(wall_segment, far_corner, left_angle_limit):
                    return
                # Note that if we find an actual inner corner, we need to call the cut function.
            else: # If we have an opening
                outer_left = self.to_polar(self.outer_corners[local_index][self.right] + journey)
                new_left = min(far_corner[self.phi], outer_left[self.phi], left_angle_limit)
                inner_right = self.to_polar(self.inner_corners[(local_index-1)%4] + journey)
                outer_right = self.to_polar(self.outer_corners[local_index][self.right] + journey)
                new_right = max(inner_right, outer_right, right_angle_limit)

            #       If corner segment to the right is visible: (should gra this information from the min/max above)
            #           If previous was a wall:
            #               Extend the saved wall_segment to include it.
            #           Else: If previous was an opening,
            #               Draw the wall_segment.
            #               Update it to represent the new small segment.
            #               Draw it on the screen and update right_limit.
            #       If left_limit still is > right_limit:
            #           Call a new recursion with the updated limits and coordinate_additions.
            #       If corner segment to the left is visible:
            #           Draw the small corner segment to the right of the far inner corner and update left_limit.
            #   Update the right edge of the dynamic wall_segment parameter to the value of the left edge.
            #   If the right edge is now larger than left_limit:
            #       Draw the stored wall segment and reset it
            #       Break the loop.



        # After for loop check the leftmost corner segment like we did with the right before the loop.

        pass

    def cut(self, wall_segment):  # TODO: This is completely wrong. wall_segment is in polar coordinates first of all.
        pygame.draw.polygon(self.screen, self.edge_color, wall_segment, 3)
        pygame.draw.polygon(self.screen, self.wall_color, wall_segment)
        wall_segment[0] = wall_segment[1]

    # Assumes straight wall segment without corners between itself and the new point.
    def extend(self, wall_segment, new_point, left_limit):
        if self.compute_angle(new_point) > left_limit:  # TODO: new_point is now in polar!
            wall_segment[0] = np.array([-1, -1])  # TODO: Figure out a way to compute the new coordinates.
            self.cut(wall_segment)
            return True
        else:
            wall_segment[0] = new_point
            return False

    def to_polar(self, points):  # [r, phi]
        polar = np.empty_like(points)
        for i in range(len(points)):
            polar[i][0] = np.hypot(points[0] - self.explorer.pos[0], points[1] - self.explorer.pos[1])
            polar[i][1] = self.compute_angle(points[i])
        return polar

    def compute_angle(self, point):
        return np.degrees(np.arctan2(point[1] - self.explorer.pos[1], point[0] - self.explorer.pos[0]))

    def list_those_corners(self):
        s = self.explorer.tile_size
        w = self.explorer.wall_thickness
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




