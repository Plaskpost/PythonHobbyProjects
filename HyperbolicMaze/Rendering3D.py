import math
import time

import numpy as np
import pygame.display

import MiniMap
import config
from Rendering import Rendering


class Rendering3D(Rendering):
    # Index names
    r = 0
    phi = 1
    right = 0
    left = 1

    def __init__(self, dynamic_maze, explorer, include_mini_map=False):
        super().__init__("First person view", dynamic_maze, explorer)
        self.vertical_scale = 50000
        self.edge_line_thickness = 1
        self.wall_color = self.WHITE
        self.edge_color = self.BLACK
        self.floor_color = self.GRAY
        self.background_color = (0, 0, 0)

        self.camera_y_angle = 0
        self.camera_shift = 0*self.camera_y_angle  # To be continued..

        self.inner_corners, self.outer_corners = self.list_those_corners()  # [local_side_index][right=0, left=1]
        self.journey_steps = config.tile_size * np.array([[0, -1], [1, 0], [0, 1], [-1, 0]])

        self.debugging_in_2D = False
        self.wall_cut = True
        self.include_mini_map = include_mini_map

        self.mini_map = MiniMap.MiniMap(dynamic_maze, explorer, self.SCREEN_SIZE[0] / 4, 'bottom-right')

    def update(self):
        self.draw_background()
        self.draw_walls()
        self.write_debug_info()

        if self.include_mini_map:
            self.mini_map.update()


    def draw_background(self):
        screen_width = self.SCREEN_SIZE[0]
        screen_height = self.SCREEN_SIZE[1]
        self.screen.fill(self.background_color)
        polygon_points = [(0, screen_height / 2 + self.camera_shift), (0, screen_height),
                          (screen_width, screen_height), (screen_width, screen_height / 2 + self.camera_shift)]
        pygame.draw.polygon(self.screen, self.floor_color, polygon_points)

    def draw_walls(self):
        left_limit = config.camera_span[0] / 2.0
        right_limit = -config.camera_span[0] / 2.0
        wall_segment = np.array([[None, None], [None, None]])  # [[r_left, phi_left], [r_right, phi_right]]
        wall_segment[1] = self.to_polar(self.inner_corners[(self.explorer.get_facing()-2) % 4])
        self.wall_cut = True
        self.draw_walls_recursive(probe=self.explorer.__copy__(), prev=(self.explorer.get_facing()-2),
                                  left_angle_limit=left_limit, right_angle_limit=right_limit, wall_segment=wall_segment,
                                  journey=np.array([0, 0]))

    def draw_walls_recursive(self, probe, prev, left_angle_limit, right_angle_limit, wall_segment, journey):
        """
        :param probe: An Explorer object that properly traverses the maze in search for visible walls.
        :param prev: Local index to the probe's previously visited tile (initially set to tile behind payer based on facing).
        :param left_angle_limit: Local angle (facing = 0). We can't se any part of the current tile with relative angle
        higher than left_angle_limit.
        :param right_angle_limit: We can't se any part of the current tile with relative angle lower than right_angle_limit.
        :param wall_segment: [[distance_0, angle_0], [distance_1, angle_1]]. The variable that tracks one straight wall
        at a time, with 0 indexing the left edge of the wall and 1 indexing the right edge.
        :param journey: The probe's relative position to the player in cartesian coordinates.
        :return:
        """
        for i in range(1, 4):
            local_index = (i + prev) % 4
            global_index = probe.global_index_to(local_index)
            wall_here = self.maze.check_wall_with_placement(probe.pos_tile, global_index)

            # Find the far (ahead in rotation) inner corner.
            inner_left = self.to_polar(self.inner_corners[local_index] + journey)
            # Adjusting for an angle computation error that can happen in the player's tile
            if probe.pos_tile == self.explorer.pos_tile and i == 3:
                # The angle to this corner can't be positive for the wall to the left of the player (i = 3).
                if inner_left[self.phi] < 0.:
                    inner_left[self.phi] += 360.

            #if self.explorer.pos_tile == "" and self.explorer.global_index_to_previous_tile == 2:
            #   self.print_debug_info(probe.pos_tile, i, wall_segment, (left_angle_limit, right_angle_limit), inner_left)

            # Make a cut in the corner behind if it is an inner or outer corner.
            if not (wall_here ^ self.maze.check_wall_with_placement(probe.pos_tile, (global_index-1) % 4)) and not self.wall_cut:
                self.join_cut(wall_segment)

            # Extend wall if there is a wall here.
            # We'll do the same if we're outside view to the right to be in phase when back.
            if wall_here or right_angle_limit > inner_left[self.phi]:
                try:
                    right_met = self.extend(wall_segment, inner_left, left_angle_limit, right_angle_limit)
                except TypeError:
                    print("TypeError encountered!")
                    self.print_debug_info(probe.pos_tile, i, wall_segment, (left_angle_limit, right_angle_limit), inner_left)
                    raise RuntimeError
                if right_met:
                    return  # Break the loop if we met the left angle limit.
            else:  # If we have an opening
                extra_step = self.outer_corners[local_index][self.left] - self.inner_corners[local_index]
                inner_right = self.to_polar(self.inner_corners[(local_index - 1) % 4] + journey)
                outer_right = self.to_polar(self.outer_corners[local_index][self.right] + extra_step + journey)
                outer_left = self.to_polar(self.outer_corners[local_index][self.left] + extra_step + journey)
                # Adjusting for potential angle computation error
                if probe.pos_tile == self.explorer.pos_tile:
                    if i == 1:
                        # The angle to these corners can't be positive for the wall to the left of the player (i = 1).
                        if inner_right[self.phi] > 0.:
                            inner_right[self.phi] -= 360.
                        if outer_right[self.phi] > 0.:
                            outer_right[self.phi] -= 360.
                    if i == 3:
                        # The angle to this corner can't be positive for the wall to the left of the player (i = 3).
                        if outer_left[self.phi] < 0.:
                            outer_left[self.phi] += 360.

                right_options = np.array([inner_right[self.phi], outer_right[self.phi], right_angle_limit])
                sorted_right_indices = np.flip(np.argsort(right_options))

                left_options = np.array([inner_left[self.phi], outer_left[self.phi], left_angle_limit])
                sorted_left_indices = np.argsort(left_options)

                # Evaluate the little wall segment to the right.
                if sorted_right_indices[0] == 1:  # If outer_right is inmost this segment is visible.
                    if self.extend(wall_segment, outer_right, left_angle_limit, right_angle_limit):
                        return

                # Recursion call.
                new_probe = probe.__copy__()
                new_probe.transfer_tile(self.maze, local_index, global_index)
                self.draw_walls_recursive(probe=new_probe, prev=new_probe.local_index_to_previous,
                                          right_angle_limit=right_options[sorted_right_indices[0]],
                                          left_angle_limit=left_options[sorted_left_indices[0]],
                                          wall_segment=wall_segment, journey=(journey+self.journey_steps[local_index]))

                # Evaluate the little wall segment to the left
                if sorted_left_indices[0] == 1:  # If outer_left is inmost, this segment is visible.
                    if self.wall_cut:
                        if right_angle_limit > outer_left[self.phi]:
                            wall_segment[1] = self.find_on_line(inner_left, outer_left, right_angle_limit)
                        else:
                            wall_segment[1] = outer_left
                        self.wall_cut = False
                    if self.extend(wall_segment, inner_left, left_angle_limit, right_angle_limit):
                        return
                elif sorted_left_indices[0] == 0:  # If inner_left is inmost, wall_segment starts over here.
                    wall_segment[1] = inner_left
                    self.wall_cut = False
                elif sorted_left_indices[0] == 2:  # If the limit is inmost there is no point in continuing the loop.
                    return


# ----------------- FUNCTIONS FOR THE RECURSIVE ALGORITHM TO CALL -----------------------

    # Assumes straight wall segment without corners between itself and the new point.
    # Returns True if we met the left limit.
    def extend(self, wall_segment, new_point, left_limit, right_limit):
        if right_limit > new_point[self.phi]:  # Right limit exceeded.
            if not self.wall_cut:
                self.split_cut(wall_segment)
            wall_segment[1] = new_point
            return False
        elif new_point[self.phi] > left_limit:  # Left limit exceeded.
            if self.wall_cut:
                wall_segment[1] = self.find_on_line(wall_segment[1], new_point, right_limit)
                self.wall_cut = False
            wall_segment[0] = self.find_on_line(wall_segment[1], new_point, left_limit)
            self.split_cut(wall_segment)
            return True
        else:  # If new point within the span.
            if self.wall_cut:
                wall_segment[1] = self.find_on_line(wall_segment[1], new_point, right_limit)
                self.wall_cut = False
            wall_segment[0] = new_point
            #self.join_cut(wall_segment)  # TEMPORARY LINE
            return False

    def join_cut(self, wall_segment):
        self.draw_wall_segment(wall_segment)
        wall_segment[1] = wall_segment[0]

    def split_cut(self, wall_segment):
        self.draw_wall_segment(wall_segment)
        wall_segment[1] = [None, None]
        self.wall_cut = True

    def draw_wall_segment(self, wall_segment):
        if self.debugging_in_2D:
            self.draw_top_view_corner_dots(wall_segment)
            #pygame.display.flip()
            return

        polygon_points = [(0, 0), (0, 0), (0, 0), (0, 0)]
        polygon_points[0], polygon_points[1] = self.get_vertical_points(self.angle_to_column(wall_segment[0][self.phi]),
                                                                        wall_segment[0][self.r])
        polygon_points[3], polygon_points[2] = self.get_vertical_points(self.angle_to_column(wall_segment[1][self.phi]),
                                                                        wall_segment[1][self.r])
        pygame.draw.polygon(self.screen, self.edge_color, polygon_points, 3)
        pygame.draw.polygon(self.screen, self.wall_color, polygon_points)

    def draw_top_view_corner_dots(self, wall_segment):
        flip_y = np.array([1, -1])
        internal0 = wall_segment[0]#self.to_cartesian([wall_segment[0][0], wall_segment[0][1]])
        internal1 = wall_segment[1]#self.to_cartesian([wall_segment[1][0], wall_segment[1][1]])
        xy0 = flip_y * self.to_cartesian([wall_segment[0][0], wall_segment[0][1]-self.explorer.rotation+config.initial_rotation]) + self.SCREEN_SIZE/2
        xy1 = flip_y * self.to_cartesian([wall_segment[1][0], wall_segment[1][1]-self.explorer.rotation+config.initial_rotation]) + self.SCREEN_SIZE/2
        xy0, xy1 = tuple(xy0), tuple(xy1)
        pygame.draw.circle(self.screen, (0, 200, 100), xy0, 4)
        pygame.draw.circle(self.screen, (0, 200, 100), xy1, 4)
        formatted_strings0 = [format(f, '.1f') for f in internal0]
        label0 = "(" + ', '.join(formatted_strings0) + ")"
        formatted_strings1 = [format(f, '.1f') for f in internal1]
        label1 = "(" + ', '.join(formatted_strings1) + ")"
        text0 = self.font.render(label0, True, (0, 100, 50))
        text1 = self.font.render(label1, True, (0, 100, 50))
        self.screen.blit(text0, xy0)
        self.screen.blit(text1, xy1)

    def find_on_line(self, point1, point2, angle3):
        angle1 = point1[self.phi] + self.explorer.rotation
        angle2 = point2[self.phi] + self.explorer.rotation
        angle3 += self.explorer.rotation

        cond1 = self.is_in_quadrant(angle1, 2) and self.is_in_quadrant(angle2, 3) and self.is_in_quadrant(angle3, 2)
        cond2 = self.is_in_quadrant(angle1, 3) and self.is_in_quadrant(angle2, 2) and self.is_in_quadrant(angle3, 3)
        if cond1 or cond2:
            dummy = angle1
            angle1 = angle2
            angle2 = dummy
            a = point2[self.r]
            b = point1[self.r]
        else:
            a = point1[self.r]
            b = point2[self.r]
        gamma = math.radians(abs(angle1 - angle2))
        gamma_bc = math.radians(abs(angle3 - angle2))

        c = math.sqrt(a ** 2 + b ** 2 - 2 * a * b * math.cos(gamma))
        alpha = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
        distance = math.sin(alpha) * b / math.sin(math.pi - alpha - gamma_bc)
        return np.array([distance, (angle3-self.explorer.rotation)])


# -------------------------- SOME NUMERICAL OPERATIONS ----------------------------

    def to_polar(self, point):
        x, y = point
        player_x, player_y = self.explorer.pos
        r = np.hypot(x - player_x, y - player_y)
        phi = np.degrees(np.arctan2(y - player_y, x - player_x)) - self.explorer.rotation
        phi = (phi+180) % 360 - 180
        return np.array([r, phi])

    def to_cartesian(self, point):
        r, phi = point
        phi_rad = np.radians(phi + self.explorer.rotation)
        x = r * np.cos(phi_rad)
        y = r * np.sin(phi_rad)
        return np.array([x, y])

    def is_in_quadrant(self, angle, quadrant):
        lower_limit = 90 * (quadrant - 1)
        upper_limit = 90 * quadrant
        return lower_limit < (angle % 360) < upper_limit

    def angle_to_column(self, angle):
        left_edge = config.camera_span[0]//2
        right_edge = -config.camera_span[0]//2

        col = (left_edge - angle) * self.SCREEN_SIZE[0] / config.camera_span[0]
        if angle > left_edge or right_edge > angle:
            message = "Warning! Angle ", angle, "outside visible span [", left_edge, ", ", right_edge, "]!"
            #raise ValueError(message)
        return round(col)

    def get_vertical_points(self, column, distance):
        if column < 0 or column > self.SCREEN_SIZE[0]:
            raise ValueError("Column value ", column, " outside range!")

        line_height = self.vertical_height(distance)
        start = self.camera_shift + (self.SCREEN_SIZE[1] - line_height) // 2
        end = self.camera_shift + (self.SCREEN_SIZE[1] + line_height) // 2

        return (column, start), (column, end)

    def vertical_height(self, distance):
        return np.round(self.vertical_scale/max(distance, 0.001)).astype(int)


    def col_invert(self, col):
        return self.SCREEN_SIZE[0]-1 - col


# ----------------------------- ERROR MESSAGE & STUFF -------------------------------


    def print_debug_info(self, current_tile, wall_direction_index, wall_segment, limits, front_left_point):
        print("")
        print("Wall_segment: ", wall_segment)
        in_cartesian = []
        for point in wall_segment:
            if point[0] is not None:
                in_cartesian.append(self.to_cartesian(point))
        print("In cartesian: ", in_cartesian)
        print("Tile: ", current_tile)
        print("Wall direction index: ", wall_direction_index)
        print("Limits: ", limits)
        print("Front left point: ", front_left_point)
        print("Wall_cut: ", self.wall_cut)


# --------------- THIS GUY THAT WAS TOO MANY LINES TO BE IN __init__() --------------

    def list_those_corners(self):
        s = config.tile_size
        w = config.wall_thickness
        inner = np.array([[s - w, w], [s - w, s - w], [w, s - w], [w, w]])
        outer = np.array([[[w, 0], [s - w, 0]], [[s, w], [s, s - w]], [[s - w, s], [w, s]], [[0, s - w], [0, w]]])
        return inner, outer




