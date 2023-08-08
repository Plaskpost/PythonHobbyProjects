import math
import time
import pygame
import numpy as np

import Explorer
import config
from Rendering import Rendering
from Rendering3D import Rendering3D


class Rendering2D(Rendering):

    def __init__(self, dynamic_maze, explorer):
        super().__init__("Overview", dynamic_maze, explorer)
        self.NUM_RAYS = 3
        self.SQUARE_SIZE = config.tile_size
        self.WALL_THICKNESS = 5  # *2
        self.SQUARE_COLOR = (255, 255, 255)
        self.BG_COLOR = (60, 60, 60)
        self.WALL_COLOR = (0, 0, 0)
        self.TEXT_COLOR = (0, 0, 0)
        self.DOT_COLOR = (255, 0, 0)

        self.DOT_SIZE = config.player_radius
        self.screen.fill(self.BG_COLOR)

        self.engine_3D = Rendering3D(dynamic_maze, explorer)
        self.engine_3D.debugging_in_2D = True
        self.engine_3D.screen = self.screen

        self.drawn_tiles = set()
        self.update()

    def update(self):
        self.screen.fill(self.BG_COLOR)
        # self.maze.update_visibility(self.explorer.pos_tile)
        self.drawn_tiles = set()
        # self.update_recursive(tile=self.explorer.pos_tile, prev_tile=None, screen_position=np.array([0, 0]))
        self.draw_view_field()
        pygame.draw.circle(self.screen, self.DOT_COLOR, tuple(self.SCREEN_SIZE//2), self.DOT_SIZE)
        self.draw_spotted_corners()
        self.write_debug_info()
        pygame.display.flip()
        pygame.display.update()
        time.sleep(0.01)

    def update_recursive(self, tile, prev_tile, screen_position):
        if tile not in self.maze.visible_tiles or tile in self.drawn_tiles:
            return

        # Define some useful parameters
        rotation_radians = math.radians(self.explorer.rotation - config.initial_rotation)
        rotation_degrees = self.explorer.rotation - config.initial_rotation
        rotation_matrix = np.array([[math.cos(rotation_radians), -math.sin(rotation_radians)],
                                    [math.sin(rotation_radians), math.cos(rotation_radians)]])
        flip_y = np.array([1, -1])

        # Draw the tile
        square_center = -flip_y*self.explorer.pos + screen_position * self.SQUARE_SIZE + flip_y*self.SQUARE_SIZE//2
        square_center = np.dot(rotation_matrix, square_center)
        square_center = square_center + self.SCREEN_SIZE//2
        self.draw_square(tuple(square_center), rotation_degrees, (self.SQUARE_SIZE, self.SQUARE_SIZE), self.SQUARE_COLOR)

        # Label the tile
        text = self.font.render(tile, True, self.TEXT_COLOR)
        self.screen.blit(text, (square_center[0], square_center[1]))

        self.drawn_tiles.add(tile)

        shifts = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        for i in range(4):
            if self.maze.check_wall_with_placement(tile, i):
                # Draw walls
                wall_x, wall_y, wall_w, wall_h = self.where_wall(i, square_center)
                wall_center = (wall_x + wall_w//2 - self.SQUARE_SIZE//2, wall_y + wall_h//2 - self.SQUARE_SIZE//2)
                # self.draw_square(wall_center, rotation_degrees, (wall_w, wall_h), self.WALL_COLOR)
            else:  # Else call drawing function for surrounding tiles.
                global_index = self.explorer.global_index_to(i)  # To be replaced with the probe strategy.
                neighbor = self.maze.adjacency_map[tile][global_index]
                if neighbor == prev_tile:
                    continue
                self.update_recursive(tile=neighbor, prev_tile=tile, screen_position=(screen_position + shifts[i]))

    def draw_spotted_corners(self):
        self.engine_3D.draw_walls()

    def where_wall(self, i, pos):  # Completely incorrect nowdays?
        x, y, w, h = None, None, None, None
        if i == 0:
            x = pos[0] - self.WALL_THICKNESS
            y = pos[1] + self.SQUARE_SIZE - self.WALL_THICKNESS
            w = self.SQUARE_SIZE + 2*self.WALL_THICKNESS
            h = self.WALL_THICKNESS
        elif i == 1:
            x = pos[0] + self.SQUARE_SIZE - self.WALL_THICKNESS
            y = pos[1] - self.WALL_THICKNESS
            w = self.WALL_THICKNESS
            h = self.SQUARE_SIZE + 2*self.WALL_THICKNESS
        elif i == 2:
            x = pos[0] - self.WALL_THICKNESS
            y = pos[1]
            w = self.SQUARE_SIZE + 2*self.WALL_THICKNESS
            h = self.WALL_THICKNESS
        elif i == 3:
            x = pos[0]
            y = pos[1] - self.WALL_THICKNESS
            w = self.WALL_THICKNESS
            h = self.SQUARE_SIZE + 2*self.WALL_THICKNESS
        return x, y, w, h

    def draw_square(self, center, rotation, size, color):
        rect = pygame.Surface(size)
        rect.fill(color)
        square = pygame.Surface((size[0], size[1]), pygame.SRCALPHA)
        pygame.draw.rect(square, color, (0, 0, size[0], size[1]))
        rot_image = pygame.transform.rotate(square, 360-rotation)
        rot_rect = rot_image.get_rect(center=center)
        self.screen.blit(rot_image, rot_rect)

    def draw_all_squares(self):
        self.update_recursive(tile=self.explorer.pos_tile, prev_tile=None, screen_position=np.array([0, 0]))

    def draw_view_field(self):
        left_limit = self.explorer.rotation + config.camera_span[0]/2
        right_limit = self.explorer.rotation - config.camera_span[0]/2
        ray_directions = np.linspace(left_limit, right_limit, self.NUM_RAYS)
        ray_distances = np.empty_like(ray_directions)
        for i in range(len(ray_directions)):
            ray = Explorer.Ray(self.explorer)
            ray_distances[i] = ray.shoot(self.maze, ray_directions[i], True)
        self.draw_all_squares()
        for i in range(len(ray_directions)):
            self.draw_overview_line(ray_directions[i], ray_distances[i])

    def draw_overview_line(self, direction, distance):
        center = (config.SCREEN_SIZE[0] // 2, config.SCREEN_SIZE[1] // 2)
        end_point = (center[0] + distance * math.cos(math.radians(direction - self.explorer.rotation + 90)),
                     center[1] + distance * math.sin(math.radians(direction - self.explorer.rotation - 90)))
        pygame.draw.line(self.screen, (100, 100, 255), center, end_point, 1)

    def print_debug_info(self, current_tile, wall_direction_index, wall_segment, limits, front_left_point):
        self.engine_3D.print_debug_info(current_tile, wall_direction_index, wall_segment, limits, front_left_point)
