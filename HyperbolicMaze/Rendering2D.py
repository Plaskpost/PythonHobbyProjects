import math
import time

import pygame
import numpy as np
from abc import ABC, abstractmethod


class Rendering(ABC):

    def __init__(self, caption, dynamic_maze, explorer):
        self.SCREEN_SIZE = np.array([800, 600])
        self.TEXT_SIZE = 12
        self.camera_span = (60, 60)  # Degrees
        pygame.init()
        self.screen = pygame.display.set_mode(tuple(self.SCREEN_SIZE))
        self.font = pygame.font.SysFont('arial', self.TEXT_SIZE)
        pygame.display.set_caption(caption)
        self.maze = dynamic_maze
        self.explorer = explorer

    @abstractmethod
    def update(self):
        pass

    def write_debug_info(self):
        directions = ["DOWN", "RIGHT", "UP", "LEFT"]
        walkable_directions = ""
        walkable_neighbors = ""
        for i in range(4):
            if self.maze.wall_map[self.explorer.pos_tile][i] == 1:
                walkable_directions += directions[i] + "  "
                walkable_neighbors += self.maze.adjacency_map[self.explorer.pos_tile][i] + "  "
        convert_to_string = np.vectorize(lambda x: "{:.{}f}".format(x, 1))
        string_pos = convert_to_string(self.explorer.pos)
        lines = ["Active tile: " + self.explorer.pos_tile,
                 "Walkable global directions: " + walkable_directions,
                 "Neighbors at those directions: " + walkable_neighbors,
                 "Position coordinates: (" + string_pos[0] + ", " + string_pos[1] + ")",
                 "Last local step: " + directions[self.explorer.opposite_of(self.explorer.local_index_to_previous)]]
        for i in range(len(lines)):
            text = self.font.render(lines[i], True, (150, 150, 255))
            self.screen.blit(text, (10, 10 + i*(self.TEXT_SIZE+10)))


class Rendering2D(Rendering):

    def __init__(self, dynamic_maze, explorer):
        super().__init__("Overview", dynamic_maze, explorer)
        self.NUM_RAYS = 2
        self.SQUARE_SIZE = self.explorer.tile_size
        self.WALL_THICKNESS = 5  # *2
        self.SQUARE_COLOR = (255, 255, 255)
        self.BG_COLOR = (60, 60, 60)
        self.WALL_COLOR = (0, 0, 0)
        self.TEXT_COLOR = (0, 0, 0)
        self.DOT_COLOR = (255, 0, 0)

        self.DOT_SIZE = self.explorer.player_radius
        self.screen.fill(self.BG_COLOR)

        self.initial_rotation = 90  # Not following changes in this at other locations then
        self.drawn_tiles = set()
        self.update()

    def update(self):
        self.screen.fill(self.BG_COLOR)
        # self.maze.update_visibility(self.explorer.pos_tile)
        self.drawn_tiles = set()
        # self.update_recursive(tile=self.explorer.pos_tile, prev_tile=None, screen_position=np.array([0, 0]))
        self.draw_view_field()
        pygame.draw.circle(self.screen, self.DOT_COLOR, tuple(self.SCREEN_SIZE//2), self.DOT_SIZE)
        self.write_debug_info()
        pygame.display.flip()
        pygame.display.update()
        time.sleep(0.01)

    def update_recursive(self, tile, prev_tile, screen_position):
        if tile not in self.maze.visible_tiles or tile in self.drawn_tiles:
            return

        # Define some useful parameters
        rotation_radians = math.radians(self.explorer.rotation - self.initial_rotation)
        rotation_degrees = self.explorer.rotation - self.initial_rotation
        rotation_matrix = np.array([[math.cos(rotation_radians), -math.sin(rotation_radians)],
                                    [math.sin(rotation_radians), math.cos(rotation_radians)]])
        flip_y = np.array([1, -1])

        # Draw the tile
        square_center = -flip_y*self.explorer.pos + screen_position * self.SQUARE_SIZE + flip_y*self.SQUARE_SIZE//2
        square_center = np.dot(rotation_matrix, square_center)
        square_center = square_center + self.SCREEN_SIZE//2
        self.draw_square(tuple(square_center), rotation_degrees)

        # Label the tile
        text = self.font.render(tile, True, self.TEXT_COLOR)
        self.screen.blit(text, (square_center[0], square_center[1]))

        self.drawn_tiles.add(tile)

        shifts = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        for i in range(4):
            if self.maze.wall_map[tile][i] == -1:  # If wall draw wall.
                # Draw walls
                wall_x, wall_y, wall_w, wall_h = self.where_wall(i, square_center)
                # pygame.draw.rect(self.screen, self.WALL_COLOR, ((wall_x, wall_y), (wall_w, wall_h)))
            else:  # Else call drawing function for surrounding tiles.
                global_index = self.explorer.global_index_to(i)  # This will be false occasionally since explorer.global_index_to has the player's orientation as reference.
                neighbor = self.maze.adjacency_map[tile][global_index]
                if neighbor == prev_tile:
                    continue
                self.update_recursive(tile=neighbor, prev_tile=tile, screen_position=(screen_position + shifts[i]))

    def where_wall(self, i, pos):  # Completely incorrect nowdays.
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

    def draw_square(self, center, rotation):
        size = (self.SQUARE_SIZE, self.SQUARE_SIZE)
        rect = pygame.Surface(size)
        rect.fill(self.SQUARE_COLOR)
        square = pygame.Surface((size[0], size[1]), pygame.SRCALPHA)
        pygame.draw.rect(square, self.SQUARE_COLOR, (0, 0, size[0], size[1]))
        rot_image = pygame.transform.rotate(square, 360-rotation)
        rot_rect = rot_image.get_rect(center=center)
        self.screen.blit(rot_image, rot_rect)

    def draw_all_squares(self):
        self.update_recursive(tile=self.explorer.pos_tile, prev_tile=None, screen_position=np.array([0, 0]))

    def draw_view_field(self):
        left_limit = self.explorer.rotation + self.camera_span[0]/2
        right_limit = self.explorer.rotation - self.camera_span[0]/2
        ray_directions = np.linspace(left_limit, right_limit, self.NUM_RAYS)
        ray_distances = np.empty_like(ray_directions)
        for i in range(len(ray_directions)):
            ray_distances[i] = self.explorer.compute_distance(self.maze, ray_directions[i], True)
        self.draw_all_squares()
        for i in range(len(ray_directions)):
            self.draw_overview_line(ray_directions[i], ray_distances[i])

    def draw_overview_line(self, direction, distance):
        center = (self.SCREEN_SIZE[0] // 2, self.SCREEN_SIZE[1] // 2)
        end_point = (center[0] + distance * math.cos(math.radians(direction - self.explorer.rotation + 90)),
                     center[1] + distance * math.sin(math.radians(direction - self.explorer.rotation - 90)))
        pygame.draw.line(self.screen, (100, 100, 255), center, end_point, 1)
