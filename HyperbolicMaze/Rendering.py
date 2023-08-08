import pygame
import config
import numpy as np
from abc import ABC, abstractmethod


class Rendering(ABC):

    def __init__(self, caption, dynamic_maze, explorer):
        pygame.init()
        self.screen = pygame.display.set_mode(config.SCREEN_SIZE)
        self.font = pygame.font.SysFont('arial', config.TEXT_SIZE)
        pygame.display.set_caption(caption)
        self.maze = dynamic_maze
        self.explorer = explorer

        self.SCREEN_SIZE = np.array([config.SCREEN_SIZE[0], config.SCREEN_SIZE[1]])

        self.drawn_wall_segments = []

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def print_debug_info(self, current_tile, wall_direction_index, wall_segment, limits, front_left_point):
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
            self.screen.blit(text, (10, 10 + i*(config.TEXT_SIZE+10)))
