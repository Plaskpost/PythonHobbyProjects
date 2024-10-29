import pygame
import config
import numpy as np
from abc import ABC, abstractmethod

class Rendering(ABC):
    """
    Rendering Abstract Base Class

    This class serves as an abstract base for rendering a dynamic maze exploration environment using Pygame.
    The `Rendering` class manages the graphical display of a maze, including walls, player position,
    and debugging information, which assists in visualizing both the state of the maze and the explorer's movements.
    """

    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (100, 100, 100)
    RED = (255, 0, 0)
    DEBUG_BLUE = (150, 150, 255)

    def __init__(self, caption, dynamic_maze, explorer):
        pygame.init()
        self.screen = pygame.display.set_mode(config.screen_size)
        self.font = pygame.font.SysFont('arial', config.TEXT_SIZE)
        pygame.display.set_caption(caption)
        self.maze = dynamic_maze
        self.explorer = explorer
        self.SCREEN_SIZE = np.array([config.screen_size[0], config.screen_size[1]])


    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def print_debug_info(self, current_tile, wall_direction_index, wall_segment, limits, front_left_point):
        pass


    def make_debug_lines(self):
        directions = ["DOWN", "RIGHT", "UP", "LEFT"]
        walkable_directions = ""
        walkable_neighbors = ""
        for i in range(4):
            if self.maze.wall_map[self.explorer.pos_tile][i] == 1:
                walkable_directions += directions[i] + "  "
                walkable_neighbors += self.maze.adjacency_map[self.explorer.pos_tile][i] + "  "
        convert_to_string = np.vectorize(lambda x: "{:.{}f}".format(x, 1))
        string_pos = convert_to_string(self.explorer.pos)

        debug_lines = ["Active tile: " + self.explorer.pos_tile,
                       "Walkable global directions: " + walkable_directions,
                       "Neighbors at those directions: " + walkable_neighbors,
                       "Position coordinates: (" + string_pos[0] + ", " + string_pos[1] + ")",
                       "Last local step: " + directions[
                           self.explorer.opposite_of(self.explorer.local_index_to_previous)],
                       f"Player rotation: {self.explorer.rotation:.2f}"]

        return debug_lines

    def write_debug_info(self):
        lines = self.make_debug_lines()

        for i in range(len(lines)):
            text = self.font.render(lines[i], True, self.DEBUG_BLUE)
            self.screen.blit(text, (10, 10 + i*(config.TEXT_SIZE+10)))
