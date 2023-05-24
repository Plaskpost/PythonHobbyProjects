import math

import pygame
import numpy as np


class Rending2D:

    def __init__(self, dynamic_maze, explorer, player_radius, tile_size):
        self.SCREEN_SIZE = np.array([1200, 800])
        self.SQUARE_SIZE = tile_size
        self.WALL_THICKNESS = 5  # *2
        self.SQUARE_COLOR = (255, 255, 255)
        self.BG_COLOR = (60, 60, 60)
        self.WALL_COLOR = (0, 0, 0)
        self.TEXT_COLOR = (0, 0, 0)
        self.DOT_COLOR = (255, 0, 0)
        self.DOT_SIZE = player_radius
        pygame.init()
        self.screen = pygame.display.set_mode(ndarray_to_tuple(self.SCREEN_SIZE))
        pygame.display.set_caption("Overview")
        self.screen.fill(self.BG_COLOR)
        self.font = pygame.font.SysFont('arial', 12)
        self.maze = dynamic_maze
        self.explorer = explorer
        self.update()

    def update(self):
        self.screen.fill(self.BG_COLOR)
        self.update_recursive(tile=self.explorer.pos_tile, prev_tile=None, screen_position=np.array([0, 0]))
        pygame.draw.circle(self.screen, self.DOT_COLOR, ndarray_to_tuple(self.SCREEN_SIZE//2), self.DOT_SIZE)
        pygame.display.flip()
        pygame.display.update()

    def update_recursive(self, tile, prev_tile, screen_position):
        if not self.maze.visibility_map[tile]:
            return

        # Define some useful parameters
        rotation_radians = math.radians(self.explorer.rotation)
        rotation_matrix = np.array([[math.cos(rotation_radians), -math.sin(rotation_radians)],
                                    [math.sin(rotation_radians), math.cos(rotation_radians)]])
        flip_y = np.array([1, -1])

        # Draw the tile
        square_center = -flip_y*self.explorer.pos + screen_position * self.SQUARE_SIZE
        square_center = np.dot(rotation_matrix, square_center)
        square_center = square_center + self.SCREEN_SIZE//2
        self.draw_square(ndarray_to_tuple(square_center), self.explorer.rotation)

        # Label the tile
        text = self.font.render(tile, True, self.TEXT_COLOR)
        self.screen.blit(text, (square_center[0], square_center[1]))

        shifts = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        for i in range(4):
            if self.maze.wall_map[tile][i] == -1:  # If wall draw wall.
                # Draw walls
                wall_x, wall_y, wall_w, wall_h = self.where_wall(i, square_center)
                # pygame.draw.rect(self.screen, self.WALL_COLOR, ((wall_x, wall_y), (wall_w, wall_h)))
            else:  # Else call drawing function for surrounding tiles.
                global_index = self.explorer.global_index_to(i)
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


def ndarray_to_tuple(ndarray):  # Fixed to 2D because we work in 2D.
    retval = (ndarray[0], ndarray[1])
    return retval
