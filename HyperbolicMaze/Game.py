import math
import pygame
from DynamicMaze import DynamicMaze
from Rending2D import Rending2D
import numpy as np

# Constants
DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

movement_speed = 1.0
rotation_speed = 1
tile_size = 80  # Make independent of SQUARE_SIZE later?
player_radius = 10


class Explorer:
    def __init__(self):
        self.pos = np.array([tile_size / 2.0, tile_size / 2.0])
        self.pos_tile = ""
        self.index_to_previous_tile = 0  # Pointing to 'D' initially.
        self.rotation = 90  # Initially UP

    def transfer_tile(self, index_to_new, direction):
        # Change active tile
        new_tile = maze.adjacency_map[self.pos_tile][index_to_new]
        self.index_to_previous_tile = maze.adjacency_map[new_tile].index(self.pos_tile)
        self.pos_tile = new_tile

        # Change local coordinates
        [x, y] = self.pos
        if direction == "DOWN":
            x = tile_size - x
            y = -y
            self.rotate(True, 180)
        elif direction == "RIGHT":
            old_x = x
            x = tile_size -  y
            y = old_x
            self.rotate(left=False, amount=90)
        elif direction == "UP":
            y = y - tile_size
        elif direction == "LEFT":
            old_x = x
            x = y
            y = x
            self.rotate(left=True, amount=90)
        else:
            raise ValueError("Invalid direction!")
        self.pos = np.array([x, y])

        maze.update_visibility(self.pos_tile)

    def move(self, forward):
        heading = 1 if forward else -1
        v = heading * movement_speed * np.array([math.cos(math.radians(self.rotation)),
                                                 math.sin(math.radians(self.rotation))])
        self.pos += v
        near_edge = [(self.pos[1] < player_radius), (self.pos[0] >= tile_size - player_radius),
                     (self.pos[1] >= tile_size - player_radius), (self.pos[0] < player_radius)]
        across_edge = [(self.pos[1] < 0), (self.pos[0] >= tile_size), (self.pos[1] >= tile_size), (self.pos[0] < 0)]
        directions = ["DOWN", "RIGHT", "UP", "LEFT"]
        for i in range(4):  # i in {DOWN, RIGHT, UP, LEFT}
            index_to_tile_ahead = (i + self.index_to_previous_tile) % 4
            x_or_y = (1+i) % 2
            wall_ahead = maze.wall_map[self.pos_tile][index_to_tile_ahead] == -1
            if near_edge[i] and wall_ahead:
                self.pos[x_or_y] -= v[x_or_y]  # Move back.
            elif across_edge[i]:
                self.transfer_tile(index_to_tile_ahead, directions[i])

    def rotate(self, left, amount):
        if left:
            self.rotation += amount
            if self.rotation >= 360:
                self.rotation -= 360
        else:  # Right
            self.rotation -= amount
            if self.rotation < 0:
                self.rotation += 360


if __name__ == '__main__':

    explorer = Explorer()
    maze = DynamicMaze(explorer.pos_tile)
    renderer = Rending2D(maze, explorer.pos_tile, explorer.pos, player_radius)
    maze.update_visibility(explorer.pos_tile)
    renderer.update(explorer.pos_tile, explorer.pos)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # I think this section can look better.
        keys = pygame.key.get_pressed()
        arrow_keys = [pygame.K_DOWN, pygame.K_RIGHT, pygame.K_UP,
                      pygame.K_LEFT]  # This way i should match the direction.

        if keys[pygame.K_UP]:
            explorer.move(forward=True)
        if keys[pygame.K_DOWN]:
            explorer.move(forward=False)
        if keys[pygame.K_LEFT]:
            explorer.rotate(left=True, amount=rotation_speed)
        if keys[pygame.K_RIGHT]:
            explorer.rotate(left=False, amount=rotation_speed)

        if keys:
            renderer.update(explorer.pos_tile, explorer.pos)

    # Quit Pygame
    pygame.quit()
