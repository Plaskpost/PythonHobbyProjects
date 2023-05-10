import numpy as np
import math


class Explorer:
    def __init__(self, movement_speed, rotation_speed, tile_size, player_radius):
        # Constants
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed
        self.tile_size = tile_size
        self.player_radius = player_radius

        # Variables
        self.pos = np.array([tile_size / 2.0, tile_size / 2.0])
        self.pos_tile = ""
        self.index_to_previous_tile = 0  # Pointing to 'D' initially.
        self.rotation = 90  # Initially UP.

    def transfer_tile(self, maze, index_to_new, direction):
        # Change active tile
        new_tile = maze.adjacency_map[self.pos_tile][index_to_new]
        self.index_to_previous_tile = maze.adjacency_map[new_tile].index(self.pos_tile)
        self.pos_tile = new_tile

        # Change local coordinates
        [x, y] = self.pos
        if direction == "DOWN":
            x = self.tile_size - x
            y = -y
            self.rotate(True, 180)
        elif direction == "RIGHT":
            old_x = x
            x = self.tile_size - y
            y = old_x
            self.rotate(left=False, amount=90)
        elif direction == "UP":
            y = y - self.tile_size
        elif direction == "LEFT":
            old_x = x
            x = y
            y = old_x
            self.rotate(left=True, amount=90)
        else:
            raise ValueError("Invalid direction!")
        self.pos = np.array([x, y])

        maze.update_visibility(self.pos_tile)

    def move(self, maze, forward):
        heading = 1 if forward else -1
        v = heading * self.movement_speed * np.array([math.cos(math.radians(self.rotation)),
                                                      math.sin(math.radians(self.rotation))])
        self.pos += v
        near_edge = [(self.pos[1] < self.player_radius), (self.pos[0] >= self.tile_size - self.player_radius),
                     (self.pos[1] >= self.tile_size - self.player_radius), (self.pos[0] < self.player_radius)]
        across_edge = [(self.pos[1] < 0), (self.pos[0] >= self.tile_size),
                       (self.pos[1] >= self.tile_size), (self.pos[0] < 0)]
        directions = ["DOWN", "RIGHT", "UP", "LEFT"]
        for i in range(4):  # i in {DOWN, RIGHT, UP, LEFT}
            index_to_tile_ahead = (i + self.index_to_previous_tile) % 4
            x_or_y = (1 + i) % 2
            wall_ahead = maze.wall_map[self.pos_tile][index_to_tile_ahead] == -1
            if near_edge[i] and wall_ahead:
                self.pos[x_or_y] -= v[x_or_y]  # Move back.
            elif across_edge[i]:
                self.transfer_tile(maze, index_to_tile_ahead, directions[i])

    def rotate(self, left, amount):
        if left:
            self.rotation += amount
            if self.rotation >= 360:
                self.rotation -= 360
        else:  # Right
            self.rotation -= amount
            if self.rotation < 0:
                self.rotation += 360
