import numpy as np
import math


class Explorer:

    def __init__(self, movement_speed, rotation_speed, tile_size, player_radius):
        # Constants
        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed
        self.tile_size = tile_size
        self.player_radius = player_radius
        self.directions = ["DOWN", "RIGHT", "UP", "LEFT"]

        # Variables
        self.pos = np.array([tile_size / 2.0, tile_size / 2.0])
        self.pos_tile = ""
        self.index_to_previous_tile = 0  # Pointing to 'D' initially.
        self.rotation = 90  # Initially UP.
        self.local_direction_to_previous = "DOWN"

    def opposite_of(self, direction):
        return self.directions[(self.directions.index(direction) + 2) % 4]

    def global_index_to(self, local_index_to_next_tile):
        local_index_to_previous_tile = self.directions.index(self.local_direction_to_previous)
        # "The amount of clockwise steps from direction_to_previous to direction_to_next is added to index_to_last."
        return (((local_index_to_next_tile - local_index_to_previous_tile) % 4) + self.index_to_previous_tile) % 4

    def transfer_tile(self, maze, index_to_new, direction):
        # Change active tile
        new_tile = maze.adjacency_map[self.pos_tile][index_to_new]
        self.index_to_previous_tile = maze.adjacency_map[new_tile].index(self.pos_tile)
        self.pos_tile = new_tile
        self.local_direction_to_previous = self.opposite_of(direction)

        # Change local coordinates
        [x, y] = self.pos
        if direction == "DOWN":
            y += self.tile_size
        elif direction == "RIGHT":
            x -= self.tile_size
        elif direction == "UP":
            y -= self.tile_size
        elif direction == "LEFT":
            x += self.tile_size
        else:
            raise ValueError("Invalid direction!")
        self.pos = np.array([x, y])

    def move(self, maze, forward):
        heading = 1 if forward else -1
        v = heading * self.movement_speed * np.array([math.cos(math.radians(self.rotation)),
                                                      math.sin(math.radians(self.rotation))])
        self.pos += v
        near_edge = [(self.pos[1] < self.player_radius and v[1] < 0),
                     (self.pos[0] >= self.tile_size - self.player_radius and v[0] > 0),
                     (self.pos[1] >= self.tile_size - self.player_radius and v[1] > 0),
                     (self.pos[0] < self.player_radius and v[0] < 0)]
        across_edge = [(self.pos[1] < 0), (self.pos[0] >= self.tile_size),
                       (self.pos[1] >= self.tile_size), (self.pos[0] < 0)]
        for i in range(4):  # i in local {DOWN, RIGHT, UP, LEFT}
            index_to_tile_ahead = self.global_index_to(i)
            x_or_y = (1 + i) % 2
            wall_ahead = maze.wall_map[self.pos_tile][index_to_tile_ahead] == -1
            if near_edge[i] and wall_ahead:
                self.pos[x_or_y] -= v[x_or_y]  # Move back.
            elif across_edge[i]:
                self.transfer_tile(maze, index_to_tile_ahead, self.directions[i])
                #maze.update_visibility(self.pos_tile)  # 2D Strategy only!

    def rotate(self, left, amount):
        if left:
            self.rotation += amount
            if self.rotation >= 360:
                self.rotation -= 360
        else:  # Right
            self.rotation -= amount
            if self.rotation < 0:
                self.rotation += 360

    def compute_distance(self, maze, direction):
        # Indices
        x = 0
        y = 1

        pos = self.pos.copy()
        tile = self.pos_tile
        distance = 0
        while True:
            cos_r = math.cos(math.radians(direction))
            sin_r = math.sin(math.radians(direction))
            dx = self.tile_size - pos[x] if cos_r > 0 else pos[x]
            dy = self.tile_size - pos[y] if sin_r > 0 else pos[y]

            if cos_r == 0.0:
                dimension_hit = y
                d = dy
            elif sin_r == 0.0:
                dimension_hit = x
                d = dx
            else:
                d_options = np.abs(np.array([dx / cos_r, dy / sin_r]))
                dimension_hit = np.argmin(d_options)
                d = d_options[dimension_hit]
            distance += d  # Update total distance here

            # Which tile are we approaching and update pos for next iteration
            if dimension_hit == x:
                if cos_r > 0:
                    local_border_hit = 1  # RIGHT
                    pos[x] = 0
                else:
                    local_border_hit = 3  # LEFT
                    pos[x] = self.tile_size
                pos[y] += d*sin_r
            else:
                if sin_r > 0:
                    local_border_hit = 2  # UP
                    pos[y] = 0
                else:
                    local_border_hit = 0  # DOWN
                    pos[y] = self.tile_size
                pos[x] += d*cos_r

            global_border_hit = self.global_index_to(local_border_hit)
            if maze.wall_map[tile][global_border_hit] == 0:  # Update maze when a 0 is in view
                maze.place_wall_or_opening(tile, global_border_hit)
            if maze.wall_map[tile][global_border_hit] == -1:  # Break loop when wall is hit.
                break

            # Update pos_tile for next iteration
            tile = maze.adjacency_map[tile][global_border_hit]

        return distance
