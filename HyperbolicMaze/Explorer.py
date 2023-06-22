import numpy as np
import math


class Explorer:
    DOWN = 0
    RIGHT = 1
    UP = 2
    LEFT = 3

    x = 0
    y = 1

    def __init__(self, tile_size, pos, pos_tile, global_prev, local_prev):
        self.pos = pos
        self.pos_tile = pos_tile
        self.tile_size = tile_size
        self.global_index_to_previous_tile = global_prev
        self.local_index_to_previous = local_prev

    def global_index_to(self, local_index_to_next_tile):
        # "The amount of clockwise steps from direction_to_previous to direction_to_next is added to index_to_last."
        return (((local_index_to_next_tile - self.local_index_to_previous) % 4) + self.global_index_to_previous_tile) % 4

    def transfer_tile(self, maze, global_index_to_new, local_index_to_new):
        # Change active tile
        new_tile = maze.adjacency_map[self.pos_tile][global_index_to_new]
        self.global_index_to_previous_tile = maze.adjacency_map[new_tile].index(self.pos_tile)
        self.pos_tile = new_tile
        self.local_index_to_previous = self.opposite_of(local_index_to_new)

        # Change local coordinates
        if local_index_to_new == self.DOWN:
            self.pos[self.y] += self.tile_size
        elif local_index_to_new == self.RIGHT:
            self.pos[self.x] -= self.tile_size
        elif local_index_to_new == self.UP:
            self.pos[self.y] -= self.tile_size
        elif local_index_to_new == self.LEFT:
            self.pos[self.x] += self.tile_size
        else:
            raise ValueError("Invalid direction!")

    def opposite_of(self, direction):
        return (direction + 2) % 4


class Player(Explorer):

    def __init__(self, movement_speed, rotation_speed, tile_size, player_radius):
        super().__init__(pos=np.array([tile_size / 2.0, tile_size / 2.0]), pos_tile="",
                         tile_size=tile_size, global_prev=0, local_prev=Explorer.DOWN)

        self.movement_speed = movement_speed
        self.rotation_speed = rotation_speed
        self.player_radius = player_radius

        self.rotation = 90  # Initially UP.

    def move(self, maze, flbr):
        rotation_matrix = np.array([[0, -1], [1, 0]])
        rotation_matrix = np.linalg.matrix_power(rotation_matrix, flbr)
        v = self.movement_speed * np.array([math.cos(math.radians(self.rotation)),
                                                      math.sin(math.radians(self.rotation))])
        v = np.dot(rotation_matrix, v)
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
                self.transfer_tile(maze, index_to_tile_ahead, i)

    def rotate(self, left, amount):
        if left:
            self.rotation += amount
            if self.rotation >= 360:
                self.rotation -= 360
        else:  # Right
            self.rotation -= amount
            if self.rotation < 0:
                self.rotation += 360

    def compute_distance(self, maze, direction, debugging_in_2D):
        # Indices
        x = self.x
        y = self.y

        ray = Explorer(self.tile_size, self.pos.copy(), self.pos_tile,
                       self.global_index_to_previous_tile, self.local_index_to_previous)
        distance = 0
        tile_path = []

        while True:
            cos_r = math.cos(math.radians(direction))
            sin_r = math.sin(math.radians(direction))
            dx = ray.tile_size - ray.pos[x] if cos_r > 0 else ray.pos[x]
            dy = ray.tile_size - ray.pos[y] if sin_r > 0 else ray.pos[y]

            # Start by checking if we've collided with the edge of a wall.
            wall_thickness = 5
            if ray.pos[x] < wall_thickness and \
                    maze.check_wall_with_placement(ray.pos_tile, ray.global_index_to(self.LEFT)):
                break  # This simple?
            elif ray.pos[x] > ray.tile_size - wall_thickness and \
                    maze.check_wall_with_placement(ray.pos_tile, ray.global_index_to(self.RIGHT)):
                break
            if ray.pos[y] < wall_thickness and \
                    maze.check_wall_with_placement(ray.pos_tile, ray.global_index_to(self.DOWN)):
                break
            elif ray.pos[y] > ray.tile_size - wall_thickness and \
                    maze.check_wall_with_placement(ray.pos_tile, ray.global_index_to(self.UP)):
                break

            # If wall edge wasn't hit immediately this tile is included in the ray's path.
            if debugging_in_2D:
                tile_path.append(ray.pos_tile)
                maze.visible_tiles.add(ray.pos_tile)

            # Now trace across the tile.
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
            distance += d  # Update total distance here.

            # Which tile are we approaching and update pos for next iteration.
            if dimension_hit == x:
                if cos_r > 0:
                    local_border_hit = 1  # RIGHT
                    ray.pos[x] = self.tile_size
                else:
                    local_border_hit = 3  # LEFT
                    ray.pos[x] = 0
                ray.pos[y] += d*sin_r
            else:
                if sin_r > 0:
                    local_border_hit = 2  # UP
                    ray.pos[y] = self.tile_size
                else:
                    local_border_hit = 0  # DOWN
                    ray.pos[y] = 0
                ray.pos[x] += d*cos_r

            # Check if we hit a wall.
            global_border_hit = ray.global_index_to(local_border_hit)  # Why ray is a separate object.
            if maze.check_wall_with_placement(ray.pos_tile, global_border_hit):  # True if wall.
                break  # Break loop when wall is hit.
            if distance > 100 * self.tile_size:
                raise RuntimeError("Error: Distance", distance, " too large! Something must have gone wrong.")

            # Update ray object for next iteration
            ray.transfer_tile(maze=maze, global_index_to_new=global_border_hit, local_index_to_new=local_border_hit)

        return distance
