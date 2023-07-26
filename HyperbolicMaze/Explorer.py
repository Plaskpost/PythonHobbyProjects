import numpy as np
import math
import config


class Explorer:
    DOWN = 0
    RIGHT = 1
    UP = 2
    LEFT = 3

    x = 0
    y = 1

    def __init__(self, pos, pos_tile, global_prev, local_prev):
        self.pos = pos
        self.pos_tile = pos_tile
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
            self.pos[self.y] += config.tile_size
        elif local_index_to_new == self.RIGHT:
            self.pos[self.x] -= config.tile_size
        elif local_index_to_new == self.UP:
            self.pos[self.y] -= config.tile_size
        elif local_index_to_new == self.LEFT:
            self.pos[self.x] += config.tile_size
        else:
            raise ValueError("Invalid direction!")

        return self.__copy__()

    def opposite_of(self, direction):
        return (direction + 2) % 4

    def __copy__(self):
        return Explorer(self.pos, self.pos_tile,
                        self.global_index_to_previous_tile, self.local_index_to_previous)


class Player(Explorer):

    def __init__(self):
        super().__init__(pos=np.array([config.tile_size / 2.0, config.tile_size / 2.0]), pos_tile="",
                         global_prev=0, local_prev=Explorer.DOWN)

        self.rotation = 90  # Initially UP.

    def move(self, maze, flbr):
        rotation_matrix = np.array([[0, -1], [1, 0]])
        rotation_matrix = np.linalg.matrix_power(rotation_matrix, flbr)
        v = config.movement_speed * np.array([math.cos(math.radians(self.rotation)),
                                              math.sin(math.radians(self.rotation))])
        v = np.dot(rotation_matrix, v)
        self.pos += v
        near_edge = [(self.pos[1] < config.player_radius and v[1] < 0),
                     (self.pos[0] >= config.tile_size - config.player_radius and v[0] > 0),
                     (self.pos[1] >= config.tile_size - config.player_radius and v[1] > 0),
                     (self.pos[0] < config.player_radius and v[0] < 0)]
        across_edge = [(self.pos[1] < 0), (self.pos[0] >= config.tile_size),
                       (self.pos[1] >= config.tile_size), (self.pos[0] < 0)]
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


class Ray(Explorer):

    def __init__(self, player, direction):
        super().__init__(player.pos, player.pos_tile, player.global_prev, player.local_prev)
        self.direction = direction

    def old_shoot(self, maze, direction, debugging_in_2D):
        # Indices
        x = self.x
        y = self.y

        outer_wall_limit = config.tile_size - config.wall_thickness
        distance = 0
        tile_path = []

        while True:
            cos_r = math.cos(math.radians(direction))
            sin_r = math.sin(math.radians(direction))
            dx = config.tile_size - self.pos[x] if cos_r > 0 else self.pos[x]
            dy = config.tile_size - self.pos[y] if sin_r > 0 else self.pos[y]

            # Start by checking if we've collided with the edge of a wall.
            if self.pos[x] < config.wall_thickness and \
                    maze.check_wall_with_placement(self.pos_tile, self.global_index_to(self.LEFT)):
                break  # This simple?
            elif self.pos[x] > outer_wall_limit and \
                    maze.check_wall_with_placement(self.pos_tile, self.global_index_to(self.RIGHT)):
                break
            if self.pos[y] < config.wall_thickness and \
                    maze.check_wall_with_placement(self.pos_tile, self.global_index_to(self.DOWN)):
                break
            elif self.pos[y] > outer_wall_limit and \
                    maze.check_wall_with_placement(self.pos_tile, self.global_index_to(self.UP)):
                break

            # If wall edge wasn't hit immediately this tile is included in the ray's path.
            if debugging_in_2D:
                tile_path.append(self.pos_tile)
                maze.visible_tiles.add(self.pos_tile)

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
                    local_border_hit = self.RIGHT
                    self.pos[x] = config.tile_size
                else:
                    local_border_hit = self.LEFT
                    self.pos[x] = 0
                self.pos[y] += d*sin_r
            else:
                if sin_r > 0:
                    local_border_hit = self.UP
                    self.pos[y] = config.tile_size
                else:
                    local_border_hit = self.DOWN
                    self.pos[y] = 0
                self.pos[x] += d*cos_r

            # Check if we hit a wall.
            global_border_hit = self.global_index_to(local_border_hit)  # Why ray is a separate object.
            if maze.check_wall_with_placement(self.pos_tile, global_border_hit):  # True if wall.
                wall_reductions = np.abs(np.array([config.wall_thickness / cos_r, config.wall_thickness / sin_r]))
                distance -= wall_reductions[dimension_hit]
                break  # Break loop when wall is hit.

            # Check if we hit one of our walls in the corner.
            mirrored_pos = np.array([self.pos, config.tile_size-self.pos])
            over_or_under = np.argmin(mirrored_pos, axis=0)
            dists_from_wall = np.array([mirrored_pos[over_or_under[0], 0], mirrored_pos[over_or_under[1], 1]])
            if np.sum(dists_from_wall) < config.wall_thickness:
                rot_matrix = -(2*(dimension_hit-0.5)).astype(int) * np.array([[1, -1], [-1, 1]])
                rotation = rot_matrix[over_or_under[0], over_or_under[1]]
                global_index_to_wall_of_interest = (global_border_hit + rotation) % 4
                if maze.check_wall_with_placement(self.pos_tile, global_index_to_wall_of_interest):
                    distance -= dimension_hit * (config.wall_thickness - dists_from_wall[self.x]) / abs(cos_r) + \
                                (1-dimension_hit) * (config.wall_thickness - dists_from_wall[self.y]) / abs(sin_r)
                    break

            if distance > 100 * config.tile_size:
                raise RuntimeError("Error: Distance", distance, " too large! Something must have gone wrong.")

            # Update ray object for next iteration.
            self.transfer_tile(maze=maze, global_index_to_new=global_border_hit, local_index_to_new=local_border_hit)

        return distance

    @staticmethod
    def shoot_all(maze, player, left_edge, right_edge):
        ray = Ray(player, left_edge)
        # - Initialize Tile every time a new Tile is encountered.
        # May happen that we drop the tile memory after we exit this function, I'll see about that.

        tiles = {ray.pos_tile: Tile(ray.pos_tile)}

        while True:
            tile_id = ray.pos_tile
            if tile_id not in tiles:
                tiles[tile_id] = Tile(tile_id)

        return None
