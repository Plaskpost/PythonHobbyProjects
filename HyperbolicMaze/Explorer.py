import numpy as np
import math
import config


class Explorer:
    """
    Represents a generic entity that can explore a maze, keeping track of its position and navigating between tiles.

    Attributes:
       - pos_tile (str): The current tile's identifying key.
       - global_index_to_previous_tile (int): Tracks the global direction index (final to the tile) to the
            previously visited tile.
       - local_index_to_previous (int): Tracks the local direction index (relative to the explorer) to the
            previously visited tile.

    Methods:
       - transfer_tile(maze, local_index_to_new, global_index_to_new=None, generate_for_unexplored=False):
           Transfers the explorer to an adjacent tile, optionally generating new tiles if they do not exist.

       - directional_tile_step(maze, brfl): Moves in a direction specified by 'back', 'right', 'forward' or 'left'.

       - global_index_to(local_index_to_next_tile): Converts a local direction index to a global direction index.

       - opposite_of(direction): Returns the index of the opposite direction for a given index.

       - __copy__(): Creates a shallow copy of the Explorer instance.
    """

    # Constants
    DOWN = 0
    RIGHT = 1
    UP = 2
    LEFT = 3

    FORWARD = 2
    BACKWARDS = 0

    # Index names
    x = 0
    y = 1

    def __init__(self, pos_tile, global_prev, local_prev):
        self.pos_tile = pos_tile
        self.global_index_to_previous_tile = global_prev
        self.local_index_to_previous = local_prev


    def transfer_tile(self, maze, local_index_to_new, global_index_to_new=None, generate_for_unexplored=False):
        """
        Moves the explorer to another tile that is adjacent to its current one.

        :param maze: DynamicMaze object.
        :param local_index_to_new: Local index (= index relative to the explorer's orientation) to the new tile.
        :param global_index_to_new: This variable is computed based on the explorer's traversing history by default.
            User may however specify a value, but may be cautious that incorrect values may cause unwanted "teleportation".
        :param generate_for_unexplored: This variable is set to True if an attempted step into a non-existing tile
            should trigger initialization of that tile.
        :returns: True if the tile transfer was successful, otherwise False.
        """
        if global_index_to_new is None:
            global_index_to_new = self.global_index_to(local_index_to_new)

        new_tile = maze.adjacency_map[self.pos_tile][global_index_to_new]
        if maze.adjacency_map[new_tile] is None:
            if not generate_for_unexplored:
                return False
            else:
                maze.register_tile(new_tile)

        self.local_index_to_previous = self.opposite_of(local_index_to_new)
        try:
            self.global_index_to_previous_tile = maze.adjacency_map[new_tile].index(self.pos_tile)
        except ValueError:
            self.global_index_to_previous_tile = self.global_index_to(self.local_index_to_previous)
            maze.adjacency_map[new_tile][self.global_index_to_previous_tile] = self.pos_tile

        self.pos_tile = new_tile
        return True

    def directional_tile_step(self, maze, brfl):
        """
        Transfers tile in a direction relative to which tile the explorer just came from.

        :param maze: DynamicMaze object.
        :param brfl: 0 = BACK, 1 = RIGHT, 2 = FORWARD, 3 = LEFT. Direction to move.
        :returns: True if the tile transfer was successful, otherwise False.
        """
        local_index_to_new = (self.local_index_to_previous + brfl) % 4
        global_index_to_new = self.global_index_to(local_index_to_new)
        if maze.adjacency_map[maze.adjacency_map[self.pos_tile][global_index_to_new]] is None:
            return False

        self.transfer_tile(maze, local_index_to_new, global_index_to_new)
        return True


    def global_index_to(self, local_index_to_next_tile):
        """
        Converts a local direction index to a global direction index, based on the explorer's trace record.
        """
        # "The amount of clockwise steps from direction_to_previous to direction_to_next is added to index_to_last."
        return (((local_index_to_next_tile - self.local_index_to_previous) % 4) + self.global_index_to_previous_tile) % 4

    def opposite_of(self, direction):
        """
        Returns the index to the opposite direction of the given direction index.
        """
        return (direction + 2) % 4

    def __copy__(self):
        return Explorer(self.pos_tile, self.global_index_to_previous_tile, self.local_index_to_previous)



class Player(Explorer):
    """
    Represents a playable character navigating the maze, extending Explorer with specific movement and rotation
    within tiles.

    Attributes:
        - pos (np.array): Current position in local coordinates within a tile, initialized to the center.
        - rotation (float): Direction the player is facing, in degrees.

    Methods:
        - transfer_tile(maze, local_index_to_new, global_index_to_new=None, generate_for_unexplored=True):
            Moves the player to an adjacent tile, adjusting local coordinates accordingly.

        - move(maze, flbr): Adjusts the player’s position within a tile according to a direction (forward, left, back, right).

        - rotate(left, amount): Rotates the player by a specified number of degrees in the specified direction.

        - get_facing(): Returns the index of the wall in the direction the player is facing.

    Notes:
        - Requires `config` for tile size, initial rotation, and movement speed.
        - Designed to handle relative positioning and directionality within a tile.
    """

    def __init__(self):
        super().__init__(pos_tile="", global_prev=0, local_prev=Explorer.DOWN)

        self.pos = np.array([config.tile_size / 2., config.tile_size / 2.])
        self.rotation = config.initial_rotation


    def transfer_tile(self, maze, local_index_to_new, global_index_to_new=None, generate_for_unexplored=True):
        super().transfer_tile(maze, local_index_to_new, global_index_to_new, generate_for_unexplored)

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
            raise ValueError(f"ERROR: {local_index_to_new} is an invalid direction")

    def move(self, maze, flbr):
        """
        Moves the player's position. Either forward, left, back or right.
        """
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
            global_index_to_ahead = self.global_index_to(i)
            x_or_y = (1 + i) % 2
            wall_ahead = maze.wall_map[self.pos_tile][global_index_to_ahead] == -1
            if near_edge[i] and wall_ahead:
                self.pos[x_or_y] -= v[x_or_y]  # Move back.
            elif across_edge[i]:
                self.transfer_tile(maze, i, global_index_to_ahead)

    def rotate(self, left, amount):
        """
        Rotates the player.

        :param left: True if rotate to the left. False if rotate to the right.
        :param amount: Number of degrees to rotate.
        :return:
        """
        if left:
            self.rotation += amount
            if self.rotation >= 360:
                self.rotation -= 360
        else:  # Right
            self.rotation -= amount
            if self.rotation < 0:
                self.rotation += 360

    def get_facing(self):
        """
        Returns the index to the wall the player is looking towards.
        """
        return ((round(self.rotation)-225) % 360)//90


class Ray(Explorer):
    """
    Represents a ray-casting entity, extending Explorer to calculate distances to walls for vision simulation.

    Attributes:
        - pos (np.array): Local position within the tile, initialized to the player's position.

    Methods:
        - transfer_tile(maze, local_index_to_new, global_index_to_new=None, tile_size=config.tile_size):
            Moves the ray to an adjacent tile, updating local coordinates.

        - shoot(maze, direction, debugging_in_2D): Shoots the ray in a specified direction, measuring distance to
            the first wall encountered.

    Parameters:
        - player (Player): The Player instance from which the ray inherits starting position and direction.

    Notes:
        - Uses a `config` module to access tile size and wall thickness values.
        - Primarily used for collision detection or line-of-sight simulation in the maze.
    """

    def __init__(self, player):
        super().__init__(player.pos_tile, player.global_index_to_previous_tile, player.local_index_to_previous)
        self.pos = player.pos.__copy__()


    def transfer_tile(self, maze, local_index_to_new, global_index_to_new=None, tile_size=config.tile_size):
        super().transfer_tile(maze, local_index_to_new, global_index_to_new)

        # Change local coordinates
        if local_index_to_new == self.DOWN:
            self.pos[self.y] += tile_size
        elif local_index_to_new == self.RIGHT:
            self.pos[self.x] -= tile_size
        elif local_index_to_new == self.UP:
            self.pos[self.y] -= tile_size
        elif local_index_to_new == self.LEFT:
            self.pos[self.x] += tile_size
        else:
            raise ValueError(f"ERROR: {local_index_to_new} is an invalid direction")

    def shoot(self, maze, direction, debugging_in_2D):
        """
        Shoots the ray in a given direction. Returns the distance to the first wall encountered in that direction.

        :param maze: DynamicMaze object.
        :param direction: Direction relative the player's facing, given in degrees.
        :param debugging_in_2D: Set to True if the top view 2D debugging display is in use. That class wants to know
            what tiles the ray traverses.
        :returns: The distance to the first encountered wall in the given direction.
        """
        x = self.x
        y = self.y

        wall_thickness = config.wall_thickness
        outer_wall_limit = config.tile_size - wall_thickness
        distance = 0
        tile_path = []

        while True:
            cos_r = math.cos(math.radians(direction))
            sin_r = math.sin(math.radians(direction))
            dx = config.tile_size - self.pos[x] if cos_r > 0 else self.pos[x]
            dy = config.tile_size - self.pos[y] if sin_r > 0 else self.pos[y]

            # Start by checking if we've collided with the edge of a wall.
            if self.pos[x] < wall_thickness and \
                    maze.check_wall_with_placement(self.pos_tile, self.global_index_to(self.LEFT)):
                break  # This simple?
            elif self.pos[x] > outer_wall_limit and \
                    maze.check_wall_with_placement(self.pos_tile, self.global_index_to(self.RIGHT)):
                break
            if self.pos[y] < wall_thickness and \
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
                try:
                    wall_reductions = np.abs(np.array([wall_thickness / cos_r, wall_thickness / sin_r]))
                    distance -= wall_reductions[dimension_hit]
                except ZeroDivisionError:
                    print("Not doing a proper bug catcher here lol.")
                break  # Break loop when wall is hit.

            # Check if we hit one of our walls in the corner.
            mirrored_pos = np.array([self.pos, config.tile_size-self.pos])
            over_or_under = np.argmin(mirrored_pos, axis=0)
            dists_from_wall = np.array([mirrored_pos[over_or_under[0], 0], mirrored_pos[over_or_under[1], 1]])
            if np.sum(dists_from_wall) < wall_thickness:
                rot_matrix = -(2*(dimension_hit-0.5)).astype(int) * np.array([[1, -1], [-1, 1]])
                rotation = rot_matrix[over_or_under[0], over_or_under[1]]
                global_index_to_wall_of_interest = (global_border_hit + rotation) % 4
                if maze.check_wall_with_placement(self.pos_tile, global_index_to_wall_of_interest):
                    distance -= dimension_hit * (wall_thickness - dists_from_wall[self.x]) / abs(cos_r) + \
                                (1-dimension_hit) * (wall_thickness - dists_from_wall[self.y]) / abs(sin_r)
                    break

            if distance > 100 * config.tile_size:
                raise RuntimeError("Error: Distance", distance, " too large! Something must have gone wrong.")

            # Update ray object for next iteration.
            self.transfer_tile(maze=maze, local_index_to_new=local_border_hit, global_index_to_new=global_border_hit)

        return distance
