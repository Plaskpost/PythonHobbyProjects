import numpy as np

import HyperbolicGrid
import config


class DynamicMaze:
    """
        Class: DynamicMaze
        Description:
            The DynamicMaze class generates and manages a dynamic maze structure
            where tiles are explored, and walls are generated as tiles
            become visible. The maze is represented using an adjacency map
            and a wall map to that are each constructed as the player traverses
            the maze.

        Attributes:
            adjacency_map (dict): Maps each tile to its neighbors.
            wall_map (dict): Tracks the status of walls for each tile, where
                             1 = passable, -1 = wall, and 0 = unexplored.
            visible_tiles (set): A set of tiles currently visible in the maze.
                                 Used primarily for debug purposes.

        Methods:
            __init__(): Initializes instance variables.
            register_tile(tile): Adds a tile and accompanying neighbors to the
                adjacency map.
            check_wall_with_placement(tile, global_index): Checks for a wall at
                a specific location. Initializes unexplored locations.
            place_wall_or_opening(tile, global_index, which_one): Places either a
                wall or an opening at a specified location.
            make_walls(tile): Places either a wall or an opening in every
                direction of a given tile.
            initialize_wall_map(): Creates wall_map() with the same tile keys as
                adjacency_map.

        Usage:
            - Initialize a maze instance and manipulate walls dynamically.

        Notes:
            - Requires 'HyperbolicGrid' for tile registration.
            - Configuration settings in `config` affect the wall placement logic.
        """

    def __init__(self):
        self.adjacency_map = {"": None}  # Tracks which tiles are adjacent to which.
        self.wall_map = self.initialize_wall_map()  # Tracks where there are walls or openings. 1: passable, -1: wall, 0: unexplored.
        self.visible_tiles = set()  # Only used by the Euclidean top view 2D-rendering strategy.
        self.register_tile("")  # Will Initialize all adjacencies to the ""-tile.
        self.make_walls("")


    def register_tile(self, tile):
        """
        Establishes a tile and utilizes HyperbolicGrid.iterative_registration to find the suggested tile keys to its
        neighbors.
        """
        HyperbolicGrid.iterative_registration(tile, self.adjacency_map)
        self.wall_map[tile] = [0, 0, 0, 0]

    def make_walls(self, tile):
        """
        Sets any not already set edge (0) of a given tile to either wall (-1) or opening (+1).
        :param tile: Tile key to the tile in question.
        :returns: False if there were no unexplored edges to the tile. True if anything got updated.
        :raises KeyError: If the given tile is missing from the adjacency map or from the wall map.
        """
        if tile not in self.adjacency_map.keys():  # Shouldn't trigger because make_walls adds all neighbors.
            raise KeyError(f"ERROR: Tile {tile} explored before added!")

        if tile not in self.wall_map.keys():
            raise KeyError(f"ERROR: Tile {tile} is in the adjacency map but not the wall map.")

        # make_walls is called whenever a tile becomes visible.
        # These lines enable ending iterations when all walls are complete.
        if 0 not in self.wall_map[tile]:
            return False

        for i in range(4):
            if self.wall_map[tile][i] == 0:
                self.place_wall_or_opening(tile, i)

        return True

    def place_wall_or_opening(self, tile, global_index, which_one='random'):
        """
        Places either a wall (-1) or an opening (-1) at both sides of an unexplored (0) edge.

        :param tile: Tile key to one of the tiles at the edge.
        :param global_index: Global index to the referred edge at tile.
        :param which_one: 'wall' if wall. 'opening' if opening. For other values the result will be random.
        :raises KeyError: If tile does not exist in the adjacency map.
        :raises ValueError: If tile is not included in the neighboring tile's list of adjacencies.
        :raises AttributeError: If the neighboring tile wasn't properly registered.
        :return:
        """
        # Line needed here for the Rendering3D strategy.
        if self.adjacency_map[tile] is None:
            self.register_tile(tile)

        # I believe it can be done better than setting this every call.
        num_walls = np.round(0.5 * np.random.randn() + config.average_walls_per_tile).astype(int)
        num_zeros = self.wall_map[tile].count(0)
        existing_walls = self.wall_map[tile].count(-1)
        prob = (num_walls - existing_walls) / num_zeros
        neighbor = self.adjacency_map[tile][global_index]

        if self.adjacency_map[neighbor] is None:  # Always add the tile behind the observed border.
            self.register_tile(neighbor)

        if which_one == 'wall':
            value = -1
        elif which_one == 'opening':
            value = 1
        else:  # Random if anything else.
            value = -1 if np.random.random() < prob else 1

        try:
            self.wall_map[tile][global_index] = value
            self.wall_map[neighbor][self.adjacency_map[neighbor].index(tile)] = value

        except KeyError:
            print(f"Tile: {tile}")
            print(f"Neighbor: {neighbor}")
            print("Full adjacency map:")
            for name, adjacents in self.wall_map.items():
                if adjacents is not None:
                    print(name, ":", adjacents)
            raise KeyError(f"ERROR: Non-existing tile {tile} encountered.")
        except ValueError:
            print(f"Tile: {tile}")
            print(f"Neighbor: {neighbor}")
            print(f"Entries: \n {tile} : {self.adjacency_map[tile]} \n {neighbor} : {self.adjacency_map[neighbor]}")
            raise ValueError(f"ERROR: {tile} is not in {neighbor}'s adjacencies.")
        except AttributeError:
            print(f"Tile: {tile}")
            print(f"Neighbor: {neighbor}")
            print(f"Entries: \n {tile} : {self.adjacency_map[tile]} \n {neighbor} : {self.adjacency_map[neighbor]}")
            raise AttributeError(f"ERROR: {neighbor}, neighbor to {tile} was accessed before initialized.")

    def check_wall_with_placement(self, tile, global_index):
        """
        Checks whether there is a wall (-1) or an opening (+1) at a specific location (tile and specified edge).
        If the edge is unexplored (0), it generates either one at random.

        :param tile: Tile key to the tile in question.
        :param global_index: Global index ("true index") to the tile edge to look at. 0=down, 1=right, 2=up, 3=left.
        :returns: True if wall. False if opening.
        """
        if self.wall_map[tile][global_index] == 0:
            self.place_wall_or_opening(tile, global_index)
        return self.wall_map[tile][global_index] == -1


    def initialize_wall_map(self):
        """
        Fills up the wall_map with the same entries as in adjacency_map.
        """
        wall_map = {}
        for tile, adjacents in self.adjacency_map.items():
            if adjacents is None:
                continue
            wall_map[tile] = [0, 0, 0, 0]

        return wall_map


def get_plain_map(num_layers, walls='openings', simple_registration_algorithm=False):
    """
    Written to test whether all walls on the MiniMap are placed correctly. Initializes a small maze where every edge is
    an opening, or every edge is a wall. Alternatively, walls except from the edges to the starting tile.

    :param num_layers: "radius" of the maze.
    :param walls: 'walls' for walls, 'openings' for openings, 'walls+' for walls except adjacent to the starting tile.
    :param simple_registration_algorithm: Added to challenge the current iterative key generating algorithm.
        The purpose was to test if greedy key generation combined with post-processing "fixes" yielded a better result.
    :returns: DynamicMaze object of the generated maze.
    """
    maze = DynamicMaze()
    HyperbolicGrid.bulk_registration(maze.adjacency_map, "", num_layers)
    for key in maze.adjacency_map:
        if maze.adjacency_map[key] is not None:
            maze.wall_map[key] = [1, 1, 1, 1] if walls == 'openings' else [-1, -1, -1, -1]

    if walls == 'walls+':
        maze.wall_map[''] = [1, 1, 1, 1]
        maze.wall_map['D'][2] = 1
        maze.wall_map['R'][3] = 1
        maze.wall_map['U'][0] = 1
        maze.wall_map['L'][1] = 1

    return maze


