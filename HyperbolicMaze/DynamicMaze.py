import numpy as np
import HyperbolicGrid
import config


class DynamicMaze:

    def __init__(self, pos):
        self.adjacency_map = {}
        self.wall_map = {}  # 1: passable, -1: wall, 0: unexplored
        self.visible_tiles = set()  # Set of visible tiles
        self.register_tile(pos)
        self.make_walls(pos)

    def register_tile(self, tile):
        HyperbolicGrid.register_tile(tile, self.adjacency_map)
        self.wall_map[tile] = [0, 0, 0, 0]

    def make_walls(self, tile):
        if tile not in self.adjacency_map:  # Shouldn't trigger because make_walls adds all neighbors.
            raise RuntimeError("ERROR: Tile ", tile, " explored before added!")

        # make_walls is called whenever a tile becomes visible. This ends the function whenever all walls are complete.
        if 0 not in self.wall_map[tile]:
            return

        for i in range(4):
            if self.wall_map[tile][i] == 0:
                self.place_wall_or_opening(tile, i)

    def check_wall_with_placement(self, tile, global_index):
        if self.wall_map[tile][global_index] == 0:
            self.place_wall_or_opening(tile, global_index)
        return self.wall_map[tile][global_index] == -1

    def place_wall_or_opening(self, tile, global_index):
        # Line needed here for the Rendering3D strategy.
        if tile not in self.adjacency_map:
            HyperbolicGrid.register_tile(tile, self.adjacency_map)

        # I believe it can be done better than setting this every call.
        num_walls = np.round(0.5 * np.random.randn() + config.average_walls_per_tile).astype(int)
        num_zeros = self.wall_map[tile].count(0)
        existing_walls = self.wall_map[tile].count(-1)
        prob = (num_walls - existing_walls) / num_zeros
        neighbor = self.adjacency_map[tile][global_index]

        if neighbor not in self.adjacency_map:  # Always add the tile behind the observed border.
            self.register_tile(neighbor)

        try:
            if np.random.random() < prob:
                self.wall_map[tile][global_index] = -1
                self.wall_map[neighbor][self.adjacency_map[neighbor].index(tile)] = -1
            else:
                self.wall_map[tile][global_index] = 1
                self.wall_map[neighbor][self.adjacency_map[neighbor].index(tile)] = 1
        except KeyError:
            print("Full adjacency map:")
            for name, adjacents in self.adjacency_map.items():
                print(name, ":", adjacents)
            raise RuntimeError