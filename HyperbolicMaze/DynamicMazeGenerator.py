import random
import numpy as np
import HyperbolicGrid


class DynamicMazeGenerator:

    def __init__(self, starting_pos):
        self.adjacency_map = {}
        self.wall_map = {}  # 1: passable, -1: wall, 0: unexplored
        self.visibility_map = {}  # True: visible, False: not visible
        self.register_tile(starting_pos)

    def register_tile(self, tile):
        HyperbolicGrid.register_tile(tile, self.adjacency_map)
        self.wall_map[tile] = [0, 0, 0, 0]
        self.visibility_map[tile] = [False, False, False, False]

    # Should update self.visibility_map following a given position.
    def update_visibility(self, tile, pos_tile):

        for i in range(4):
            neighbor = self.adjacency_map[tile][i]
            if visible(pos_tile, tile, neighbor):
                self.visibility_map[tile][i] = True
                self.make_walls(neighbor)  # This statement guarantees that the next cannot find a '0' in wall_map.
                self.update_visibility(neighbor)
            else:
                self.visibility_map[tile][i] = False

    def make_walls(self, tile):
        if tile not in self.adjacency_map:  # Shouldn't trigger because make_walls adds all neighbors.
            raise RuntimeError("ERROR: Tile ", tile, " explored before added!")

        # make_walls is called whenever a tile becomes visible. This ends the function whenever all walls are complete.
        if 0 not in self.wall_map[tile]:
            return

        num_walls = np.round(0.5 * np.random.randn() + 2.0).astype(int)
        for i in range(4):
            neighbor = self.adjacency_map[tile][i]
            if neighbor not in self.adjacency_map:
                self.register_tile(neighbor)
            if self.wall_map[tile][i] == 0:
                if compute_if_wall(num_walls, self.wall_map[tile]):
                    self.wall_map[tile][i] = -1
                    self.wall_map[neighbor][i - 2] = -1
                else:
                    self.wall_map[tile][i] = 1
                    self.wall_map[neighbor][i - 2] = 1


# ----------- Just some help functions ----------------
def compute_if_wall(self, num_walls, wall_vec):
    num_zeros = wall_vec.count(0)
    existing_walls = wall_vec.count(-1)
    prob = (num_walls - existing_walls) / num_zeros
    if random.random() < prob:
        return True
    return False
