import numpy as np
import HyperbolicGrid


class DynamicMaze:

    def __init__(self, pos, average_walls_per_tile):
        self.adjacency_map = {}
        self.wall_map = {}  # 1: passable, -1: wall, 0: unexplored
        self.visible_tiles = set()  # Set of visible tiles
        self.average_walls_per_tile = average_walls_per_tile
        self.register_tile(pos)
        self.make_walls(pos)

    def register_tile(self, tile):
        HyperbolicGrid.register_tile(tile, self.adjacency_map)
        self.wall_map[tile] = [0, 0, 0, 0]

    def update_visibility(self, tile):  # TODO: Den här ska bort och ersättas med rayuppdatering
        self.visible_tiles = set()
        self.update_visibility_recursive(tile, "")

    # Recursive. Should update self.visible_tiles following a given position.
    def update_visibility_recursive(self, tile, move_sequence):
        if tile in self.visible_tiles:
            return

        if check_visibility(move_sequence):
            self.visible_tiles.add(tile)
            for new_direction in range(4):
                # Walls do not lead to a visible tile.
                if self.wall_map[tile][new_direction] == -1:
                    continue
                # No need to look where we came from.
                if len(move_sequence) >= 2 and HyperbolicGrid.check_opposites(move_sequence[-1], move_sequence[-2]):
                    continue

                neighbor = self.adjacency_map[tile][new_direction]
                self.make_walls(neighbor)  # This statement first guarantees that the next cannot find a '0' in wall_map
                d = ["D", "R", "U", "L"]
                self.update_visibility_recursive(neighbor, move_sequence+d[new_direction])

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
        num_walls = np.round(0.5 * np.random.randn() + self.average_walls_per_tile).astype(int)
        num_zeros = self.wall_map[tile].count(0)
        existing_walls = self.wall_map[tile].count(-1)
        prob = (num_walls - existing_walls) / num_zeros
        neighbor = self.adjacency_map[tile][global_index]

        if neighbor not in self.adjacency_map:  # Always add the tile behind the observed border.
            self.register_tile(neighbor)

        if np.random.random() < prob:
            self.wall_map[tile][global_index] = -1
            self.wall_map[neighbor][self.adjacency_map[neighbor].index(tile)] = -1
        else:
            self.wall_map[tile][global_index] = 1
            self.wall_map[neighbor][self.adjacency_map[neighbor].index(tile)] = 1


# ----------- Just some help functions ----------------

def check_visibility(move_sequence):
    if len(move_sequence) < 4:
        return True
    tail = move_sequence[-3:]
    if tail[:2] == 'FF':
        return True

    return False
