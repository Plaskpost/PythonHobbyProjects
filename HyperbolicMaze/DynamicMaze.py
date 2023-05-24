import random
import numpy as np
import HyperbolicGrid


class DynamicMaze:

    def __init__(self, pos, average_walls_per_tile):
        self.adjacency_map = {}
        self.wall_map = {}  # 1: passable, -1: wall, 0: unexplored
        self.visibility_map = {}  # True: visible, False: not visible
        self.average_walls_per_tile = average_walls_per_tile
        self.register_tile(pos)
        self.make_walls(pos)

    def register_tile(self, tile):
        HyperbolicGrid.register_tile(tile, self.adjacency_map)
        self.wall_map[tile] = [0, 0, 0, 0]
        self.visibility_map[tile] = False

    def update_visibility(self, tile):
        self.update_visibility_recursive(tile, "", 2)  # Always up?

    # Recursive. Should update self.visibility_map following a given position.
    def update_visibility_recursive(self, tile, move_sequence, face_direction):  # (string, string, int)
        tile_visible = check_visibility(move_sequence)
        self.visibility_map[tile] = tile_visible
        if tile_visible:
            turn_letters = ['F', 'L', 'B', 'R']
            for new_direction in range(4):
                if self.wall_map[tile][new_direction] == -1:  # Walls do not lead to a visible tile.
                    continue
                turning = (new_direction-face_direction) % 4  # face_direction: [D, R, U, L], i: [F, L, B, R]
                if turning == 2 and len(move_sequence) > 0:  # No need to look where we came from.
                    continue
                neighbor = self.adjacency_map[tile][new_direction]
                self.make_walls(neighbor)  # This statement first guarantees that the next cannot find a '0' in wall_map
                self.update_visibility_recursive(neighbor, move_sequence+turn_letters[turning], new_direction)

    def make_walls(self, tile):
        if tile not in self.adjacency_map:  # Shouldn't trigger because make_walls adds all neighbors.
            raise RuntimeError("ERROR: Tile ", tile, " explored before added!")

        # make_walls is called whenever a tile becomes visible. This ends the function whenever all walls are complete.
        if 0 not in self.wall_map[tile]:
            return

        num_walls = np.round(0.5 * np.random.randn() + self.average_walls_per_tile).astype(int)
        for i in range(4):
            neighbor = self.adjacency_map[tile][i]
            if neighbor not in self.adjacency_map:
                self.register_tile(neighbor)
            if self.wall_map[tile][i] == 0:
                if compute_if_wall(num_walls, self.wall_map[tile]):
                    self.wall_map[tile][i] = -1
                    self.wall_map[neighbor][self.adjacency_map[neighbor].index(tile)] = -1
                else:
                    self.wall_map[tile][i] = 1
                    self.wall_map[neighbor][self.adjacency_map[neighbor].index(tile)] = 1


# ----------- Just some help functions ----------------
def compute_if_wall(num_walls, wall_vec):
    num_zeros = wall_vec.count(0)
    existing_walls = wall_vec.count(-1)
    prob = (num_walls - existing_walls) / num_zeros
    if random.random() < prob:
        return True
    return False


def check_visibility(move_sequence):
    if len(move_sequence) < 4:
        return True
    tail = move_sequence[-3:]
    if tail[:2] == 'FF':
        return True

    return False
    # TODO: Continue here.
