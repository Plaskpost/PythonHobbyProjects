import json
import math
from collections import deque
import pygame
import numpy as np

import DynamicMaze
import Game
import Rendering2D
import DataGenerator
import config
from Rendering import Rendering


class MiniMap(Rendering):
    """
    MiniMap Class

    The `MiniMap` class extends the `Rendering` abstract base class to create a hyperbolic minimap on a Poincaré disc,
    that visualizes the maze from above.

    Main attributes:
    -----------
    - `d` (float): Hyperbolic distance between tile centers. Found it by accident. There might be some mathematical
            motivation for its value.
    - `visible_range` (float): Ensures not too many points close by the edge are included. Slightly redundant with the
            current algorithm, as it also has a set radius of tiles that are included in the mini-map visualisation.
    - `tile_size` (float): Width of one tile.
    - `tile_generating` (bool): Allows generation of all tiles near the player if set True.
    - `map_size_on_screen` (int): Size of the map in pixels on the screen, derived from config settings.
    - `full_screen` (bool): Indicates whether the minimap occupies the entire screen or a smaller part.
    - `map_center` (tuple[int, int]): Coordinates for the center point of the minimap on the screen.
    - `grid_map_dict` (dict): Precomputed dictionary of some coordinates of all nearby tiles as a tool to find relevant
            coordinates more efficiently.
    - `player_tile` (str): Identifies the current tile occupied by the player.
    """

    d = 1.06
    visible_range = 1.
    tile_size = 0.5

    def __init__(self, dynamic_maze, explorer, placement='center', tile_generating=False):
        super().__init__('Poincaré map', dynamic_maze, explorer)
        self.tile_generating = tile_generating  # Allows MiniMap to generate all tiles that would be visible on the map.
        self.map_size_on_screen = config.mini_map_size  # Size of the map on the screen.
        self.full_screen = False
        if placement == 'center':
            self.full_screen = True
            self.map_size_on_screen = np.min(self.SCREEN_SIZE)
            self.map_center = self.SCREEN_SIZE // 2
        elif placement == 'bottom-right':
            self.map_center = self.SCREEN_SIZE - self.map_size_on_screen / 1.5
        else:
            raise ValueError(f"ERROR: I forgot to implement code for screen placement {placement}.")

        self.grid_map_dict = load_grid_dict(f'GridMapDict{config.num_grid_points}x{config.num_grid_points}')
        self.player_tile = 'None'

        # Currently unused variables:
        self.target_positions = {}
        self.current_positions = {}
        self.new_visible_tiles = set()
        self.point_adjustment_rate = 0.02
        self.color_adjustment_rate = 5

    def update(self):
        """
        Main function to update the display.
        """
        if self.full_screen:
            self.screen.fill(self.BLACK)

        self.hyperbolic_map_display()

        # self.write_debug_info()

    # ----------------- DISPLAY FUNCTIONS ------------------------

    def hyperbolic_map_display(self):
        """
        Main drawing function.
        """
        visible_tiles, all_walls = self.map_out_walls()

        # Draw background
        pygame.draw.circle(self.screen, self.BLACK, self.map_center, 5 + self.map_size_on_screen // 2, width=0)
        pygame.draw.circle(self.screen, self.WHITE, self.map_center, 3 + self.map_size_on_screen // 2, 2)

        # Drawing the floor of every tile.
        # self.draw_tiles(visible_tiles)

        # Walls.
        wall_thickness = (self.map_size_on_screen / 8000.) * config.wall_thickness
        for tile_key, direction_index in all_walls:
            corners = visible_tiles[tile_key][1:, :]
            wall_corner_1 = corners[direction_index - 1, :]
            wall_corner_2 = corners[direction_index, :]
            if self.full_screen:
                pygame.draw.line(self.screen, self.WHITE, self.screen_scaled_point(wall_corner_1),
                                 self.screen_scaled_point(wall_corner_2), 2 * config.wall_thickness)
            else:
                self.draw_thicker_line((wall_corner_1, wall_corner_2), self.WHITE, thickness_multiplier=wall_thickness)

        # Point representing the player
        player_dot_size = (self.map_size_on_screen / 500.) * config.player_radius
        pygame.draw.circle(self.screen, self.RED, self.screen_scaled_point([0., 0.]), player_dot_size)

    def all_circles_display(self):
        """
        Mainly written for debug purposes. Draws every circle path on the map, but uses the original but slow computation method.
        """
        collection_set = DataGenerator.find_coordinates_and_circles(self.maze, self.explorer,
                                                                    self.get_tile_center_map_placement())

        # Unit circle
        pygame.draw.circle(self.screen, self.WHITE, self.map_center, self.map_size_on_screen // 2, 1)

        # Player square
        #square_height = self.screen_scaled_distance(self.tile_size) - 2 * config.wall_thickness
        #Rendering2D.draw_square(self.screen, self.screen_scaled_point(self.get_tile_center_map_placement()),
        #                        (self.explorer.rotation - config.initial_rotation),
        #                        (square_height, square_height), (50, 50, 50))

        for relative_tile_key, (tile_point, (circle_center, circle_radius), up_angle) in collection_set.items():
            # Circle path
            if np.isinf(circle_radius):
                x0 = -min(1., circle_center[1])
                y0 = -min(1., circle_center[0])
                x1 = min(1., circle_center[1])
                y1 = min(1., circle_center[0])
                pygame.draw.line(self.screen, self.WHITE, self.screen_scaled_point([x0, y0]),
                                 self.screen_scaled_point([x1, y1]), 1)
            else:
                pygame.draw.circle(self.screen, self.WHITE, self.screen_scaled_point(circle_center),
                                   self.screen_scaled_distance(circle_radius), 1)

            # The point representing the center of the tile
            size = line_width(tile_point)
            pygame.draw.circle(self.screen, self.WHITE, self.screen_scaled_point(tile_point), 8.*size)
            #text = self.font.render(relative_tile_key, True, self.DEBUG_BLUE)
            #self.screen.blit(text, self.screen_scaled_point(tile_point))

        # Point representing the player
        pygame.draw.circle(self.screen, self.RED, self.screen_scaled_point([0, 0]), 5)


# ------------------- FUNCTIONS FOR FINDING ALL RELEVANT COORDINATES ----------------------

    def map_out_walls(self):
        """
        Traverses the maze in BFS order to find walls and tile locations.
        """
        all_walls = set()
        visited = {}  # {tile_key: [tile_center, bottom_right_corner, top_right_, top_left_, bottom_left_]}
        probe = self.explorer.__copy__()
        reference_00_indices = self.find_00_indices()  # Index to bottom_left coordinates of current player subsquare.

        queue = deque([[probe, all_walls, visited, reference_00_indices, '']])

        while queue:
            tile_pack = queue.popleft()
            self.tile_specific_mapping(queue, *tile_pack)

        return visited, all_walls

    def tile_specific_mapping(self, queue, probe, all_walls, visited, reference_00_indices, local_journey):
        """
        Performs the recursive mapping algorithm for one tile, finding its coordinates, walls and adding neighboring
        tiles to the queue.

        :param queue: For storing tiles to explore in BFS order.
        :param probe: Explorer object for proper grid traversal.
        :param all_walls: Set for collecting found walls.
        :param visited: To track if a tile has already been visited by the algorithm.
        :param reference_00_indices: Index to bottom_left coordinates of current player subsquare.
        :param local_journey: A string segment telling the path traversed from the tile of the player.
        :return:
        """
        tile_points = self.get_estimated_tile_points(reference_00_indices, self.grid_map_dict[local_journey])
        tile_center = tile_points[0, :]

        # Discard iteration if we're outside the visible range
        if np.linalg.norm(tile_center) > self.visible_range:
            return

        visited[probe.pos_tile] = tile_points

        d = ['D', 'R', 'U', 'L']
        for direction_index in range(4):
            probe_ahead = probe.__copy__()
            if not probe_ahead.transfer_tile(self.maze, direction_index, generate_for_unexplored=self.tile_generating):
                continue
            journey_ahead = local_journey + d[direction_index]
            # Add the step if neighboring tile has been visited
            if probe_ahead.pos_tile in visited:
                global_index_to_ahead = self.maze.adjacency_map[probe.pos_tile].index(probe_ahead.pos_tile)
                if self.maze.check_wall_with_placement(probe.pos_tile, global_index_to_ahead):
                    all_walls.add((probe.pos_tile, direction_index))
            # Otherwise add neighbor to the queue assuming we've mapped positions for it.
            elif journey_ahead in self.grid_map_dict:
                queue.append([probe_ahead, all_walls, visited, reference_00_indices, journey_ahead])

    def find_00_indices(self):
        """
        Finds the indices to the point in the grid of size like in self.grid_map_dict, that's in the bottom left
        corner of the square grid tile that encapsulates point.

        :returns: arg(p0) = [i, j] to p0 in the unit square grid.
        """
        point = self.get_tile_center_unit_square_placement()
        n = self.grid_map_dict[''].shape[0] - 1
        for i in range(n):
            if (i + 1) / n > point[0]:
                break
        for j in range(n):
            if (j + 1) / n > point[1]:
                break

        return i, j

    def adjust_instance_sets(self, visible_tiles):
        """
        Outdated function. The purpose was to make the map transition more smoothly between tiles, by letting every
        tile's coordinates on the map move in small steps towards updated target values, which would prevent them from
        "snapping" on to the new values. The approach was discarded when tile corners were included in the map, as it
        would slowly rotate every tile that changed orientation compared to the player in an tile transition.
        """
        # If a tile transition has occurred
        if self.explorer.pos_tile != self.player_tile:
            self.player_tile = self.explorer.pos_tile
            self.new_visible_tiles = {k for k in visible_tiles.keys() if k not in self.target_positions}

        for tile in visible_tiles.keys():
            if tile not in self.target_positions:
                self.new_visible_tiles.add(tile)

        for tile in self.new_visible_tiles:
            self.current_positions[tile] = visible_tiles[tile]

        self.target_positions = visible_tiles
        for tile, target in visible_tiles.items():
            self.current_positions[tile] = self.nudge_point(self.current_positions[tile], target)

    def nudge_point(self, point, target_point):
        """
        Also outdated partner function to adjust_instance_sets(). Moves a point towards its target values.
        """
        diff = target_point - point
        return point + self.point_adjustment_rate * diff

    def get_estimated_tile_points(self, reference_00_indices, grid_mat):
        """
        Utilizes a grid matrix of saved coordinates for some tile near the player, to estimate the tile's position on
        the map given the player's current coordinates.

        :param reference_00_indices: The player is located on some square in the grid. This parameter gives the indices
            to the bottom-left corner of that square in the grid matrix.
        :param grid_mat: Grid matrix of coordinates for some tile given player positions on every point in the grid
            between -+tile_size / 2 = player tile center at +-tile_size / 2.
        :returns: 5x2 array of estimated coordinates for [tle_center, bottom_right_corner, top_right_corner,
            top_left_corner, bottom_left_corner] of the tile in question.
        """
        tile_points = np.zeros((5, 2))

        player_tile_center = self.get_tile_center_map_placement()
        i, j = reference_00_indices
        n = config.num_grid_points - 1
        reference_targets = grid_mat[i:(i + 2), j:(j + 2)]
        reference_targets = reference_targets.reshape((4, 5, 2))
        reference_00_point = self.tile_size * (-0.5 + np.array([i / n, j / n]))
        reference_points = reference_00_point + self.tile_size / n * np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])

        for i in range(5):
            tile_points[i, :] = bilinear_interpolation(reference_points, reference_targets[:, i, :], player_tile_center)

        return tile_points

    def get_tile_center_map_placement(self, player_unit_pos=None):
        """
        Returns the coordinates of the center of the tile of the player. May be used with another specified value,
        but will fetch self.explorer.pos as player position otherwise.

        :param player_unit_pos: Point within the unit square that represents the player's position.
        :returns: The unscaled position in the unit circle of the center of the tile the player is standing in.
        """
        if player_unit_pos is None:
            player_unit_pos = self.explorer.pos / config.tile_size
        elif player_unit_pos.any() < 0. or player_unit_pos.any() > 1.:
            raise ValueError(f"ERROR: Point {player_unit_pos} outside the unit square.")

        # One of the few Euclidean calculations in this class. Because of the short distance, it would be a little
        # computationally overkill to try to find the relative hyperbolic position.
        return self.tile_size * (0.5 - player_unit_pos)

    def get_tile_center_unit_square_placement(self):
        """
        Returns the position of the tile center inside the unit square, given the player position.
        """
        return 1. - self.explorer.pos / config.tile_size


# ----------------------- SCREEN AND DRAWING FUNCTIONS --------------------------
    def screen_scaled_point(self, point):
        """
        Returns the screen adapted coordinates of a given point.
        """
        rotation = np.radians(360. - self.explorer.rotation + config.initial_rotation)
        rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation)],
                                    [np.sin(rotation), np.cos(rotation)]])
        new_point = np.matmul(rotation_matrix, point)
        new_point = self.map_center + (self.map_size_on_screen // 2) * np.array([new_point[0], -new_point[1]])
        return new_point

    def screen_scaled_distance(self, distance):
        """
        Returns the screen adapted scalar.
        """
        return (self.map_size_on_screen // 2) * distance

    def draw_tiles(self, visible_tiles):
        """
        Draws every found tile as a polygon between its four corner points.
        """
        for tile_key, tile_points in visible_tiles.items():
            # color_strength = min(self.ticks_since_tile_transition * self.color_adjustment_rate, 255) \
            #    if tile_key in self.new_visible_tiles else 255
            # color = (color_strength, color_strength, color_strength)

            polygon_points = [tuple(self.screen_scaled_point(tile_points[i + 1, :])) for i in range(4)]

            pygame.draw.polygon(self.screen, self.GRAY, polygon_points)
            self.debug_labels(tile_points[0, :], tile_key, color=self.RED)

    def draw_thicker_line(self, line, color, thickness_multiplier=1.):
        """
        Draws a "line" between two points given in line = (p1, p2), but thickness of both ends are determined by how far
        away they are from the circle center.

        :param line: (p1, p2), the two points to draw the line between.
        :param color: (r, g, b), color of the line.
        :param thickness_multiplier: An optional parameter to adjust the thickness of the line futrher.
        :return:
        """
        x1, y1 = line[0]
        x2, y2 = line[1]
        ta = line_width(line[0]) * thickness_multiplier
        tb = line_width(line[1]) * thickness_multiplier
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        # Avoid division by zero
        if length == 0:
            return
        dx /= length
        dy /= length
        nx = -dy
        ny = dx

        # Calculate the four corners of the quadrilateral
        x1_offset = x1 + ta * nx
        y1_offset = y1 + ta * ny
        x2_offset = x2 + tb * nx
        y2_offset = y2 + tb * ny
        x1_opposite = x1 - ta * nx
        y1_opposite = y1 - ta * ny
        x2_opposite = x2 - tb * nx
        y2_opposite = y2 - tb * ny

        # Define the points for the quadrilateral
        points = [(x1_offset, y1_offset), (x2_offset, y2_offset),
                  (x2_opposite, y2_opposite), (x1_opposite, y1_opposite)]
        points = np.array([self.screen_scaled_point(point) for point in points])

        pygame.draw.polygon(self.screen, color, points)
        pygame.draw.circle(self.screen, self.WHITE, self.screen_scaled_point(line[0]),
                           5.4 * line_width(line[0]))
        pygame.draw.circle(self.screen, self.WHITE, self.screen_scaled_point(line[1]),
                           5.4 * line_width(line[1]))


# --------------------- FUNCTIONS FOR DEBUG SPECIFICATIONS -------------------

    def print_debug_info(self, current_tile, wall_direction_index, wall_segment, limits, front_left_point):
        pass

    def debug_labels(self, tile_center, string, second_string='', color=None, second_color=None):
        color = self.DEBUG_BLUE if color is None else color
        second_color = self.RED if second_color is None else second_color
        text1 = self.font.render(string, True, color)
        text2 = self.font.render(second_string, True, second_color)
        self.screen.blit(text1, self.screen_scaled_point(tile_center))
        self.screen.blit(text2, np.array([0, 10]) + self.screen_scaled_point(tile_center))

    def make_debug_lines(self):
        debug_lines = super().make_debug_lines()

        reference_00_indices = self.find_00_indices()
        debug_lines.append(f"Grid bin no. ({reference_00_indices[0]}, {reference_00_indices[1]})")

        return debug_lines


# ----------------------- STATIC FUNCTIONS WITH MORE MATH ------------------------

def get_circular_direction(circle, point, facing_angle):
    """
    Finds out if point with facing angle is facing clockwise (-1) or counter-clockwise (+1) around a circle.
    """
    point_to_c_vector = np.array(circle[0]) - point
    facing_vector = to_cartesian((1., facing_angle))
    cross_product = np.cross(point_to_c_vector, facing_vector)
    circular_direction = np.sign(cross_product)
    return circular_direction


def translate_along_circle(point, circle, distance):
    """
    Translates a point an amount of steps along a given circle or line.

    :param point: The point to translate.
    :type point: numpy.ndarray
    :param circle: (center, radius), the circle to follow (positive counter-clockwise).
    :type circle: tuple[numpy.ndarray, float]
    :param distance: Distance to translate measured in angle * radius.
    :returns new_point: The new point position.
    """

    center, radius = circle

    # Check for infinite radius (straight line case)
    if np.isinf(radius):
        # Translate along the line defined by the center and point
        direction = (point - center)
        direction_normalized = np.zeros_like(direction)
        direction_normalized[direction == np.inf] = 1
        direction_normalized[direction == -np.inf] = -1
        direction_normalized[np.isfinite(direction)] = 0
        new_point = point + distance * direction_normalized

        if np.linalg.norm(new_point) >= 1:
            new_point /= np.linalg.norm(new_point) - 1e-10

        return new_point, 0.
    else:
        # Handle the circular case
        # Convert point and center to complex numbers for easier angle calculation
        p_complex = complex(point[0], point[1])
        c_complex = complex(center[0], center[1])

        # Find the current angle of the point with respect to the center
        current_angle = np.angle(p_complex - c_complex)

        # Calculate the new angle after moving by the given distance
        translation_angle = current_angle + distance / radius

        # Calculate the new point position
        new_point_complex = c_complex + radius * np.exp(1j * translation_angle)
        new_point = np.array([new_point_complex.real, new_point_complex.imag])

        if np.linalg.norm(new_point) >= 1:
            new_point /= np.linalg.norm(new_point) - 1e-10

        return new_point, translation_angle


def linearization_estimation(x_reference, y_reference, y_target):
    """
    Using two points in a 2D-plane and the y-value for a third, finds the x-value of that point assuming linear correlation.
    """
    if x_reference[0] == y_reference[0]:  # Edge case
        return x_reference[0]

    x1, x2 = x_reference
    y1, y2 = y_reference
    x_target = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)

    return x_target


def bilinear_interpolation(reference_points, reference_targets, point):
    """
    Takes a point in a plane, reference square in the same plane, and a translated version of the reference square.

    :param reference_points: A reference square in the 2D-plane. The unit square would have the form: [[0, 0], [0, 1], [1, 0], [1, 1]]
    :param reference_targets: [p0, p1, p2, p3] translated from reference_points.
    :type reference_targets: list[list] or numpy.ndarray.
    :param point: Point = [x, y] in the frame of reference_points.
    :returns: Interpolated point following the same translation as reference_targets has done from reference_points.
    :rtype: list or numpy.ndarray.
    """

    if isinstance(reference_points, list):
        reference_points = np.array(reference_points)
    if isinstance(reference_targets, list):
        reference_targets = np.array(reference_targets)

    if reference_points.shape != (4, 2) or reference_targets.shape != (4, 2):
        raise ValueError(f"ERROR: Both reference and target matrices must be of shape (4, 2).")

    p0_ref, p1_ref, p2_ref, p3_ref = reference_points

    # Compute scaling factors in both x and y directions
    scale_x = np.linalg.norm(p2_ref - p0_ref)  # Distance between [0, 0] and [1, 0]
    scale_y = np.linalg.norm(p1_ref - p0_ref)  # Distance between [0, 0] and [0, 1]

    # Compute translation (shift from [0, 0])
    translation = p0_ref  # Reference point p0_ref is where [0, 0] in the unit square is mapped

    # ransform the point to the unit square coordinates
    point_in_unit_square = (point - translation) / [scale_x, scale_y]

    # Apply the bilinear interpolation in the target frame
    p0_tgt, p1_tgt, p2_tgt, p3_tgt = reference_targets
    x, y = point_in_unit_square

    # Bilinear interpolation to find the corresponding point in the target frame
    target_point = (1 - x) * (1 - y) * p0_tgt + (1 - x) * y * p1_tgt + x * (1 - y) * p2_tgt + x * y * p3_tgt

    return target_point


def rotate_normal(point, normal):
    """
    Rotates a normal vector 90 degrees always towards the origin.
    """
    if np.cross(point, normal) < 0:  # Clockwise
        return np.array([normal[1], -normal[0]])
    else:  # Counter-clockwise
        return np.array([-normal[1], normal[0]])


def find_normal(point, circle):
    """
    :param point: A point on the circle.
    :param circle: [center, radius], circle we're looking for the normal to.
    :returns: A vector representing a normal to the circle at given point.
    """
    center = circle[0]
    direction_vec = point - center
    norm = np.linalg.norm(direction_vec)
    if math.isinf(norm):
        for i in range(len(direction_vec)):
            if direction_vec[i] == float('inf'):
                direction_vec[i] = 1.
            elif direction_vec[i] == float('-inf'):
                direction_vec[i] = -1.
            else:
                direction_vec[i] = 0.
        new_normal = direction_vec
    else:
        new_normal = direction_vec / np.linalg.norm(direction_vec)

    return new_normal


def find_circle(point, normal):
    """
    Finds a circle that touches the given point within the unit circle and crosses the unit circle at right angles.

    :param point: Coordinates of the point inside the unit circle.
    :type point: np.ndarray
    :param normal: The normal vector at the point.
    :type normal: np.ndarray
    :return: The center and radius of the circle.
    """
    denominator = 2 * np.dot(point, normal)

    if pretty_much_0(denominator):
        # Special case: when the normal is perpendicular to the radius at 'point'
        if pretty_much_0(point[1]) and pretty_much_0(normal[0]):
            # Case of a straight line aligned with the x-axis
            if normal[1] > 0:  # Normal points upward, so center is at -inf
                center = np.array([point[0], -np.inf])
            else:  # Normal points downward, so center is at +inf
                center = np.array([point[0], np.inf])
            radius = np.inf  # Infinite radius for a line
        elif pretty_much_0(point[0]) and pretty_much_0(normal[1]):
            # Case of a straight line aligned with the y-axis
            if normal[0] > 0:  # Normal points to the right, so center is at -inf
                center = np.array([-np.inf, point[1]])
            else:  # Normal points to the left, so center is at +inf
                center = np.array([np.inf, point[1]])
            radius = np.inf  # Infinite radius for a line
        else:
            # If no specific case, the result is 0
            center = np.array([0, 0])
            radius = 0
    else:
        # General case
        nominator = 1 - np.linalg.norm(point) ** 2
        result = nominator / denominator * normal
        center = point + result
        radius = np.linalg.norm(point - center)

    return center, radius

def pretty_much_0(val):
    """
    Since floating point values tend to cause some rounding errors, this function is here to replace 'val == 0.'.
    """
    return np.abs(val) < 0.00001

def line_width(point):
    """
    draw_thicker_line() adjusts thickness based on the edges' distances from the center. Here's how.
    """
    norm = np.linalg.norm(point)
    if norm > 1.:
        return 0.

    # Spherical function instead of hyperbolic, because it looks cool. Like circulating around a planet.
    sin = np.sqrt(1 - np.power(norm, 2))
    return sin

def to_polar(p_cartesian):
    x, y = p_cartesian
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return np.array([r, phi])


def to_cartesian(p_polar):
    r, phi = p_polar
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.array([x, y])


# ----------------- FILE LOADING --------------------

def load_grid_dict(filename):
    with open(f'SavedModels/{filename}.json', 'r') as json_file:
        loaded_dict = json.load(json_file)

    for key, value in loaded_dict.items():
        loaded_dict[key] = np.array(value)

    return loaded_dict


# ------------------- TESTING THE MAP ---------------------

if __name__ == '__main__':
    Game.run_game(default_render="MiniMap", maze=DynamicMaze.get_plain_map(6, 'walls+'))
