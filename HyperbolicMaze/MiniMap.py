import json
import math
from collections import deque
import pygame
import sys
import numpy as np
import DynamicMaze
import HyperbolicGrid
import Rendering2D
import TrainingDataGenerator
import config
from Explorer import Explorer
from Explorer import Player
from Rendering import Rendering


class MiniMap(Rendering):
    exploration_directions = [Explorer.FORWARD, Explorer.RIGHT, Explorer.LEFT]
    d = 1.06
    visible_range = 1.
    tile_size = 0.5

    def __init__(self, dynamic_maze, explorer, map_size=600, placement='center'):
        super().__init__('Poincaré map', dynamic_maze, explorer)
        self.map_size_on_screen = map_size
        if placement == 'center':
            self.map_center = (self.SCREEN_SIZE // 2) * np.ones(2)
        else:
            raise ValueError(f"ERROR: I forgot to implement code for screen placement {placement}.")

        self.grid_map_dict = load_grid_dict(f'GridMapDict{config.num_grid_bins}x{config.num_grid_bins}')

    def update(self):
        self.screen.fill(self.BLACK)
        self.hyperbolic_map_display()
        # self.all_circles_display()

        self.write_debug_info()
        pygame.display.flip()
        pygame.display.update()

    def all_circles_display(self):
        collection_set = TrainingDataGenerator.find_coordinates_and_circles(self.maze, self.explorer, self.get_tile_center_map_placement())

        pygame.draw.circle(self.screen, self.WHITE, self.map_center, self.map_size_on_screen // 2, 1)
        square_height = self.screen_scaled_distance(self.tile_size) - 2 * config.wall_thickness
        Rendering2D.draw_square(self.screen, self.screen_scaled_point(self.get_tile_center_map_placement()), (self.explorer.rotation - config.initial_rotation),
                                (square_height, square_height), (50, 50, 50))

        for relative_tile_key, (tile_point, (circle_center, circle_radius)) in collection_set.items():

            # Circle path first
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
            pygame.draw.circle(self.screen, self.WHITE, self.screen_scaled_point(tile_point), 5)
            text = self.font.render(relative_tile_key, True, self.DEBUG_BLUE)
            self.screen.blit(text, self.screen_scaled_point(tile_point))

        # Point representing the player
        pygame.draw.circle(self.screen, self.RED, self.screen_scaled_point([0, 0]), 5)


    def hyperbolic_map_display(self):
        step_set = self.find_steps()

        # Background
        #pygame.draw.circle(self.screen, self.BLACK, self.map_center, self.map_size_on_screen // 2, width=0)
        pygame.draw.circle(self.screen, self.WHITE, self.map_center, self.map_size_on_screen // 2, 1)

        for step in step_set:
            pygame.draw.line(self.screen, self.WHITE, self.screen_scaled_point(step[0]), self.screen_scaled_point(step[1]), 2)

        # Point representing the player
        pygame.draw.circle(self.screen, self.RED, self.screen_scaled_point([0, 0]), 5)


    def find_steps(self):
        all_steps = set()
        visited = {}
        probe = self.explorer.__copy__()
        reference_00_indices = self.find_00_indices()

        queue = deque([[probe, all_steps, visited, reference_00_indices, '']])

        while queue:
            tile_pack = queue.popleft()
            self.find_tile_specific_step(queue, *tile_pack)

        return all_steps


    def find_tile_specific_step(self, queue, probe, all_steps, visited, reference_00_indices, relative_journey):

        tile_center = self.get_estimated_tile_center(reference_00_indices, self.grid_map_dict[relative_journey])

        # Discard iteration if we're outside the visible range
        if np.linalg.norm(tile_center) > self.visible_range:
            return

        visited[probe.pos_tile] = tile_center

        for brfl in range(4):
            probe_ahead = probe.__copy__()
            probe_ahead.directional_tile_step(self.maze, brfl)
            journey_ahead = relative_journey + Explorer.relative_directions[brfl]
            # Add the step if neighboring tile has been visited
            if probe_ahead.pos_tile in visited:
                all_steps.add((tuple(tile_center), tuple(visited[probe_ahead.pos_tile])))
                # DEBUG TEXT
                text = self.font.render(relative_journey, True, self.DEBUG_BLUE)
                self.screen.blit(text, self.screen_scaled_point(tile_center))
            # Otherwise add neighbor to the queue assuming we've mapped positions for it.
            elif journey_ahead in self.grid_map_dict:
                queue.append([probe_ahead, all_steps, visited, reference_00_indices, journey_ahead])



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


    def get_estimated_tile_center(self, reference_00_indices, grid_mat):
        """

        :param reference_00_indices:
        :param grid_mat:
        :return:
        """
        player_tile_center = self.get_tile_center_map_placement()
        i, j = reference_00_indices
        n = config.num_grid_bins
        reference_targets = grid_mat[i:(i + 2), j:(j + 2)]
        reference_targets = reference_targets.reshape((4, 2))
        reference_00_point = self.tile_size * (-0.5 + np.array([i / n, j / n]))
        reference_points = reference_00_point + self.tile_size / n * np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
        tile_center = bilinear_interpolation(reference_points, reference_targets, player_tile_center)

        return tile_center


    def get_tile_center_map_placement(self, player_unit_pos=None):
        """
        :param unit_pos: Point within the unit square that represents the player's position.
        :returns: The unscaled position in the unit circle of the center of the tile the player is standing in.
        """
        if player_unit_pos is None:
            player_unit_pos = self.explorer.pos/config.tile_size
        elif player_unit_pos.any() < 0. or player_unit_pos.any() > 1.:
            raise ValueError(f"ERROR: Point {player_unit_pos} outside the unit square.")

        return self.tile_size * (0.5 - player_unit_pos)

    def get_tile_center_unit_square_placement(self):
        """
        :returns: The position of the tile center inside the unit square, given the player position.
        """
        return 1. - self.explorer.pos/config.tile_size

    def screen_scaled_point(self, point):
        rotation_degrees = np.radians(360. - self.explorer.rotation + config.initial_rotation)
        rotation_matrix = np.array([[np.cos(rotation_degrees), -np.sin(rotation_degrees)],
                                    [np.sin(rotation_degrees), np.cos(rotation_degrees)]])
        new_point = np.matmul(rotation_matrix, point)
        new_point = self.map_center + (self.map_size_on_screen // 2) * np.array([new_point[0], -new_point[1]])
        return new_point

    def screen_scaled_distance(self, distance):
        return (self.map_size_on_screen // 2) * distance


    def draw_step(self, step_vector):
        x1, y1 = step_vector[0]
        x2, y2 = step_vector[1]

        ta = line_width(np.linalg.norm(step_vector[0]))
        tb = line_width(np.linalg.norm(step_vector[1]))

        y1 *= -1
        y2 *= -1

        # Calculate the direction vector of the line
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy)
        if length == 0:
            return  # Avoid division by zero

        # Normalize direction vector
        dx /= length
        dy /= length

        # Calculate the perpendicular vector (normal) for the thickness
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
        points = np.array([(x1_offset, y1_offset), (x2_offset, y2_offset),
                           (x2_opposite, y2_opposite), (x1_opposite, y1_opposite)])

        points = self.screen_placement + (self.map_size_on_screen // 2) * points

        # Draw the quadrilateral
        pygame.draw.polygon(self.screen, (255, 255, 255), points)
        # FOR DEBUGGING
        pygame.display.flip()


    def print_debug_info(self, current_tile, wall_direction_index, wall_segment, limits, front_left_point):
        pass

    def make_debug_lines(self):
        debug_lines = super().make_debug_lines()

        reference_00_indices = self.find_00_indices()
        debug_lines.append(f"Grid bin no. ({reference_00_indices[0]}, {reference_00_indices[1]})")

        return debug_lines




def line_width(norm):
    if norm > 1.:
        raise ValueError("ERROR: Some point has sneaked out from the unit circle!")
    sin = np.sqrt(1 - np.power(norm, 2))  # Spherical function. Should probably change to hyperbolic.
    return sin / 30.


def get_circular_direction(circle, point, facing_angle):
    """
    Finds out if point with facing angle is facing clockwise (-1) or counter-clockwise (+1) around the circle.
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
    if x_reference[0] == y_reference[0]:  # Edge case
        return x_reference[0]

    x1, x2 = x_reference
    y1, y2 = y_reference
    x_target = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)

    return x_target


def bilinear_interpolation(reference_points, reference_targets, point):
    """
    Assumes the standard points [[0, 0], [0, 1], [1, 0], [1, 1]] are translated to reference_targets, as well as linear
    relation, and uses that to estimate a translated coordinate of point.

    :param reference_targets: [p0, p1, p2, p3] translated from reference_points.
    :type reference_targets: list[list] or numpy.ndarray.
    :param reference_points:
    :param point: Point = [x, y] to be translated.
    :return: Estimated translated point using the same linear transformation.
    :rtype: list or numpy.ndarray.
    """

    if isinstance(reference_points, list):
        reference_points = np.array(reference_points)
    if isinstance(reference_targets, list):
        reference_targets = np.array(reference_targets)

    if reference_points.shape != (4, 2) or reference_targets.shape != (4, 2):
        raise ValueError(f"ERROR: Both reference and target matrices must be of shape (4, 2).")

    p0_ref, p1_ref, p2_ref, p3_ref = reference_points

    # Step 1: Compute scaling factors in both x and y directions
    scale_x = np.linalg.norm(p2_ref - p0_ref)  # Distance between [0, 0] and [1, 0]
    scale_y = np.linalg.norm(p1_ref - p0_ref)  # Distance between [0, 0] and [0, 1]

    # Step 2: Compute translation (shift from [0, 0])
    translation = p0_ref  # Reference point p0_ref is where [0, 0] in the unit square is mapped

    # Step 3: Transform the point to the unit square coordinates
    point_in_unit_square = (point - translation) / [scale_x, scale_y]

    # Step 4: Apply the bilinear interpolation in the target frame
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
    #if np.linalg.norm(point) >= 1:
    #    raise ValueError(f"ERROR: The point must be within the unit circle. {point} is an invalid location.")

    # normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

    # Calculate the denominator
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
    return np.abs(val) < 0.00001


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



# -------------- RETIRED FUNCTIONS ------------
def get_step_distance(point, facing_angle, step_size):
    p_r, p_phi = to_polar(point)
    theta = facing_angle - p_phi
    dzdr = 2 / (1 - p_r ** 2)
    acceleration = dzdr * np.cos(theta)
    step_distance = step_size * np.exp(acceleration)  # Current step function. Let's see how it works.
    return step_distance




# ------------------- TESTING THE MAP ---------------------

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    maze = DynamicMaze.DynamicMaze()
    explorer = Player()
    mini_map = MiniMap(maze, explorer)

    HyperbolicGrid.bulk_registration(maze.adjacency_map, "", 2)
    for key in maze.adjacency_map:
        if maze.adjacency_map[key] is not None:
            maze.wall_map[key] = [1, 1, 1, 1]  # Making a map with no walls


    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        mini_map.find_steps()

        pygame.display.flip()
