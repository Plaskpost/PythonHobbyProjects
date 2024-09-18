import math
from collections import deque
import pygame
import sys
import numpy as np

import DynamicMaze
import HyperbolicGrid
from Explorer import Explorer
from Explorer import Player


class MiniMap:
    WHITE = (255, 255, 255)

    def __init__(self, screen, screen_placement, size_on_screen):
        self.screen = screen
        self.screen_placement = screen_placement  # Center of map
        self.map_size_on_screen = size_on_screen
        self.tile_size = size_on_screen // 5
        self.exploration_directions = [Explorer.FORWARD, Explorer.RIGHT, Explorer.LEFT]

        self.d = 0.1
        self.visible_range = 0.99

        pygame.draw.circle(screen, self.WHITE, screen_placement, size_on_screen // 2, 1)

    def display_map(self):
        steps = self.find_steps()
        for step in steps:
            self.draw_step(step)

    def find_all_steps(self, maze, explorer):
        all_steps = []
        visited = set()

        probe = explorer.__copy__()

        point = np.array([0.1, 0.])
        normal = np.array([-1., 0.])
        facing_angle = np.pi / 2.
        circle = find_circle(point, normal)

        queue = deque([[probe, maze, all_steps, visited, circle, point, -1, facing_angle]])

        while queue:
            tile_pack = queue.popleft()
            self.find_tile_specific_steps(queue, *tile_pack)

        return all_steps

    def find_tile_specific_steps(self, queue, probe, maze, all_steps, visited, current_circle, projection_coord,
                                 circular_direction, facing_angle):
        """
        Loops through three of the tile's neighbors in order: forward, right, left (previous tile as reference), finds
        the coordinated of the blocks to draw and adds neighboring tiles to the queue.

        :param queue: Queue with argument lists as elements that lines up the tiles to be searched in BFS order.
        :param maze: DynamicMaze object.
        :param probe: Explorer object to maintain correct orientation in the maze
        :param all_steps: A set of steps (point1, point2) to save all the visible steps (tile->tile).
        :param current_circle: (center, radius). Circle along which the probe is currently searching.
        :param visited: A set of visited tiles.
        :param projection_coord:
        :type projection_coord: numpy.ndarray
        :param circular_direction: Which way is forward on the current circle. 1 for counter-clockwise and -1 for clockwise.
        :param facing_angle: Angle of the direction forward along the current circle arc.
        :return:
        """

        print(f"Exploring {probe.pos_tile}")
        if probe.pos_tile == 'U':
            a = 0
        visited.add(probe.pos_tile)

        # Find the circle perpendicular to the current one
        current_normal = find_normal(projection_coord, current_circle)
        perpendicular_normal = rotate_normal(projection_coord, current_normal)
        perpendicular_circle = find_circle(projection_coord, perpendicular_normal)

        circle = current_circle

        for exploration_direction in self.exploration_directions:
            probe_ahead = probe.__copy__()
            if not probe_ahead.directional_tile_step(maze, exploration_direction):
                continue  # Skip any step towards a tile that hasn't been generated in the maze.

            if exploration_direction == Explorer.FORWARD:
                # Continue the translation forward without finding new circle and all that.
                distance = get_step_distance(projection_coord, facing_angle, self.d)
                projection_coord_ahead, translation_angle = translate_along_circle(projection_coord, circle, circular_direction*distance)

            else:  # If we're turning to a sideways direction
                circle = perpendicular_circle
                if exploration_direction == Explorer.RIGHT:
                    facing_angle -= np.pi / 2
                    # Figure out if we're following this circle clockwise or counter-clockwise
                    point_to_c_vector = np.array(circle[0]) - projection_coord
                    facing_vector = to_cartesian((1., facing_angle))
                    cross_product = np.cross(point_to_c_vector, facing_vector)
                    circular_direction = np.sign(cross_product)
                else:  # Left.
                    facing_angle += np.pi  # Adding half a lap as this angle was the other way around in the last loop.
                    circular_direction *= -1  # Same here, we re-use the computation from last time.

                # Translate the point along the circle
                distance = get_step_distance(projection_coord, facing_angle, self.d)
                projection_coord_ahead, translation_angle = translate_along_circle(projection_coord, perpendicular_circle, circular_direction*distance)

            # If wall passable, add step to the list of all steps.
            if maze.wall_map[probe_ahead.pos_tile][probe_ahead.global_index_to_previous_tile] == 1:  # If passable
                all_steps.append((projection_coord, projection_coord_ahead))
                # STEP-WISE DRAWING FOR DEBUGGING
                inverted_circle_center = np.array([circle[0][0], -circle[0][1]])
                pygame.draw.circle(screen, self.WHITE,
                                   self.screen_placement + (self.map_size_on_screen / 2.) * inverted_circle_center,
                                   (self.map_size_on_screen / 2.) * circle[1], 1)
                self.draw_step((projection_coord, projection_coord_ahead))
                pygame.time.wait(10)

            # Lastly, add neighbors to queue
            if np.linalg.norm(projection_coord_ahead) < self.visible_range and \
                    maze.wall_map[probe_ahead.pos_tile][probe_ahead.global_index_to_previous_tile] != 0 and \
                    probe_ahead.pos_tile not in visited:
                facing_angle_ahead = facing_angle + translation_angle  # First update the facing angle.
                queue.append([probe_ahead, maze, all_steps, visited, circle, projection_coord_ahead,
                               circular_direction, facing_angle_ahead])





    def draw_step(self, step_vector):
        x1, y1 = step_vector[0]
        x2, y2 = step_vector[1]

        ta = self.line_width(np.linalg.norm(step_vector[0]))
        tb = self.line_width(np.linalg.norm(step_vector[1]))

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

    def line_width(self, norm):
        if norm > 1.:
            raise ValueError("ERROR: Some point has sneaked out from the unit circle!")
        sin = np.sqrt(1 - np.power(norm, 2))
        return sin / 30.


def get_step_distance(point, facing_angle, step_size):
    p_r, p_phi = to_polar(point)
    theta = facing_angle - p_phi
    dzdr = 2 / (1 - p_r ** 2)
    acceleration = dzdr * np.cos(theta)
    step_distance = step_size * np.exp(acceleration)  # Current step function. Let's see how it works.
    return step_distance



def translate_along_circle(point, circle, distance):
    """
    Translates a point an amount of steps along a given circle or line.

    :param point: The point to translate.
    :type point: numpy.ndarray
    :param circle: (center, radius), the circle to follow (positive counter-clockwise).
    :type circle: tuple[numpy.ndarray, float]
    :param distance: Distance to translate measured in angle / radius.
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
        translation_angle = current_angle + distance * radius

        # Calculate the new point position
        new_point_complex = c_complex + radius * np.exp(1j * translation_angle)
        new_point = np.array([new_point_complex.real, new_point_complex.imag])

        if np.linalg.norm(new_point) >= 1:
            new_point /= np.linalg.norm(new_point) - 1e-10

        return new_point, translation_angle


def rotate_normal(point, normal):
    """
    Rotates a normal vector 90 degrees always towards the origin.

    :param point:
    :param normal:
    :return:
    """
    if np.cross(point, normal) < 0:
        # Rotate clockwise
        return np.array([normal[1], -normal[0]])
    else:
        # Rotate counterclockwise
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
    if np.linalg.norm(point) >= 1:
        raise ValueError(f"ERROR: The point must be within the unit circle. {point} is an invalid location.")

    # normal = normal / np.linalg.norm(normal)  # Normalize the normal vector

    # Calculate the denominator
    denominator = 2 * np.dot(point, normal)

    if denominator == 0:
        # Special case: when the normal is perpendicular to the radius at 'point'
        if point[1] == 0 and normal[0] == 0:
            # Case of a straight line aligned with the x-axis
            if normal[1] > 0:  # Normal points upward, so center is at -inf
                center = np.array([point[0], -np.inf])
            else:  # Normal points downward, so center is at +inf
                center = np.array([point[0], np.inf])
            radius = np.inf  # Infinite radius for a line
        elif point[0] == 0 and normal[1] == 0:
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


# ------------------- TESTING THE MAP ---------------------

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    mini_map = MiniMap(screen, (300, 300), 600)

    maze = DynamicMaze.DynamicMaze()
    HyperbolicGrid.bulk_registration(maze.adjacency_map, "", 2)
    for key in maze.adjacency_map:
        if maze.adjacency_map[key] is not None:
            maze.wall_map[key] = [1, 1, 1, 1]  # Making a map with no walls

    explorer = Player()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        mini_map.find_all_steps(maze, explorer)

        pygame.display.flip()
