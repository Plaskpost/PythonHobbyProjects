import math

import pygame
import sys
import numpy as np


class MiniMap:

    def __init__(self, screen, screen_placement):
        self.screen = screen
        self.screen_placement = screen_placement  # Center of map
        self.d = 0.1
        self.max_depth = 5


    def display_map(self):
        steps = self.find_steps()
        for step in steps:
            ta = np.linalg.norm(step[0])
            tb = np.linalg.norm(step[1])
            self.draw_step(step[0], step[1], ta, tb)


    def find_steps(self, maze, explorer):
        steps = []
        probe = explorer.__copy__()


        return steps

    def find_steps_recursive(self, maze, probe, steps, depth):
        

    def draw_step(self, a, b, ta, tb):
        # Unpack points
        x1, y1 = a
        x2, y2 = b

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
        points = [(x1_offset, y1_offset), (x2_offset, y2_offset),
                  (x2_opposite, y2_opposite), (x1_opposite, y1_opposite)]

        # Draw the quadrilateral
        pygame.draw.polygon(self.screen, (255, 255, 255), points)

    def translate_along_circle(self, point, circle, steps):
        """
        Translates point an amount of steps along a given circle

        :param point: The point to translate.
        :type point: numpy.ndarray
        :param circle: (center, radius), the circle to follow (positive counter-clockwise).
        :type circle: tuple[numpy.ndarray]
        :param steps: How many steps (negative allowed) of size self.d to translate.
        :type steps: int
        :returns new_point: The new point position.
        :returns new_normal: Nhe normal to circle at new_point.
        """

        center, radius = circle

        # Convert point and center to complex numbers for easier angle calculation
        p_complex = complex(point[0], point[1])
        c_complex = complex(center[0], center[1])

        apparent_distance = steps * self.d * (1 - np.linalg.norm(point))
        angle = 2 * np.arctan(np.tanh(apparent_distance / 2))

        # Find the current angle of the point with respect to the center
        current_angle = np.angle(p_complex - c_complex)

        # Calculate the new angle after moving by the given distance
        new_angle = current_angle + angle

        # Calculate the new point position
        new_point_complex = c_complex + radius * np.exp(1j * new_angle)
        new_point = np.array([new_point_complex.real, new_point_complex.imag])

        if np.linalg.norm(new_point) >= 1:
            new_point /= np.linalg.norm(new_point) - 1e-10

        tangent = (new_point - center) / np.linalg.norm(new_point - center)
        new_normal = np.array([-tangent[1], tangent[0]])

        return new_point, new_normal

def find_circle(point, normal):
    """
    Finds a circle that touches point within the unit circle, and crosses the unit circle at right angles.

    :param point:
    :param normal:
    :return:
    """
    if np.linalg.norm(point) >= 1:
        raise ValueError(f"ERROR: The point must be within the unit circle. {point} is an invalid location.")

    normal = normal / np.linalg.norm(normal)

    # Finding the correct circle center
    center = point + (1 - np.linalg.norm(point) ** 2) / (2 * np.dot(point, normal)) * normal

    center = np.array(center)
    radius = np.linalg.norm(point - center)

    return center, radius


# ------------------- TESTING THE MAP ---------------------

if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    mini_map = MiniMap((0, 0))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        mini_map.draw_step(screen, (100, 100), (300, 300), 20, 100)

        pygame.display.flip()
