import math
import pygame
import numpy as np

import Explorer
import config
from Rendering3D import Rendering3D


class Rendering2D(Rendering3D):
    """
    The first visualization tool used for the project. This class has been reworked a lot since it was first written,
    but it is still valuable for its ability to draw the maze from a top-view Euclidean perspective.
    """

    def __init__(self, dynamic_maze, explorer, placement='center'):
        super().__init__(dynamic_maze, explorer, title='Top ray view')
        self.NUM_RAYS = 3
        self.SQUARE_SIZE = config.tile_size
        self.SQUARE_COLOR = self.WHITE
        self.BG_COLOR = self.BLACK
        self.WALL_COLOR = (60, 60, 60)
        self.TEXT_COLOR = self.BLACK
        self.DOT_COLOR = self.WHITE  # Previously: (0, 200, 100)
        self.PLAYER_COLOR = self.RED
        self.RAY_COLOR = (150, 150, 150)
        self.DOT_SIZE = config.player_radius

        self.full_screen = False
        self.map_size_on_screen = config.mini_map_size
        if placement == 'center':
            self.full_screen = True
            self.map_size_on_screen = np.min(self.SCREEN_SIZE)
            self.map_center = self.SCREEN_SIZE // 2
        else:
            raise ValueError(f"ERROR: I forgot to implement code for screen placement {placement}.")

        self.player_center = (self.SCREEN_SIZE * np.array([0.5, 0.75])).astype(int)

        self.screen.fill(self.BG_COLOR)
        self.drawn_tiles = set()
        self.update()


    def update(self):
        if self.full_screen:
            self.screen.fill(self.BG_COLOR)
        # self.maze.update_visibility(self.explorer.pos_tile)
        self.drawn_tiles = set()
        # self.update_recursive(tile=self.explorer.pos_tile, prev_tile=None, screen_position=np.array([0, 0]))
        self.draw_view_field(only_draw_limit_rays=True)

        self.draw_spotted_corners()
        pygame.draw.circle(self.screen, self.PLAYER_COLOR, tuple(self.player_center), self.DOT_SIZE)
        #self.write_debug_info()
        pygame.display.flip()
        pygame.display.update()

    def recursive_tile_drawer(self, tile, prev_tile, screen_position):
        if tile not in self.maze.visible_tiles or tile in self.drawn_tiles:
            return

        # Define some useful parameters
        rotation_radians = math.radians(self.explorer.rotation - config.initial_rotation)
        rotation_degrees = self.explorer.rotation - config.initial_rotation
        rotation_matrix = np.array([[math.cos(rotation_radians), -math.sin(rotation_radians)],
                                    [math.sin(rotation_radians), math.cos(rotation_radians)]])
        flip_y = np.array([1, -1])

        # Draw the tile
        square_center = -flip_y*self.explorer.pos + screen_position * self.SQUARE_SIZE + flip_y*self.SQUARE_SIZE//2
        square_center = np.dot(rotation_matrix, square_center)
        square_center = square_center + self.player_center
        draw_square(self.screen, tuple(square_center), rotation_degrees, (self.SQUARE_SIZE, self.SQUARE_SIZE), self.SQUARE_COLOR)

        # Label the tile
        text = self.font.render(tile, True, self.TEXT_COLOR)
        self.screen.blit(text, (square_center[0], square_center[1]))

        self.drawn_tiles.add(tile)

        shifts = np.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        for i in range(4):
            if self.maze.check_wall_with_placement(tile, i):
                continue
                # Draw walls
                #wall_x, wall_y, wall_w, wall_h = self.where_wall(i, square_center)
                #wall_center = (wall_x + wall_w//2 - self.SQUARE_SIZE//2, wall_y + wall_h//2 - self.SQUARE_SIZE//2)
                # self.draw_square(wall_center, rotation_degrees, (wall_w, wall_h), self.WALL_COLOR)
            else:  # Else call drawing function for surrounding tiles.
                global_index = self.explorer.global_index_to(i)  # To be replaced with the probe strategy.
                neighbor = self.maze.adjacency_map[tile][global_index]
                if neighbor == prev_tile:
                    continue
                self.recursive_tile_drawer(tile=neighbor, prev_tile=tile, screen_position=(screen_position + shifts[i]))


    def draw_spotted_corners(self):
        self.draw_walls()

    def draw_wall_segment(self, wall_segment):
        flip_y = np.array([1, -1])
        xy0 = flip_y * self.to_cartesian([wall_segment[0][0], wall_segment[0][
            1] - self.explorer.rotation + config.initial_rotation]) + self.player_center
        xy1 = flip_y * self.to_cartesian([wall_segment[1][0], wall_segment[1][
            1] - self.explorer.rotation + config.initial_rotation]) + self.player_center
        xy0, xy1 = tuple(xy0), tuple(xy1)
        pygame.draw.line(self.screen, self.DOT_COLOR, xy0, xy1, 2*config.wall_thickness)

        self.draw_corner_rays(xy0, xy1)
        #self.draw_top_view_corner_dots(wall_segment, xy0, xy1, False)

    def draw_corner_rays(self, xy0, xy1):
        pygame.draw.line(self.screen, self.RAY_COLOR, xy0, self.player_center)
        pygame.draw.line(self.screen, self.RAY_COLOR, xy1, self.player_center)

    def draw_top_view_corner_dots(self, wall_segment, xy0, xy1, include_labels=True):
        internal0 = wall_segment[0]
        internal1 = wall_segment[1]
        pygame.draw.circle(self.screen, self.DOT_COLOR, xy0, 4)
        pygame.draw.circle(self.screen, self.DOT_COLOR, xy1, 4)
        if include_labels:
            formatted_strings0 = [format(f, '.1f') for f in internal0]
            label0 = "(" + ', '.join(formatted_strings0) + ")"
            formatted_strings1 = [format(f, '.1f') for f in internal1]
            label1 = "(" + ', '.join(formatted_strings1) + ")"
            text0 = self.font.render(label0, True, (0, 100, 50))
            text1 = self.font.render(label1, True, (0, 100, 50))
            self.screen.blit(text0, xy0)
            self.screen.blit(text1, xy1)

    def draw_all_squares(self):
        self.recursive_tile_drawer(tile=self.explorer.pos_tile, prev_tile=None, screen_position=np.array([0, 0]))

    def draw_view_field(self, only_draw_limit_rays=False):
        left_limit = self.explorer.rotation + config.camera_span[0]/2
        right_limit = self.explorer.rotation - config.camera_span[0]/2
        ray_directions = np.linspace(left_limit, right_limit, self.NUM_RAYS)
        ray_distances = np.empty_like(ray_directions)
        for i in range(len(ray_directions)):
            ray = Explorer.Ray(self.explorer)
            ray_distances[i] = ray.shoot(self.maze, ray_directions[i], True)
        #self.draw_all_squares()
        for i in range(len(ray_directions)):
            if only_draw_limit_rays and i != 0 and i != len(ray_directions)-1:
                continue
            self.draw_overview_line(ray_directions[i], ray_distances[i])

    def draw_overview_line(self, direction, distance):
        end_point = (self.player_center[0] + distance * math.cos(math.radians(direction - self.explorer.rotation + 90)),
                     self.player_center[1] + distance * math.sin(math.radians(direction - self.explorer.rotation - 90)))
        pygame.draw.line(self.screen, self.RAY_COLOR, tuple(self.player_center), end_point, 1)


def draw_square(screen, center, rotation, size, color):
    rect = pygame.Surface(size)
    rect.fill(color)
    square = pygame.Surface((size[0], size[1]), pygame.SRCALPHA)
    pygame.draw.rect(square, color, (0, 0, size[0], size[1]))
    rot_image = pygame.transform.rotate(square, 360-rotation)
    rot_rect = rot_image.get_rect(center=center)
    screen.blit(rot_image, rot_rect)
