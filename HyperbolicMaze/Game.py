import time

import pygame
import numpy as np
import multiprocessing

from DynamicMaze import DynamicMaze
from Rendering2D import Rendering2D
from Rendering3D import Rendering3D
from Explorer import Player


# Constants
DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

movement_speed = 1.4
rotation_speed = 1.3  # Ok degrees
tile_size = 80
wall_thickness = 5  # *2
player_radius = 10
average_walls_per_tile = 2.0
seed = 44
fixed_seed = True


def run_game():
    if fixed_seed:
        np.random.seed(seed)
    explorer = Player(movement_speed, rotation_speed, tile_size, wall_thickness, player_radius)
    maze = DynamicMaze(explorer.pos_tile, average_walls_per_tile)
    renderer = Rendering3D(maze, explorer)
    renderer.update()

    flip = False
    printt = False  # I hate this strategy
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # I think this section can look better.
        keys = pygame.key.get_pressed()
        pressed = False

        if keys[pygame.K_UP] or keys[pygame.K_KP8]:
            explorer.move(maze=maze, flbr=0)
            pressed = True
        if keys[pygame.K_DOWN] or keys[pygame.K_KP5]:
            explorer.move(maze=maze, flbr=2)
            pressed = True
        if keys[pygame.K_KP1]:
            explorer.move(maze=maze, flbr=1)
            pressed = True
        if keys[pygame.K_KP3]:
            explorer.move(maze=maze, flbr=3)
            pressed = True
        if keys[pygame.K_LEFT] or keys[pygame.K_KP4]:
            explorer.rotate(left=True, amount=rotation_speed)
            pressed = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_KP6]:
            explorer.rotate(left=False, amount=rotation_speed)
            pressed = True

        # Switch visualizing mode with 'keypad 0'
        if keys[pygame.K_KP0]:
            flip = True
        elif flip and not keys[pygame.K_KP0]:
            if isinstance(renderer, Rendering3D):
                renderer = Rendering2D(maze, explorer)
            else:
                renderer = Rendering3D(maze, explorer)
            pressed = True
            flip = False

        # Print the whole adjacency_list with 'p'
        if keys[pygame.K_p]:
            printt = True
        elif printt and not keys[pygame.K_p]:
            print("Full adjacency map:")
            for tile, adjacents in maze.adjacency_map.items():
                print(tile, ":", adjacents)
            printt = False

        if pressed:
            renderer.update()

    pygame.quit()

if __name__ == '__main__':
    run_game()

