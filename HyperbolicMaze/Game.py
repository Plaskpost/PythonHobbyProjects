import time

import pygame
import numpy as np
import multiprocessing
import config

from DynamicMaze import DynamicMaze
from Rendering2D import Rendering2D
from Rendering3D import Rendering3D
from Explorer import Player


def run_game():
    if config.fixed_seed:
        np.random.seed(config.seed)
    explorer = Player()
    maze = DynamicMaze(explorer.pos_tile)
    renderer = Rendering2D(maze, explorer)
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
            explorer.rotate(left=True, amount=config.rotation_speed)
            pressed = True
        if keys[pygame.K_RIGHT] or keys[pygame.K_KP6]:
            explorer.rotate(left=False, amount=config.rotation_speed)
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

