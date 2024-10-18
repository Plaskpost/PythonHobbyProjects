import threading
import time

import pygame
import numpy as np
import config

import MiniMap
import DynamicMaze
from Rendering2D import Rendering2D
from Rendering3D import Rendering3D
from Explorer import Player


def run_game(default_render="3D", maze=None):
    if config.fixed_seed:
        np.random.seed(config.seed)
    explorer = Player()
    if maze is None:
        maze = DynamicMaze.DynamicMaze()
    if default_render == "3D":
        renderer = Rendering3D(maze, explorer, True)
    elif default_render == "2D":
        renderer = Rendering2D(maze, explorer)
    elif default_render == "MiniMap":
        renderer = MiniMap.MiniMap(maze, explorer)
    else:
        raise ValueError(f"ERROR: {default_render} is not a valid renderer.")
    renderer.update()

    last_render_time = time.time()
    running = True

    def render_in_background():
        nonlocal last_render_time
        while running:
            current_time = time.time()
            if current_time - last_render_time >= config.render_interval:
                renderer.update()
                pygame.display.flip()
                pygame.display.update()
                last_render_time = current_time

    # Start the rendering thread
    render_thread = threading.Thread(target=render_in_background, daemon=True)
    render_thread.start()

    flip = False
    print1 = False  # I hate this strategy
    print2 = False  # Still do
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # I think this section can look better.
        keys = pygame.key.get_pressed()

        if keys[pygame.K_UP] or keys[pygame.K_KP8]:
            explorer.move(maze=maze, flbr=0)
        if keys[pygame.K_DOWN] or keys[pygame.K_KP5]:
            explorer.move(maze=maze, flbr=2)
        if keys[pygame.K_KP1]:
            explorer.move(maze=maze, flbr=1)
        if keys[pygame.K_KP3]:
            explorer.move(maze=maze, flbr=3)
        if keys[pygame.K_LEFT] or keys[pygame.K_KP4]:
            explorer.rotate(left=True, amount=config.rotation_speed)
        if keys[pygame.K_RIGHT] or keys[pygame.K_KP6]:
            explorer.rotate(left=False, amount=config.rotation_speed)

        # Switch visualizing mode with 'keypad 0'
        if keys[pygame.K_KP0]:
            flip = True
        elif flip and not keys[pygame.K_KP0]:
            if isinstance(renderer, Rendering3D):
                renderer = Rendering2D(maze, explorer)
            elif isinstance(renderer, Rendering2D):
                renderer = MiniMap.MiniMap(maze, explorer)
            elif isinstance(renderer, MiniMap.MiniMap):
                renderer = Rendering3D(maze, explorer, True)
            flip = False

        # Print the whole adjacency_list with 'p'
        if keys[pygame.K_p]:
            print1 = True
        elif print1 and not keys[pygame.K_p]:
            renderer.print_debug_info()
            print1 = False

        # Print the whole adjacency_list with 'o'
        if keys[pygame.K_o]:
            print2 = True
        elif print2 and not keys[pygame.K_o]:
            print("Full adjacency map:")
            for tile, adjacents in maze.adjacency_map.items():
                print(tile, ":", adjacents)
            print2 = False

        time.sleep(config.render_interval)

    pygame.quit()


if __name__ == '__main__':
    run_game(default_render="3D")

