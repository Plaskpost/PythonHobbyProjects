import threading
import time

import pygame
import numpy as np

import ScreenCapture
import config

import MiniMap
import DynamicMaze
from Rendering2D import Rendering2D
from Rendering3D import Rendering3D
from Explorer import Player


def run_game(default_render="3D", maze=None, include_mini_map=True, mini_map_generates_tiles=False, screen_capture=False):
    if config.fixed_seed:
        np.random.seed(config.seed)
    explorer = Player()
    if maze is None:
        maze = DynamicMaze.DynamicMaze()
    if default_render == "3D":
        miniature_map = 'hyperbolic' if include_mini_map else None
        renderer = Rendering3D(maze, explorer, miniature_map=miniature_map,
                               mini_map_generates_tiles=mini_map_generates_tiles)
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

                if screen_capture:
                    ScreenCapture.save_frame(renderer.screen)

                last_render_time = current_time

    # Start the rendering thread
    render_thread = threading.Thread(target=render_in_background, daemon=True)
    render_thread.start()

    flip = False
    mini_map_on_off = False
    mini_map_on = include_mini_map
    print1 = False  # I hate this strategy
    print2 = False  # Still do
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # I think this section can look better.
        keys = pygame.key.get_pressed()
        if keys[pygame.K_q]:
            running = False

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
            if isinstance(renderer, Rendering2D):
                renderer = Rendering3D(maze, explorer, mini_map_on)
            elif isinstance(renderer, Rendering3D):
                renderer = MiniMap.MiniMap(maze, explorer, tile_generating=mini_map_generates_tiles)
            elif isinstance(renderer, MiniMap.MiniMap):
                renderer = Rendering2D(maze, explorer)
            flip = False

        # Delete or bring back mini map with 'keypad 2'
        if keys[pygame.K_KP2]:
            mini_map_on_off = True
        elif mini_map_on_off and not keys[pygame.K_KP2]:
            mini_map_on = not mini_map_on
            if isinstance(renderer, Rendering3D):
                if mini_map_on:
                    renderer.mini_map = MiniMap.MiniMap(maze, explorer, 'bottom-right')
                else:
                    renderer.mini_map = None
            mini_map_on_off = False

        # A segment of outdated debugging code. Will update it if I need it.
        #if keys[pygame.K_p]:
        #    print1 = True
        #elif print1 and not keys[pygame.K_p]:
        #    renderer.print_debug_info()
        #    print1 = False

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

