import pygame
import numpy as np

from DynamicMaze import DynamicMaze
from Rendering2D import Rendering2D
from Rendering3D import Rendering3D
from Explorer import Explorer


# Constants
DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

movement_speed = 0.4
rotation_speed = 0.3  # Ok degrees
tile_size = 80
player_radius = 10
average_walls_per_tile = 2.8
seed = 44
fixed_seed = True


if __name__ == '__main__':

    if fixed_seed:
        np.random.seed(seed)
    explorer = Explorer(movement_speed, rotation_speed, tile_size, player_radius)
    maze = DynamicMaze(explorer.pos_tile, average_walls_per_tile)
    renderer = Rendering3D(maze, explorer)
    if isinstance(renderer, Rendering2D):
        maze.update_visibility(explorer.pos_tile)
    renderer.update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # I think this section can look better.
        keys = pygame.key.get_pressed()
        pressed = False
        arrow_keys = [pygame.K_DOWN, pygame.K_RIGHT, pygame.K_UP,
                      pygame.K_LEFT]  # This way i should match the direction.

        if keys[pygame.K_UP]:
            explorer.move(forward=True, maze=maze)
            pressed = True
        if keys[pygame.K_DOWN]:
            explorer.move(forward=False, maze=maze)
            pressed = True
        if keys[pygame.K_LEFT]:
            explorer.rotate(left=True, amount=rotation_speed)
            pressed = True
        if keys[pygame.K_RIGHT]:
            explorer.rotate(left=False, amount=rotation_speed)
            pressed = True

        if pressed:
            renderer.update()

    pygame.quit()
