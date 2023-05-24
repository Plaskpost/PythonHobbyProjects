import pygame

from DynamicMaze import DynamicMaze
from Rending2D import Rending2D
from Explorer import Explorer


# Constants
DOWN = 0
RIGHT = 1
UP = 2
LEFT = 3

movement_speed = 0.2
rotation_speed = 0.2  # Ok degrees
tile_size = 80
player_radius = 10


if __name__ == '__main__':

    explorer = Explorer(movement_speed, rotation_speed, tile_size, player_radius)
    maze = DynamicMaze(explorer.pos_tile)
    renderer = Rending2D(maze, explorer, player_radius, tile_size)
    maze.update_visibility(explorer.pos_tile)
    renderer.update()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        # I think this section can look better.
        keys = pygame.key.get_pressed()
        arrow_keys = [pygame.K_DOWN, pygame.K_RIGHT, pygame.K_UP,
                      pygame.K_LEFT]  # This way i should match the direction.

        if keys[pygame.K_UP]:
            explorer.move(forward=True, maze=maze)
        if keys[pygame.K_DOWN]:
            explorer.move(forward=False, maze=maze)
        if keys[pygame.K_LEFT]:
            explorer.rotate(left=True, amount=rotation_speed)
        if keys[pygame.K_RIGHT]:
            explorer.rotate(left=False, amount=rotation_speed)

        if keys:
            renderer.update()

    pygame.quit()
