import pygame
from DynamicMaze import DynamicMaze
from Rending2D import Rending2D

movement_speed = 1
tile_size = 80  # Make independent of SQUARE_SIZE later?
player_radius = 10
pos = [tile_size//2, tile_size//2]
pos_tile = ""

maze = DynamicMaze(pos_tile)
renderer = Rending2D(maze, pos_tile, pos, player_radius)
maze.update_visibility(pos_tile)
renderer.update(pos_tile, pos)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # I think this section can look better.
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        pos[1] -= movement_speed
        if pos[1] < player_radius and maze.wall_map[pos_tile][2] == -1:
            pos[1] += movement_speed
        elif pos[1] < 0:
            pos[1] += tile_size
            pos_tile = maze.adjacency_map[pos_tile][2]
            maze.update_visibility(pos_tile)
    if keys[pygame.K_DOWN]:
        pos[1] += movement_speed
        if pos[1] >= tile_size-player_radius and maze.wall_map[pos_tile][0] == -1:
            pos[1] -= movement_speed
        elif pos[1] >= tile_size:
            pos[1] -= tile_size
            pos_tile = maze.adjacency_map[pos_tile][0]
            maze.update_visibility(pos_tile)
    if keys[pygame.K_LEFT]:
        pos[0] -= movement_speed
        if pos[0] < player_radius and maze.wall_map[pos_tile][3] == -1:
            pos[0] += movement_speed
        elif pos[0] < 0:
            pos[0] += tile_size
            pos_tile = maze.adjacency_map[pos_tile][3]
            maze.update_visibility(pos_tile)
    if keys[pygame.K_RIGHT]:
        pos[0] += movement_speed
        if pos[0] >= tile_size-player_radius and maze.wall_map[pos_tile][1] == -1:
            pos[0] -= movement_speed
        elif pos[0] >= tile_size:
            pos[0] -= tile_size
            pos_tile = maze.adjacency_map[pos_tile][1]
            maze.update_visibility(pos_tile)


    renderer.update(pos_tile, pos)

# Quit Pygame
pygame.quit()
