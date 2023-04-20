import pygame
from DynamicMaze import DynamicMaze
from Rending2D import Rending2D

pos = ""
maze = DynamicMaze(pos)
renderer = Rending2D(maze, pos)

maze.update_visibility(pos, "", 2)
renderer.update(pos)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()
