import numpy as np

import pygame
import sys
from GameOfLife import GameOfLife

# Create a sample NxN matrix and a 3x3 kernel
N = 50

# Initialize Pygame
pygame.init()
gol = GameOfLife(N)

# Constants
SCREEN_SIZE = (400, 400)
SQUARE_SIZE = SCREEN_SIZE[0] // N
LOOPS_PER_TICK = 20

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Variables
loop_counter = 0

# Create the screen
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Click to Mark Squares")


marked_squares = []
mouse_button_held = False

# Main loop
running = True
while running:

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            mouse_button_held = True
        elif event.type == pygame.MOUSEBUTTONUP:
            for square in marked_squares:
                gol.board[square[0], square[1]] = 1
            marked_squares = []
            mouse_button_held = False
        elif event.type == pygame.MOUSEMOTION and mouse_button_held:
            x, y = pygame.mouse.get_pos()
            row = y // SQUARE_SIZE
            col = x // SQUARE_SIZE
            if [row, col] not in marked_squares:
                marked_squares.append([row, col])

    # Draw the grid
    for row in range(N):
        for col in range(N):
            color = WHITE if gol.board[row][col] else BLACK
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in marked_squares:
        pygame.draw.rect(screen, WHITE, (square[1] * SQUARE_SIZE, square[0] * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    pygame.display.flip()
    loop_counter += 1
    if loop_counter == LOOPS_PER_TICK:
        gol.tick()
        loop_counter = 0

# Clean up
pygame.quit()
sys.exit()

