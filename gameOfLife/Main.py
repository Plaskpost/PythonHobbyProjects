import numpy as np
from scipy.signal import convolve2d
import pygame
import sys

# Create a sample NxN matrix and a 3x3 kernel
N = 20
board = np.zeros((N, N))
board[1][1] = 1
board[1][0] = 1
board[1][2] = 1
kernel = np.ones((3, 3))
kernel[1][1] = 0

# Compute the convolution using scipy.signal.convolve2d
live_neighbors = convolve2d(board, kernel, mode='same', boundary='wrap')
alive_condition = np.logical_or(live_neighbors == 3, np.logical_and(board == 1, live_neighbors == 2))
board = np.where(alive_condition, 1, 0)


print(board)


# Initialize Pygame
pygame.init()

# Constants
SCREEN_SIZE = (400, 400)
GRID_SIZE = 4
SQUARE_SIZE = SCREEN_SIZE[0] // GRID_SIZE

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Create the screen
screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption("Click to Mark Squares")

# Create a grid to keep track of marked squares
grid = [[False for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            row = y // SQUARE_SIZE
            col = x // SQUARE_SIZE
            grid[row][col] = True

    screen.fill(BLACK)

    # Draw the grid
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            color = WHITE if grid[row][col] else BLACK
            pygame.draw.rect(screen, color, (col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    pygame.display.flip()

# Clean up
pygame.quit()
sys.exit()
