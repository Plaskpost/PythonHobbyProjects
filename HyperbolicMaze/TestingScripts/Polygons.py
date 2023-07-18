import pygame

# Initialize Pygame
pygame.init()

# Set the window size
window_size = (800, 600)
screen = pygame.display.set_mode(window_size)

# Set the colors
black = (0, 0, 0)
white = (255, 255, 255)

# Define the polygon points
polygon_points = [(100, 100), (200, 50), (300, 150), (250, 300)]

# Clear the screen
screen.fill((100, 100, 100))

# Draw the filled polygon
pygame.draw.polygon(screen, white, polygon_points)

# Draw the polygon outline
pygame.draw.polygon(screen, black, polygon_points, 2)

# Update the display
pygame.display.flip()

# Event loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

# Quit Pygame
pygame.quit()
