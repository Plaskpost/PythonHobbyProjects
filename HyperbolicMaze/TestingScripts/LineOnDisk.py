import pygame
import math

# Set up Pygame
pygame.init()
screen = pygame.display.set_mode((800, 800))
pygame.display.set_caption("Poincaré Disk")

# Define Poincaré disk parameters
disk_radius = 400
disk_center = (400, 400)

# Define line endpoints (Euclidean coordinates)
start_point = (100, 200)
end_point = (600, 500)

# Find midpoint of the given line
midpoint_x = (start_point[0] + end_point[0]) / 2
midpoint_y = (start_point[1] + end_point[1]) / 2

# Compute slope of the given line
dx = end_point[0] - start_point[0]
dy = end_point[1] - start_point[1]
slope = dy / dx

# Compute slope of the orthogonal line
orthogonal_slope = -1 / slope

# Compute length of the orthogonal line segment
length = 200

# Compute displacement vector along the orthogonal line
dx_orthogonal = length / (2 * math.sqrt(1 + orthogonal_slope**2))
dy_orthogonal = orthogonal_slope * dx_orthogonal

# Compute endpoints of the orthogonal line segment
orthogonal_start_point = (midpoint_x - dx_orthogonal, midpoint_y - dy_orthogonal)
orthogonal_end_point = (midpoint_x + dx_orthogonal, midpoint_y + dy_orthogonal)

# Perform Poincaré disk projection for line endpoints
start_point_poincare = (
    (2 * start_point[0] - 800) / (800 - 2 * start_point[0] * disk_radius**2),
    (2 * start_point[1] - 800) / (800 - 2 * start_point[1] * disk_radius**2)
)
end_point_poincare = (
    (2 * end_point[0] - 800) / (800 - 2 * end_point[0] * disk_radius**2),
    (2 * end_point[1] - 800) / (800 - 2 * end_point[1] * disk_radius**2)
)

# Perform Poincaré disk projection for orthogonal line endpoints
orthogonal_start_point_poincare = (
    (2 * orthogonal_start_point[0] - 800) / (800 - 2 * orthogonal_start_point[0] * disk_radius**2),
    (2 * orthogonal_start_point[1] - 800) / (800 - 2 * orthogonal_start_point[1] * disk_radius**2)
)
orthogonal_end_point_poincare = (
    (2 * orthogonal_end_point[0] - 800) / (800 - 2 * orthogonal_end_point[0] * disk_radius**2),
    (2 * orthogonal_end_point[1] - 800) / (800 - 2 * orthogonal_end_point[1] * disk_radius**2)
)

# Draw the lines on the Pygame screen
pygame.draw.line(screen, (255, 255, 255), start_point_poincare, end_point_poincare)
pygame.draw.line(screen, (255, 255, 255), orthogonal_start_point_poincare, orthogonal_end_point_poincare)

# Update the Pygame screen
pygame.display.flip()

# Game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
