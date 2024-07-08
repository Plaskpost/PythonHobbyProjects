import pygame
import sys
import cmath

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 800
WHITE = (255, 255, 255)
POINT_RADIUS = 10
TRANSLATION_SPEED = 0.00001  # Adjust the translation speed as needed

# Initialize the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hyperbolic Point Translation")

# Define the initial point in the Poincaré disk
point = complex(0.0, 0.0)  # Adjust the initial point as needed

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Check for arrow key presses
    keys = pygame.key.get_pressed()

    # Apply a Möbius transformation to move the point along a hyperbolic geodesic
    if keys[pygame.K_LEFT]:
        b = complex(-1.0, 0.0)
    elif keys[pygame.K_RIGHT]:
        b = complex(1.0, 0.0)
    elif keys[pygame.K_DOWN]:
        b = complex(0.0, -1.0)
    elif keys[pygame.K_UP]:
        b = complex(0.0, 1.0)
    else:
        b = complex(0.0, 0.0)

    # Apply the Möbius transformation for hyperbolic translation
    point = (point - b) / (1 - b.conjugate() * point)

    # Ensure the point remains within the Poincaré disk
    if abs(point) >= 1:
        point /= abs(point)**2

    # Clear the screen
    screen.fill(WHITE)

    # Draw the Poincaré disk boundary
    pygame.draw.circle(screen, (0, 0, 0), (WIDTH // 2, HEIGHT // 2), WIDTH // 2, 1)

    # Draw the translated point in the Poincaré disk
    translated_x = WIDTH // 2 + int(WIDTH // 2 * point.real)
    translated_y = HEIGHT // 2 - int(HEIGHT // 2 * point.imag)
    pygame.draw.circle(screen, (0, 0, 255), (translated_x, translated_y), POINT_RADIUS)

    # Update the screen
    pygame.display.flip()

# Quit Pygame
pygame.quit()
sys.exit()
