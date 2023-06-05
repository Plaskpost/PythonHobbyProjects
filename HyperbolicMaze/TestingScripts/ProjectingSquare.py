import pygame
import math
import numpy as np

def stereographic_projection(x, y):
    radius = 200  # Radius of the Poincaré disk
    center_x, center_y = window_size[0] / 2, window_size[1] / 2  # Center of the Pygame window

    # Convert the coordinates to polar form
    r = math.sqrt(x * x + y * y)
    theta = math.atan2(y, x)

    # Apply the stereographic projection formula
    mapped_r = 2 * radius * math.tanh(r / (2 * radius))

    # Convert back to Cartesian coordinates
    mapped_x = center_x + mapped_r * math.cos(theta)
    mapped_y = center_y + mapped_r * math.sin(theta)

    return mapped_x, mapped_y

if __name__ == '__main__':
    movement_speed = 0.5
    square_scale = 10
    scaling_factor = 1
    pygame.init()
    window_size = (800, 800)
    screen = pygame.display.set_mode(window_size)
    pygame.display.set_caption("Stereographic Hyperbolic Projection")

    line = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])  # Example line in the hyperbolic plane
    shift = np.zeros((5,2))
    # Convert and draw the line in the Pygame window
    mapped_line = np.array([stereographic_projection(x, y) for x, y in line])

    #pygame.draw.lines(screen, (255, 255, 255), True, mapped_line)

    background_image = pygame.image.load('Images/Yellow_grid_image.png')
    background_image = pygame.transform.scale(background_image, window_size)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_UP]:
            for point in shift:
                point[1] -= movement_speed
        if keys[pygame.K_DOWN]:
            for point in shift:
                point[1] += movement_speed
        if keys[pygame.K_LEFT]:
            for point in shift:
                point[0] -= movement_speed
        if keys[pygame.K_RIGHT]:
            for point in shift:
                point[0] += movement_speed
        if keys[pygame.K_KP0]:
            square_scale -= scaling_factor
        if keys[pygame.K_KP1]:
            square_scale += scaling_factor

        scaled_line = square_scale*line
        shifted_line = scaled_line + shift
        mapped_line = [stereographic_projection(x, y) for x, y in shifted_line]
        #screen.fill((0,0,0))
        screen.blit(background_image, (0, 0))
        mapped_line = np.array([[int(num) for num in point] for point in mapped_line])
        #mapped_line += [screen.get_width() // 2, screen.get_height() // 2]
        pygame.draw.lines(screen, (0, 0, 0), True, mapped_line)

        pygame.display.flip()  # Update the Pygame window

    pygame.quit()