import pygame
import numpy as np

import HyperbolicGrid
import TrainingDataGenerator
from DynamicMaze import DynamicMaze
from Explorer import Player
import MiniMap

SCREEN_SIZE = 600


def scaled_center(center):
    screen_half = SCREEN_SIZE // 2
    new_center = screen_half + screen_half * np.array([center[0], -center[1]])
    return new_center

def scaled_radius(radius):
    return (SCREEN_SIZE // 2) * radius

def draw_n_circles(screen, point, n, normal, bar_value):
    RED = (255, 0, 0)
    WHITE = (255, 255, 255)

    new_point = point.__copy__()
    pygame.draw.circle(screen, RED, scaled_center(new_point), 5)
    circle = MiniMap.find_circle(point, normal)
    facing_angle = np.pi / 2
    for i in range(n):
        new_point, angle_change = TrainingDataGenerator.brute_guessing_p2(new_point, circle, facing_angle, bar_value)

        pygame.draw.circle(screen, WHITE, scaled_center(circle[0]), scaled_radius(circle[1]), 1)
        pygame.draw.circle(screen, RED, scaled_center(new_point), 5)

        facing_angle += angle_change
        perpendicular_normal = MiniMap.to_cartesian(np.array([1., facing_angle]))
        facing_angle += np.pi / 2
        circle = MiniMap.find_circle(new_point, perpendicular_normal)



def draw_slider(screen, slider_x, slider_y, slider_width, slider_height, bar_value, max_value, handle_radius):
    GRAY = (100, 100, 100)
    BLUE = (0, 0, 255)
    WHITE = (255, 255, 255)
    pygame.draw.rect(screen, GRAY, (slider_x, slider_y, slider_width, slider_height))

    # Calculate handle position
    handle_x = slider_x + (bar_value / max_value) * slider_width
    pygame.draw.circle(screen, BLUE, (int(handle_x), slider_y + slider_height // 2), handle_radius)

    # Display the value as text
    font = pygame.font.Font(None, 20)
    text = font.render(f"Value: {bar_value:.5f}", True, WHITE)
    screen.blit(text, (slider_x, slider_y - 40))


def run_slider_tweaking_game():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    # mini_map = MiniMap.MiniMap(screen, (300, 300), 600)

    maze = DynamicMaze()
    HyperbolicGrid.bulk_registration(maze.adjacency_map, "", 3)
    for key in maze.adjacency_map:
        if maze.adjacency_map[key] is not None:
            maze.wall_map[key] = [1, 1, 1, 1]  # Making a map with no walls

    explorer = Player()

    slider_x = 150  # Position of the slider on the screen
    slider_y = 560
    slider_width = 300
    slider_height = 10
    handle_radius = 10

    point = np.array([0.9, 0.0])
    normal = np.array([-1., 0.0])

    # Bar values
    bar_value = 1.06
    max_value = 2.0

    n = 5

    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                # Check if the click is on the slider
                if slider_x <= mouse_x <= slider_x + slider_width and slider_y - handle_radius <= mouse_y <= slider_y + handle_radius:
                    # Calculate the new value
                    bar_value = (mouse_x - slider_x) / slider_width * max_value
            elif event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:  # Check if the left mouse button is held down
                    mouse_x, _ = pygame.mouse.get_pos()
                    # Ensure the handle stays within the slider bounds
                    if slider_x <= mouse_x <= slider_x + slider_width:
                        bar_value = (mouse_x - slider_x) / slider_width * max_value

        screen.fill(BLACK)
        pygame.draw.circle(screen, WHITE, (SCREEN_SIZE // 2, SCREEN_SIZE // 2), SCREEN_SIZE // 2, 2)
        draw_n_circles(screen, point, n, normal, bar_value)
        draw_slider(screen, slider_x, slider_y, slider_width, slider_height, bar_value, max_value, handle_radius)

        # Update the display
        pygame.display.flip()


def run_moving_player_game():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    # mini_map = MiniMap.MiniMap(screen, (300, 300), 600)

    maze = DynamicMaze()
    HyperbolicGrid.bulk_registration(maze.adjacency_map, "", 3)
    for key in maze.adjacency_map:
        if maze.adjacency_map[key] is not None:
            maze.wall_map[key] = [-1, -1, -1, -1]  # Making a map with only walls

    explorer = Player()




if __name__ == '__main__':
    run_moving_player_game()




