import math
import sys
import pygame

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
ORIGIN = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
DOT_RADIUS = 10
DOT_COLOR = (255, 0, 0)
LINE_COLOR = (0, 0, 255)

# Set up the screen
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Coordinate System")
font = pygame.font.SysFont('arial', 12)

# Dot positions
dot1_pos = [ORIGIN[0] + 100, ORIGIN[1] - 100]
dot2_pos = [ORIGIN[0] - 100, ORIGIN[1] - 100]
dot3_pos = [0, 0]
angle_dot3 = 60


def draw_coordinate_system():
    global dot1_pos, dot2_pos, dot3_pos, angle_dot3
    # Draw the coordinate axes
    pygame.draw.line(screen, LINE_COLOR, (ORIGIN[0], 0), (ORIGIN[0], SCREEN_HEIGHT))
    pygame.draw.line(screen, LINE_COLOR, (0, ORIGIN[1]), (SCREEN_WIDTH, ORIGIN[1]))

    # Draw the dots
    pygame.draw.line(screen, (0, 0, 0), dot1_pos, dot2_pos)
    pygame.draw.circle(screen, DOT_COLOR, dot1_pos, DOT_RADIUS)
    pygame.draw.circle(screen, DOT_COLOR, dot2_pos, DOT_RADIUS)
    pygame.draw.line(screen, DOT_COLOR, ORIGIN, dot1_pos)
    pygame.draw.line(screen, DOT_COLOR, ORIGIN, dot2_pos)

    dot1_cartesian = to_coordinate_pos(dot1_pos)
    dot1_polar = cartesian_to_polar(dot1_cartesian)
    dot2_cartesian = to_coordinate_pos(dot2_pos)
    dot2_polar = cartesian_to_polar(dot2_cartesian)

    dist_dot3 = find_on_line(dot1_polar, dot2_polar, angle_dot3)
    dot3_cartesian = polar_to_cartesian([dist_dot3, angle_dot3])
    dot3_pos = to_screen_pos(dot3_cartesian)
    pygame.draw.circle(screen, (0, 0, 0), dot3_pos, DOT_RADIUS)
    pygame.draw.line(screen, (0, 0, 0), ORIGIN, dot3_pos)
    screen.blit(font.render("a", True, (0, 0, 0)), dot1_pos)
    screen.blit(font.render("b", True, (0, 0, 0)), dot2_pos)





def find_on_line(point1, point2, angle3):
    cond1 = is_in_quadrant(point1[1], 2) and is_in_quadrant(point2[1], 3) and is_in_quadrant(angle_dot3, 2)
    cond2 = is_in_quadrant(point1[1], 3) and is_in_quadrant(point2[1], 2) and is_in_quadrant(angle_dot3, 3)
    if cond1 or cond2:
        dummy = point1
        point1 = point2
        point2 = dummy
    a = point1[0]
    b = point2[0]
    gamma = math.radians(abs(point1[1] - point2[1]))
    gamma_bc = math.radians(abs(angle3 - point2[1]))

    c = math.sqrt(a ** 2 + b ** 2 - 2 * a * b * math.cos(gamma))
    alpha = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
    return math.sin(alpha) * b / math.sin(math.pi - alpha - gamma_bc)


def is_in_quadrant(angle, quadrant):
    lower_limit = 90*(quadrant-1)
    upper_limit = 90*quadrant
    return lower_limit < (angle % 360) < upper_limit


def cartesian_to_polar(pos):
    [x, y] = pos
    r = math.sqrt(x**2 + y**2)
    theta = math.degrees(math.atan2(y, x))
    return [r, theta]

def polar_to_cartesian(pos):
    [r, theta] = pos
    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    return [x, y]

def to_coordinate_pos(pos):
    x = pos[0] - ORIGIN[0]
    y = ORIGIN[1] - pos[1]
    return [x, y]

def to_screen_pos(pos):
    x = pos[0] + ORIGIN[0]
    y = ORIGIN[1] - pos[1]
    return [x, y]

def main():
    global dot1_pos, dot2_pos, dot3_pos, angle_dot3
    clock = pygame.time.Clock()
    running = True
    active_dot = None

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.MOUSEBUTTONDOWN:
                mouse_pos = pygame.mouse.get_pos()
                # Check if the mouse click is within one of the dots
                if pygame.Rect(dot1_pos[0] - DOT_RADIUS, dot1_pos[1] - DOT_RADIUS, 2*DOT_RADIUS, 2*DOT_RADIUS).collidepoint(mouse_pos):
                    active_dot = 1
                elif pygame.Rect(dot2_pos[0] - DOT_RADIUS, dot2_pos[1] - DOT_RADIUS, 2*DOT_RADIUS, 2*DOT_RADIUS).collidepoint(mouse_pos):
                    active_dot = 2
                elif pygame.Rect(dot3_pos[0] - DOT_RADIUS, dot3_pos[1] - DOT_RADIUS, 2*DOT_RADIUS, 2*DOT_RADIUS).collidepoint(mouse_pos):
                    active_dot = 3

            elif event.type == pygame.MOUSEBUTTONUP:
                active_dot = None

        if active_dot:
            # If a dot is being dragged, update its position
            mouse_pos = pygame.mouse.get_pos()
            if active_dot == 1:
                dot1_pos[0] = mouse_pos[0]
                dot1_pos[1] = mouse_pos[1]
            elif active_dot == 2:
                dot2_pos[0] = mouse_pos[0]
                dot2_pos[1] = mouse_pos[1]
            elif active_dot == 3:
                mouse_pos_polar = cartesian_to_polar(to_coordinate_pos(mouse_pos))
                angle_dot3 = mouse_pos_polar[1]

        # Clear the screen
        screen.fill((255, 255, 255))

        # Draw the coordinate system and dots
        draw_coordinate_system()

        # Update the display
        pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()
