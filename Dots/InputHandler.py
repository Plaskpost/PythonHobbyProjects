import pygame
import sys
from Drawer import Drawer
import config


class InputHandler:
    WHITE = (255, 255, 255)
    GRAY = (120, 120, 120)
    LIGHT_GRAY = (200, 200, 200)
    BLACK = (0, 0, 0)
    RED = (255, 0, 0)
    BLUE = (0, 0, 255)

    def __init__(self):
        self.drawer = Drawer()
        pygame.init()

        screen_width, screen_height = 800, 600
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Mouse Position Example")

    def run(self):
        while True:
            # Get the mouse position
            mouse_x, mouse_y = pygame.mouse.get_pos()
            self.mouse_pos = (mouse_x, mouse_y)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button released
                        self.drawer.left_mouse_click(self.mouse_pos)
                    elif event.button == 3:  # Right mouse button released
                        self.drawer.right_mouse_click(self.mouse_pos)
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_d:
                        self.drawer.set_mode("d")
                    elif event.key == pygame.K_l:
                        self.drawer.set_mode("l")
                    elif event.key == pygame.K_e:
                        self.drawer.set_mode("e")

            # Clear the screen and draw something (this is optional)
            self.screen.fill((255, 255, 255))
            self.draw_dot(self.mouse_pos, self.BLACK)
            self.draw_all_dots()
            self.draw_all_lines()

            # Update the screen
            pygame.display.flip()

    def draw_all_dots(self):
        for dot in self.drawer.dots:
            self.draw_dot(dot, self.GRAY)
        if self.drawer.mode == "l":
            closest_dot = self.drawer.get_closest_dot(self.mouse_pos)
            self.draw_dot(closest_dot, self.RED)
            if self.drawer.line_started:
                self.draw_dot(self.drawer.active_line.dot1, self.BLUE)

    def draw_all_lines(self):
        for line in self.drawer.lines:
            self.draw_line(line, self.BLACK)
        if self.drawer.mode == "e":
            intersecting_line = self.drawer.get_intersecting_line(self.mouse_pos)
            if intersecting_line is not None:
                self.draw_line(intersecting_line, self.LIGHT_GRAY)

    def draw_dot(self, pos, color):
        pygame.draw.circle(self.screen, color, pos, config.dot_radius, 1)

    def draw_line(self, line, color):
        pygame.draw.line(self.screen, color, line.dot1, line.dot2, 2)


if __name__ == "__main__":
    g = InputHandler()
    g.run()
