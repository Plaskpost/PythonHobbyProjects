import pygame


class Rending2D:

    def __init__(self, dynamicMaze, pos_tile, pos_coordinates, player_radius):
        self.WIDTH = 1200
        self.HEIGHT = 800
        self.SQUARE_SIZE = 80
        self.WALL_THICKNESS = 5  # *2
        self.SQUARE_COLOR = (255, 255, 255)
        self.BG_COLOR = (60, 60, 60)
        self.WALL_COLOR = (0, 0, 0)
        self.TEXT_COLOR = (0, 0, 0)
        self.DOT_COLOR = (255, 0, 0)
        self.DOT_SIZE = player_radius
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Overview")
        self.screen.fill(self.BG_COLOR)
        self.font = pygame.font.SysFont('arial', 12)
        self.maze = dynamicMaze
        self.update(pos_tile, pos_coordinates)

    def update(self, tile, player_position):
        self.screen.fill(self.BG_COLOR)
        self.update_recursive(tile, None, (0, 0), player_position)
        pygame.draw.circle(self.screen, self.DOT_COLOR, ((self.WIDTH-self.SQUARE_SIZE)//2, (self.HEIGHT-self.SQUARE_SIZE)//2), 10)
        pygame.display.flip()
        pygame.display.update()

    def update_recursive(self, tile, prev_tile, screen_position, player_position):
        if not self.maze.visibility_map[tile]:
            return

        # Draw the tile
        square_x = -player_position[0] + screen_position[0] * self.SQUARE_SIZE + (self.WIDTH - self.SQUARE_SIZE) // 2
        square_y = -player_position[1] + screen_position[1] * self.SQUARE_SIZE + (self.HEIGHT - self.SQUARE_SIZE) // 2
        self.draw_square((square_x, square_y), 45)

        # Label the tile
        text = self.font.render(tile, True, self.TEXT_COLOR)
        self.screen.blit(text, (square_x+12, square_y+12))

        # Draw walls
        shifts = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        for i in range(4):
            if self.maze.wall_map[tile][i] == -1:  # If wall draw wall.
                wall_x, wall_y, wall_w, wall_h = self.where_wall(i, (square_x, square_y))
                pygame.draw.rect(self.screen, self.WALL_COLOR, ((wall_x, wall_y), (wall_w, wall_h)))
            else:  # Else call drawing function for surrounding tiles.
                neighbor = self.maze.adjacency_map[tile][i]
                if neighbor == prev_tile:
                    continue
                x = screen_position[0] + shifts[i][0]
                y = screen_position[1] + shifts[i][1]
                self.update_recursive(neighbor, tile, (x, y), player_position)

    def where_wall(self, i, pos):
        x, y, w, h = None, None, None, None
        if i == 0:
            x = pos[0] - self.WALL_THICKNESS
            y = pos[1] + self.SQUARE_SIZE - self.WALL_THICKNESS
            w = self.SQUARE_SIZE + 2*self.WALL_THICKNESS
            h = self.WALL_THICKNESS
        elif i == 1:
            x = pos[0] + self.SQUARE_SIZE - self.WALL_THICKNESS
            y = pos[1] - self.WALL_THICKNESS
            w = self.WALL_THICKNESS
            h = self.SQUARE_SIZE + 2*self.WALL_THICKNESS
        elif i == 2:
            x = pos[0] - self.WALL_THICKNESS
            y = pos[1]
            w = self.SQUARE_SIZE + 2*self.WALL_THICKNESS
            h = self.WALL_THICKNESS
        elif i == 3:
            x = pos[0]
            y = pos[1] - self.WALL_THICKNESS
            w = self.WALL_THICKNESS
            h = self.SQUARE_SIZE + 2*self.WALL_THICKNESS
        return x, y, w, h

    def draw_square(self, center, rotation):
        size = (self.SQUARE_SIZE, self.SQUARE_SIZE)
        rect = pygame.Surface(size)
        rect.fill(self.SQUARE_COLOR)
        square = pygame.Surface((size[0], size[1]), pygame.SRCALPHA)
        pygame.draw.rect(square, self.SQUARE_COLOR, (0, 0, size[0], size[1]))
        rot_image = pygame.transform.rotate(square, rotation)
        rot_rect = rot_image.get_rect(center=center)
        self.screen.blit(rot_image, rot_rect)
