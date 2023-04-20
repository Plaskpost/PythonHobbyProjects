import pygame


class Rending2D:

    def __init__(self, dynamicMaze, pos):
        self.WIDTH = 1200
        self.HEIGHT = 800
        self.SQUARE_SIZE = 80
        self.WALL_THICKNESS = 4  # *2
        self.SQUARE_COLOR = (255, 255, 255)
        self.WALL_COLOR = (0, 0, 0)
        self.DOT_COLOR = (255, 0, 0)
        self.DOT_SIZE = 10
        pygame.init()
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT))
        pygame.display.set_caption("Overview")
        self.screen.fill((50, 50, 50))
        self.maze = dynamicMaze
        self.update(pos)

    def update(self, tile):
        self.update_recursive(tile, None, (0, 0))
        pygame.draw.circle(self.screen, self.DOT_COLOR, ((self.WIDTH-(self.DOT_SIZE//2))//2, (self.HEIGHT-(self.DOT_SIZE//2))//2), 10)
        pygame.display.flip()

    def update_recursive(self, tile, prev_tile, screen_position):
        if not self.maze.visibility_map[tile]:
            return
        square_x = screen_position[0] * self.SQUARE_SIZE + (self.WIDTH - self.SQUARE_SIZE) // 2
        square_y = screen_position[1] * self.SQUARE_SIZE + (self.HEIGHT - self.SQUARE_SIZE) // 2
        pygame.draw.rect(self.screen, self.SQUARE_COLOR, ((square_x, square_y), (self.SQUARE_SIZE, self.SQUARE_SIZE)))

        shifts = [[0, 1], [1, 0], [0, -1], [-1, 0]]
        for i in range(4):
            if self.maze.wall_map[tile][i] == -1:
                wall_x, wall_y, wall_w, wall_h = self.where_wall(i, (square_x, square_y))
                pygame.draw.rect(self.screen, self.WALL_COLOR, ((wall_x, wall_y), (wall_w, wall_h)))
            else:
                neighbor = self.maze.adjacency_map[tile][i]
                if neighbor == prev_tile:
                    continue
                x = screen_position[0] + shifts[i][0]
                y = screen_position[1] + shifts[i][1]
                self.update_recursive(neighbor, tile, (x, y))

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
