import math
from Line import Line
from config import dot_radius

class Drawer:

    def __init__(self):
        self.dots = set()
        self.lines = set()
        self.actions = self.get_actions()

        self.mode = "d"
        self.active_line = None
        self.line_started = False
        self.closest_dot = None

    def place_dot(self, mouse_pos):
        self.dots.add(mouse_pos)

    def make_line(self, mouse_pos):
        if not self.line_started:
            self.active_line = Line(self.closest_dot, None)
        else:
            self.active_line.dot2 = self.closest_dot
            self.lines.add(self.active_line)
        self.line_started = not self.line_started

    def erase_line(self, mouse_pos):
        line = self.get_intersecting_line(mouse_pos)
        self.lines.remove(line)

    def set_mode(self, key):
        self.mode = key

    def get_closest_dot(self, mouse_pos):
        shortest_distance = 100000
        x1, y1 = mouse_pos
        for dot in self.dots:
            x2, y2 = dot
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            if distance < shortest_distance:
                shortest_distance = distance
                self.closest_dot = dot
        return self.closest_dot

    def get_intersecting_line(self, mouse_pos):
        cx, cy = mouse_pos
        for line in self.lines:
            x1, y1 = line.dot1
            x2, y2 = line.dot2
            distance = abs((y2 - y1) * cx - (x2 - x1) * cy + x2 * y1 - y2 * x1) / math.sqrt(
                (y2 - y1) ** 2 + (x2 - x1) ** 2)
            if distance <= dot_radius:
                return line

    def left_mouse_click(self, mouse_pos):
        action = self.actions[self.mode]
        action(mouse_pos)

    def right_mouse_click(self, mouse_pos):
        pass

    def get_actions(self):
        actions = {
            "d": self.place_dot,
            "l": self.make_line,
            "e": self.erase_line
        }
        return actions
