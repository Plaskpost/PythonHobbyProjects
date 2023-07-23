
class Dot:
    def __init__(self, pos):
        self.pos = pos

    def __eq__(self, other):
        return self.pos[0] == other.pos[0] and self.pos[1] == other.pos[1]