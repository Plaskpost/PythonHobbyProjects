
class WallSegment:

    def __init__(self):
        self.left_point = (-1, -1)
        self.right_point = (-1, -1)

    def extend(self, point):
        self.left_point = point

    def cut(self, ):