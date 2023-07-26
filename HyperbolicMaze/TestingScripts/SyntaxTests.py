import math
import numpy as np

r = 0
phi = 1

def find_on_line(point1, point2, phi_c):
    a = point1[r]
    b = point2[r]
    gamma = np.radians(abs((point1[phi] - point2[phi]) % 360))
    gamma_bc = np.radians(abs((phi_c - point2[phi]) % 360))
    c = math.sqrt(a**2 + b**2 - 2*a*b * math.cos(gamma))
    alpha = math.acos(b**2 + c**2 - a**2) / (2 * b * c)
    return math.sin(alpha) * b / math.sin(math.pi-alpha-gamma_bc)


# Define the points in polar coordinates (r, phi)
A = np.array([1, 179])  # r_a = 1, phi_a = 1 degree
B = np.array([1, 1])  # r_b = 1, phi_b = 179 degrees
phi_c = 80  # Angle to the third point C in degrees

distance_c = find_on_line(A, B, phi_c)
print("Distance to the third point C:", distance_c)
