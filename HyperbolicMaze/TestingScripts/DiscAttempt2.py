import numpy as np
import matplotlib.pyplot as plt

import MiniMap


def plot_circle(center, radius):
    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)
    ax.plot(x_circle, y_circle, 'g--', alpha=0.5)


# Parameters
p = np.array([0.1, 0.2])
N = np.array([0.6, 0.8])
d = 0.2
iterations = 10

fig, ax = plt.subplots()
circle = plt.Circle((0, 0), 1, color='black', fill=False)
ax.add_artist(circle)

ax.plot(p[0], p[1], 'bo')

center, radius = MiniMap.find_circle(p, N)
plot_circle(center, radius)

for _ in range(iterations):
    new_position, new_normal = translate_along_circle(p, N, center, radius, d)

    theta = np.linspace(0, 2 * np.pi, 100)
    x_circle = center[0] + radius * np.cos(theta)
    y_circle = center[1] + radius * np.sin(theta)
    ax.plot(x_circle, y_circle, 'g--', alpha=0.5)

    ax.plot(new_position[0], new_position[1], 'ro')

    p, N = new_position, new_normal

ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.2, 1.2)
ax.set_aspect('equal')
ax.set_title('Hyperbolic Translations in the Poincaré Disk')
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.show()

