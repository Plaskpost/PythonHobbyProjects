import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters for the circle
h = 1
k = 0
r = 1

# Define the parameter t
t = np.linspace(0, 2*np.pi, 1000)

# Parametric equations for x(t), y(t), and z(t)
x_t = h + r * np.cos(t)
y_t = k + r * np.sin(t)

# Function to compute z(t)
def z_t(x_t, y_t):
    rho_squared = x_t**2 + y_t**2
    return np.arccosh(1 + (2 * rho_squared) / (1 - rho_squared))

# Compute z(t)
z_t_values = z_t(x_t, y_t)

# Define the unit circle in the xy-plane
x_circle = np.cos(t)
y_circle = np.sin(t)
z_circle = np.zeros_like(t)  # z = 0 for the xy-plane

# Plotting the curve in 3D space
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the parametric curve
ax.plot(x_t, y_t, z_t_values, label='Parametric Curve', color='b')

# Plot the unit circle in the xy-plane
ax.plot(x_circle, y_circle, z_circle, label='Unit Circle in xy-plane', color='r', linestyle='--')

# Set labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Set the title
ax.set_title('3D Parametric Curve with Unit Circle')

# Set the aspect ratio for better visualization
ax.set_box_aspect([1,1,1])

# Add a legend
ax.legend()

# Show the plot
plt.show()
