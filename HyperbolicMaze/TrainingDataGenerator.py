import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import MiniMap
import math


# px py Nx Ny z
def to_cartesian(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.array([x, y])


def to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([r, phi])


def compute_rho_squared(z):  # Compute x^2 + y^2 from z
    return (np.cosh(z) - 1) / (np.cosh(z) + 1)

def circle_equation(vars, circle, rho_squared):
    x, y = vars
    (h, k), r = circle
    return [
        (x - h)**2 + (y - k)**2 - r**2,
        x**2 + y**2 - rho_squared
    ]


def plot_instance(circle, z, s_1, s_2):
    t = np.linspace(0, 2 * np.pi, 1000)
    c, r = circle
    curve_min = find_z0(circle)

    # Parametric equations for x(t), y(t), and z(t)
    x_t = c[0] + r * np.cos(t)
    y_t = c[1] + r * np.sin(t)

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
    ax.plot(x_circle, y_circle, z_circle, label='Unit Circle in xy-plane', color='black', linestyle='--')

    # Other points
    #ax.scatter(curve_min[0], curve_min[1], curve_min[2], color='yellow', s=10)
    z_1 = z_t(s_1[0], s_1[1])
    z_2 = z_t(s_2[0], s_2[1])
    ax.scatter(s_1[0], s_1[1], z_1, color='green', s=10)
    ax.scatter(s_2[0], s_2[1], z_2, color='red', s=10)
    ax.scatter(curve_min[0], curve_min[1], curve_min[2], color='yellow', s=10)

    # Add a plane at z
    x_plane = np.linspace(-1., 1., 100)  # Define the x range for the plane
    y_plane = np.linspace(-1., 1., 100)  # Define the y range for the plane
    x_plane, y_plane = np.meshgrid(x_plane, y_plane)
    z_plane = np.full_like(x_plane, z)
    ax.plot_surface(x_plane, y_plane, z_plane, alpha=0.1, color='black')
    min_z_plane = z_plane = np.full_like(x_plane, curve_min[2])
    ax.plot_surface(x_plane, y_plane, min_z_plane, alpha=0.3, color='yellow')

    # Set labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set the title
    c_polar = to_polar(c[0], c[1])
    ax.set_title(f"c (polar) = ({c_polar[0]:.2f}, {c_polar[1]:.2f}), r = {r:.2f}, z = {z}, solution 1 = {s_1}, solution 2 = {s_2}")

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 10])

    # Show the plot
    plt.show()


def top_down_plot(circle, prediction_1, prediction_2, s_1, s_2):
    # Unpack the input
    center, radius = circle
    fig, ax = plt.subplots()

    # Plot the main circle
    circle_plot = plt.Circle(center, radius, color='blue', fill=False, linewidth=2)
    ax.add_artist(circle_plot)

    # Plot the points
    ax.plot(s_1[0], s_1[1], 'bo')
    ax.plot(s_2[0], s_2[1], 'bo')
    ax.plot(prediction_1[0], prediction_1[1], 'go', label='s_1')  # green point for s_1
    ax.plot(prediction_2[0], prediction_2[1], 'ro', label='s_2')  # red point for s_2


    # Plot the unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_unit_circle = np.cos(theta)
    y_unit_circle = np.sin(theta)
    ax.plot(x_unit_circle, y_unit_circle, 'k--', label='Unit Circle')  # dotted black line for unit circle

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')
    ax.grid(True)
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    plt.show()


def find_z0(circle):
    c, r = circle
    center_r, center_phi = to_polar(c[0], c[1])
    (x0, y0) = to_cartesian(center_r-r, center_phi)
    z0 = z_t(x0, y0)
    return x0, y0, z0


def z_t(x_t, y_t):
    rho_squared = x_t ** 2 + y_t ** 2
    return np.arccosh(1 + (2 * rho_squared) / (1 - rho_squared))


def compute_outputs(point, circle, z, printt=False, plot=False):

    initial_guess_1 = MiniMap.translate_along_circle(point, circle, 1.)
    initial_guess_2 = MiniMap.translate_along_circle(point, circle, -1.)
    rho_squared = compute_rho_squared(z)
    # Solve for (x, y)
    solution_1 = fsolve(circle_equation, initial_guess_1, args=(circle, rho_squared))
    solution_2 = fsolve(circle_equation, initial_guess_2, args=(circle, rho_squared))

    if printt:
        print(f"c = ({circle[0][0]:.2f}, {circle[0][1]:.2f}), r = {circle[1]:.2f}, z = {z}, solution 1 = {solution_1}, solution 2 = {solution_2}")

    if plot:
        curve_min = find_z0(circle)
        plot_instance(circle, z, solution_1, solution_2)

    return solution_1, solution_2


def generate_data(num_samples, printt=False, plot=False):
    p_r = np.random.rand(num_samples)
    p_phi = np.random.uniform(0., 2*np.pi, num_samples)
    N_phi = np.random.uniform(0., 2*np.pi, num_samples)
    z, x1, y1, x2, y2 = np.zeros((5, num_samples))

    for i in range(num_samples):
        p_cartesian = to_cartesian(p_r[i], p_phi[i])
        N_cartesian = to_cartesian(1., N_phi[i])
        circle = MiniMap.find_circle(p_cartesian, N_cartesian)
        _, _, z0 = find_z0(circle)

        z[i] = z0 + (6. - z0) * np.random.rand()

        s1, s2 = compute_outputs(p_cartesian, circle, z[i], printt=printt, plot=plot)
        x1[i], y1[i] = s1
        x2[i], y2[i] = s2


    X = np.stack([p_r, p_phi, N_phi, z], axis=1)
    y = np.stack([x1, y1, x2, y2], axis=1)

    return X, y




if __name__ == '__main__':
    generate_data(num_samples=3, printt=True, plot=False)





