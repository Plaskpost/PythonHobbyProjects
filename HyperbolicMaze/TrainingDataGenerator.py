import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt

import DynamicMaze
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


# Hyperbolic distance function (from Equation A)
def hyperbolic_distance(p1, p2):
    norm_p1 = np.linalg.norm(p1)
    norm_p2 = np.linalg.norm(p2)

    # Ensure points are inside the unit circle
    if norm_p1 >= 1 or norm_p2 >= 1:
        return np.inf  # Return a large value if outside the unit circle

    euclidean_dist = np.linalg.norm(p1 - p2)
    try:
        return np.arccosh(1 + (2 * euclidean_dist ** 2) / ((1 - norm_p1 ** 2) * (1 - norm_p2 ** 2)))
    except ValueError:
        return np.inf  # Return a large value if arccosh domain issue arises


def find_circle_intersection_angles(circle):
    (h, k), r = circle

    # Define the system of equations to solve
    def equations(p):
        x, y = p
        return [
            (x - h) ** 2 + (y - k) ** 2 - r ** 2,  # Distance from the center of the given circle
            x ** 2 + y ** 2 - 1  # Ensuring the point lies on the unit circle
        ]

    # Provide two initial guesses (roughly opposite each other on the unit circle)
    initial_guesses = [
        (1, 0),  # First guess (on the unit circle)
        (-1, 0)  # Second guess (opposite side of the unit circle)
    ]

    # Solve the system of equations for both initial guesses
    solutions = []
    for guess in initial_guesses:
        solution = fsolve(equations, guess)
        solutions.append(solution)

    # Calculate angles from the center of the given circle to the intersection points
    angles = []
    for sol in solutions:
        x, y = sol
        angle = np.arctan2(y - k, x - h)  # Angle from the given circle center
        angles.append(angle)

    return angles


def brute_guessing_p2(p1, circle, d, num_samples):
    (h, k), r = circle
    limit_angles = find_circle_intersection_angles(circle)
    angles = np.linspace(limit_angles[1], limit_angles[0])

    limit_points = [[h + r * np.cos(limit_angles[0]), k + r * np.sin(limit_angles[0])],
                    [h + r * np.cos(limit_angles[1]), k + r * np.sin(limit_angles[1])]]
    limit_points = np.array(limit_points)
    three_points = np.vstack((limit_points, p1))
    top_down_plot(circle, three_points)

    candidate_points = np.array([h + r * np.cos(angles), k + r * np.sin(angles)]).transpose()

    dists = np.array([hyperbolic_distance(p1, p2) for p2 in candidate_points])

    plt.plot(angles, dists)
    plt.show()

    # Initialize best variables
    best_p2 = None
    best_distance_error = np.inf

    # Evaluate each candidate point
    for p2 in candidate_points:
        distance = hyperbolic_distance(p1, p2)
        error = np.abs(distance - d)

        # Update if this point is better
        if error < best_distance_error:
            best_distance_error = error
            best_p2 = p2

    return best_p2


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


def top_down_plot(circle, points):
    # Unpack the input
    center, radius = circle
    fig, ax = plt.subplots()

    # Plot the main circle
    circle_plot = plt.Circle(center, radius, color='blue', fill=False, linewidth=2)
    ax.add_artist(circle_plot)

    # Plot the points
    #ax.plot(s_1[0], s_1[1], 'bo')
    #ax.plot(s_2[0], s_2[1], 'bo')
    #ax.plot(prediction_1[0], prediction_1[1], 'go', label='s_1')  # green point for s_1
    #ax.plot(prediction_2[0], prediction_2[1], 'ro', label='s_2')  # red point for s_2
    for p in points:
        ax.plot(p[0], p[1], 'o')

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

    _, angle_guess_1 = MiniMap.translate_along_circle(point, circle, 1./circle[1])
    _, angle_guess_2 = MiniMap.translate_along_circle(point, circle, -1./circle[1])
    rho_squared = compute_rho_squared(z)

    #solution_1 = fsolve(circle_equation, initial_guess_1, args=(circle, rho_squared))
    #solution_2 = fsolve(circle_equation, initial_guess_2, args=(circle, rho_squared))
    solution_1 = brute_guessing_p2(point, circle, 0.1, 1000)
    #solution_2 = brute_guessing_p2(point, circle, 0.1, 1000)

    if printt:
        print(f"c = ({circle[0][0]:.2f}, {circle[0][1]:.2f}), r = {circle[1]:.2f}, point = ({point[0]:.2f}, {point[1]:.2f}), p2 = ({solution_1[0]:.2f}, {solution_1[1]:.2f}), distance = {hyperbolic_distance(point, solution_1)}")

    if plot:
        curve_min = find_z0(circle)
        #plot_instance(circle, z, solution_1, solution_2)
        top_down_plot(circle, [point, solution_1, solution_2])

    return solution_1, solution_2


def generate_randomized_data(num_samples, printt=False, plot=False):
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


def generate_grid_data(n, num_levels):
    linspace = np.linspace(start=0., stop=1., num=n)
    maze, explorer = DynamicMaze.get_empty_map(num_levels)
    tile_keys = [key for key, value in maze.adjacency_map.items() if value is not None]
    grid = np.zeros((n, n, len(tile_keys), 2))  # [i, j, tile_key]

    for i in range(n):
        for j in range(n):
            for key in tile_keys:
                if len(key) == 0:
                    p_cartesian = 0.5 - np.array([linspace[i], linspace[j]])
                else:
                    pass#p_past =





if __name__ == '__main__':
    generate_randomized_data(num_samples=3, printt=True, plot=False)
    #generate_grid_data(n=10, num_levels=2)




