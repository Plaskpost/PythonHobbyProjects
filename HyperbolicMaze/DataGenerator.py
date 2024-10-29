import numpy as np
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
from collections import deque

import DynamicMaze
import HyperbolicGrid
import config
import Explorer as ex
import MiniMap
import json

"""
This file was used to perform slower computations of nearby tile locations on the Poincaré disc used for the MiniMap.py.
"""

# ----------------- MAIN DATA GENERATING FUNCTIONS -------------------

def generate_neighbor_respecting_grid_data(num_levels, plot=False):
    plt.ioff()

    n = config.num_grid_points
    linspace = np.linspace(start=0., stop=1., num=n)
    maze = DynamicMaze.get_plain_map(num_levels)
    explorer = ex.Player()
    full_dict = {}
    corner_steps = [[['U', 'UR', 'R', 'RU'], ['R', 'RD', 'D', 'DR']], [['U', 'UL', 'L', 'LU'], ['L', 'LD', 'D', 'DL']]]
    d = ['D', 'R', 'U', 'L']

    for i in range(n):
        for j in range(n):
            tile_center_screen_placement = MiniMap.MiniMap.tile_size * (np.array([linspace[i], linspace[j]]) - 0.5)
            if (i == 0 and j == 0) or (i == 0 and j == n-1) or (i == n-1 and j == 0) or (i == n-1 and j == n-1):  # If at a corner
                all_centers = [tile_center_screen_placement]
                for step_sequence in corner_steps[np.sign(i)][np.sign(j)]:
                    probe = ex.Ray(explorer)
                    probe.pos = -tile_center_screen_placement
                    for step in step_sequence:
                        probe.transfer_tile(maze, d.index(step), tile_size=MiniMap.MiniMap.tile_size)
                    probe_tile_center = -probe.pos
                    alternative_tile_center = hyperbolic_traverse(probe_tile_center,
                                                                  HyperbolicGrid.get_reversed_path_string(step_sequence))
                    all_centers.append(alternative_tile_center)

                all_centers = np.array(all_centers)
                tile_center_screen_placement = np.mean(all_centers, axis=0)

            elif i == 0 or j == 0 or i == n-1 or j == n-1:  # If at edge
                step = None
                if i == 0:
                    step = 'R'
                elif i == n-1:
                    step = 'L'
                elif j == 0:
                    step = 'U'
                elif j == n-1:
                    step = 'D'

                probe = ex.Ray(explorer)
                probe.pos = -tile_center_screen_placement
                probe.transfer_tile(maze, d.index(step), tile_size=MiniMap.MiniMap.tile_size)
                probe_tile_center = -probe.pos
                alternative_tile_center = hyperbolic_traverse(probe_tile_center, HyperbolicGrid.opposite_of(step))
                tile_center_screen_placement = np.mean(np.array([tile_center_screen_placement, alternative_tile_center]), axis=0)


            position_dict = find_coordinates_and_circles(maze, explorer, tile_center_screen_placement)

            # Traverse the dict to find all available corner points
            corner_dict = {}
            corner_indices = [[0, 1], [0, 0], [1, 0], [1, 1]]
            for tile_key, (tile_center, circle, up_angle) in position_dict.items():

                corner_positions = np.zeros((4, 2))
                for corner_index in range(4):
                    nearby_tile_centers = [tile_center]
                    step_sequences = corner_steps[corner_indices[corner_index][0]][corner_indices[corner_index][1]]
                    plott = False
                    if tile_key == 'RU' and i == 1 and j == 1 and plot:
                        corner_names = ['bottom-right', 'top-right', 'top-left', 'bottom-left']
                        fig, ax = plt.subplots()
                        ax.set_title(corner_names[corner_index])
                        scatterplot(np.array([tile_center]), color='red')
                        plott = True
                    for step_sequence in step_sequences:
                        other_tile_center = hyperbolic_traverse(tile_center, step_sequence, up_angle, plot=plott)  # Doesn't require mapped out tiles.
                        nearby_tile_centers.append(other_tile_center)
                    nearby_tile_centers = np.array(nearby_tile_centers)
                    corner_positions[corner_index, :] = np.mean(nearby_tile_centers, axis=0)

                    if plott:
                        for k, (x, y) in enumerate(nearby_tile_centers):
                            if k == 0:
                                continue
                            label = corner_steps[corner_indices[corner_index][0]][corner_indices[corner_index][1]][k-1]
                            ax.text(x, y, label, fontsize=10, color='black',
                                    verticalalignment='bottom', horizontalalignment='right')

                plt.show()
                corner_dict[tile_key] = [list(corner_position) for corner_position in corner_positions]

            if not full_dict:
                for key in position_dict.keys():
                    full_dict[key] = [[[[0. for _ in range(2)] for _ in range(5)] for _ in range(n)] for _ in range(n)]

            for key in full_dict.keys():
                full_dict[key][i][j][0] = list(position_dict[key][0])
                full_dict[key][i][j][1:5] = list(corner_dict[key])

        print(f"{(i + 1) * n} out of {n * n} operations complete.")

    if plot:
        plt.show()


    with open(f'SavedModels/GridMapDict{config.num_grid_points}x{config.num_grid_points}.json', 'w') as json_file:
        json.dump(full_dict, json_file)

    print(f"Saved model GridMapDict{config.num_grid_points}x{config.num_grid_points}.json updated.")

def old_generate_grid_data(num_levels):
    n = config.num_grid_points
    linspace = np.linspace(start=0., stop=1., num=n)
    maze = DynamicMaze.get_plain_map(num_levels)
    explorer = ex.Player()
    full_dict = {}

    for i in range(n):
        for j in range(n):
            tile_center_screen_placement = MiniMap.MiniMap.tile_size * (np.array([linspace[i], linspace[j]]) - 0.5)
            position_dict = find_coordinates_and_circles(maze, explorer, tile_center_screen_placement)
            if not full_dict:
                for key in position_dict.keys():
                    full_dict[key] = [[[0. for _ in range(2)] for _ in range(n)] for _ in range(n)]

            for key in full_dict.keys():
                full_dict[key][i][j] = list(position_dict[key][0])

        print(f"{(i+1)*n} out of {n*n} operations complete.")

    with open(f'SavedModels/GridMapDict{config.num_grid_points}x{config.num_grid_points}.json', 'w') as json_file:
        json.dump(full_dict, json_file)

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

        #s1 = compute_outputs(p_cartesian, circle, z[i], printt=printt, plot=plot)  # Outdated code
        x1[i], y1[i] = None


    X = np.stack([p_r, p_phi, N_phi, z], axis=1)
    y = np.stack([x1, y1, x2, y2], axis=1)

    return X, y

def find_duplicates(num_levels):
    adjacency_map = {'': None}
    duplicates = {}
    position_dict = {}
    origin = np.array([0, 0])
    epsilon = 0.005

    for level in range(num_levels):
        none_tiles = []
        for tile_key, v in adjacency_map.items():
            if v is None:
                none_tiles.append(tile_key)

        for tile_key in none_tiles:
            HyperbolicGrid.iterative_registration(tile_key, adjacency_map, False)

        for tile_key, neighbors in adjacency_map.items():
            if neighbors is None:
                continue
            if tile_key not in position_dict:
                position = hyperbolic_traverse(origin, tile_key)
                position_dict[tile_key] = position

        for i, (tile_key_1, position_1) in enumerate(position_dict.items()):
            for j, (tile_key_2, position_2) in enumerate(position_dict.items()):
                if j >= i:
                    break
                difference = np.linalg.norm(position_2 - position_1)

                # Debug stop
                if tile_key_1 == 'ULLLD' and tile_key_2 == "LUURRD" or tile_key_2 == 'ULLLD' and tile_key_1 == "LUURRD":
                    a = 0
                if tile_key_1 == 'DRRUUL' and tile_key_2 == "RDDDL" or tile_key_2 == 'DRRUUL' and tile_key_1 == "RDDDL":
                    a = 0

                if difference < epsilon:
                    k = 0
                    while k < len(tile_key_1) and k < len(tile_key_2) and tile_key_1[k] == tile_key_2[k]:
                        k += 1
                    position_1b = hyperbolic_traverse(origin, tile_key_1[k:])
                    position_2b = hyperbolic_traverse(origin, tile_key_2[k:])
                    difference = np.linalg.norm(position_2b - position_1b)
                    if difference > epsilon:
                        continue

                    #removed_key = HyperbolicGrid.link_duplicates(adjacency_map, tile_key_2, tile_key_1)
                    duplicates[tile_key_1] = tile_key_2
                    duplicates[tile_key_2] = tile_key_1
                    #del position_dict[removed_key]

    print(duplicates)


# ---------------------- KEY OPERATING FUNCTIONS -------------------------

def find_coordinates_and_circles(maze, explorer, tile_center_screen_placement):
    all_steps = {}
    visited = {''}

    probe = explorer.__copy__()
    facing_angle = np.pi/2.
    normal = np.array([-1., 0.])
    circle = MiniMap.find_circle(tile_center_screen_placement, normal)
    circular_direction = MiniMap.get_circular_direction(circle, tile_center_screen_placement, facing_angle)

    queue = deque([[probe, maze, all_steps, visited, circle, tile_center_screen_placement, circular_direction, facing_angle, '']])

    while queue:
        tile_pack = queue.popleft()
        find_tile_specific_coordinates(queue, *tile_pack)

    return all_steps

def find_tile_specific_coordinates(queue, probe, maze, all_steps, visited, current_circle, projection_coord,
                                   current_circular_direction, current_facing_angle, local_journey):
    """
    Loops through three of the tile's neighbors in order: forward, right, left (previous tile as reference), finds
    the coordinated of the blocks to draw and adds neighboring tiles to the queue.

    :param queue: Queue with argument lists as elements that lines up the tiles to be searched in BFS order.
    :param maze: DynamicMaze object.
    :param probe: Explorer object to maintain correct orientation in the maze
    :param all_steps: Now rather a dict of relative_journey: position_coord.
    :param current_circle: (center, radius). Circle along which the probe is currently searching.
    :param visited: A set of visited tiles.
    :param projection_coord:
    :type projection_coord: numpy.ndarray
    :param current_circular_direction: Which way is forward on the current circle. 1 for counter-clockwise and -1 for clockwise.
    :param current_facing_angle: Angle of the direction forward along the current circle arc.
    :param local_journey: A string of letters (D, R, U, L) that tells how the probe has traversed relative the player tile.
    :return:
    """

    up_facing_angle = None

    # Find the circle perpendicular to the current one
    current_normal = MiniMap.find_normal(projection_coord, current_circle)
    perpendicular_normal = MiniMap.rotate_normal(projection_coord, current_normal)
    perpendicular_circle = MiniMap.find_circle(projection_coord, perpendicular_normal)

    #Ssome for-loop preparations
    exploration_directions = ['D', 'R', 'U', 'L']
    forward = exploration_directions.index(local_journey[-1]) if local_journey != '' \
        else int((1 + np.round(4. * current_facing_angle / (2. * np.pi))) % 4)
    right = (forward - 1) % 4
    left = (forward + 1) % 4
    for direction_index in range(4):
        # Set parameters
        if direction_index == forward:
            circle = current_circle
            facing_angle = current_facing_angle
            circular_direction = current_circular_direction
        elif direction_index == right or direction_index == left:
            circle = perpendicular_circle
            facing_angle = current_facing_angle - np.pi / 2. if direction_index == right else current_facing_angle + np.pi / 2.
            circular_direction = MiniMap.get_circular_direction(circle, projection_coord, facing_angle)
        else:  # Backwards (only from first tile).
            circle = current_circle
            facing_angle = current_facing_angle + np.pi
            circular_direction = -current_circular_direction

        # Track which angle points to UP.
        if exploration_directions[direction_index] == 'U':
            up_facing_angle = facing_angle

        # Situations with no need to proceed
        if (direction_index - 2) % 4 == forward and local_journey != '':
            continue  # Skip steps backwards.
        probe_ahead = probe.__copy__()
        if not probe_ahead.transfer_tile(maze, direction_index):
            continue  # Skip any step towards a tile that hasn't been generated in the maze.

        # The call we all came here for
        projection_coord_ahead, translation_angle = brute_guessing_p2(projection_coord, circle, facing_angle)

        # Lastly, add neighbors to queue.
        if probe_ahead.pos_tile not in visited:
            facing_angle_ahead = facing_angle + translation_angle  # First update the facing angle.
            local_journey_ahead = local_journey + exploration_directions[direction_index]
            visited.add(probe_ahead.pos_tile)
            queue.append([probe_ahead, maze, all_steps, visited, circle, projection_coord_ahead,
                          circular_direction, facing_angle_ahead, local_journey_ahead])

    # Add the information to the dict as a final step
    all_steps[local_journey] = (projection_coord, current_circle, up_facing_angle)

def brute_guessing_p2(p1, circle, facing_angle, d=1.06, ax=None, plot=False, show=False):
    num_guessing_points = 100
    (h, k), r = circle

    if np.isinf(r):  # If radius infinite we're on a line.
        p2_limit = np.array([np.cos(facing_angle), np.sin(facing_angle)])
        limit_points = np.array([p1, p2_limit])
        active_dim = np.argmax(np.sum(np.abs(limit_points), axis=0))
        limit_values = limit_points[:, active_dim]
        candidate_values = np.linspace(limit_values[0], limit_values[1], num_guessing_points)

        closest_values = np.zeros(2)  # Placeholder for angles to points to the left and right of the true points.
        corresponding_distances = np.zeros(2)
        p2 = np.zeros(2)
        prev_distance = 0

        # Find the closest angles.
        for i in range(1, num_guessing_points):
            p2[active_dim] = candidate_values[i]
            distance = hyperbolic_distance(p1, p2)
            if np.sign(distance - d) != np.sign(prev_distance - d):
                closest_values[0] = candidate_values[i - 1]
                closest_values[1] = candidate_values[i]
                corresponding_distances[0] = prev_distance
                corresponding_distances[1] = distance
                break

            prev_distance = distance

        p2[active_dim] = MiniMap.linearization_estimation(closest_values, corresponding_distances, d)
        angle_change = 0.

    else:  # Not infinite radius and therefore a circle shape
        limit_angles = find_circle_intersection_angles(circle)
        limit_points = [[h + r * np.cos(limit_angles[0]), k + r * np.sin(limit_angles[0])],
                        [h + r * np.cos(limit_angles[1]), k + r * np.sin(limit_angles[1])]]
        limit_points = np.array(limit_points)
        limit_angles = np.array(limit_angles)

        # Find in which direction the facing angle points
        facing_vector = to_cartesian(1., facing_angle)
        _, angle_from_circle_to_p1 = to_polar(p1[0] - h, p1[1] - k)
        dot_0 = np.dot(facing_vector, limit_points[0] - p1)
        if dot_0 > 0:  # If we seem to be facing limit point 0.
            limit_points[1] = p1
            limit_angles[1] = angle_from_circle_to_p1
        else:  # If we seem to be facing limit point 1.
            limit_points[0] = p1
            limit_angles[0] = angle_from_circle_to_p1

        angles = shortest_angle_sampling(limit_angles, num_guessing_points)
        candidate_points = np.array([h + r * np.cos(angles), k + r * np.sin(angles)]).transpose()

        closest_angles = np.zeros(2)  # Placeholder for angles to points to the left and right of the true points.
        corresponding_distances = np.zeros(2)
        prev_distance = hyperbolic_distance(p1, candidate_points[0])

        # Find the closest angles.
        for i in range(1, num_guessing_points):
            distance = hyperbolic_distance(p1, candidate_points[i])
            if np.sign(distance - d) != np.sign(prev_distance - d):
                closest_angles[0] = angles[i-1]
                closest_angles[1] = angles[i]
                corresponding_distances[0] = prev_distance
                corresponding_distances[1] = distance
                break

            prev_distance = distance

        # Use linearization for a sophisticated guess on the true angle.
        if np.abs(closest_angles[1] - closest_angles[0]) > np.pi:
            closest_angles = closest_angles % (2.*np.pi)
        p2_angle = MiniMap.linearization_estimation(closest_angles, corresponding_distances, d)
        p2 = np.array([h + r * np.cos(p2_angle), k + r * np.sin(p2_angle)])
        angle_change = p2_angle - angle_from_circle_to_p1

        if plot:
            if ax is None:
                ax = plt.gca()
            three_points = np.vstack((limit_points, p1))
            top_down_plot(circle, three_points, color='green', show=False)
            plt.scatter(p2[0], p2[1], s=30, color='black')
            plt.plot([h, limit_points[0][0]], [k, limit_points[0][1]], color='b', linestyle='--')
            plt.plot([h, limit_points[1][0]], [k, limit_points[1][1]], color='b', linestyle='--')

            #distances = [hyperbolic_distance(p1, p2) for p2 in candidate_points]
            #plt.plot(angles, distances, color='b')
            #plt.plot(angles, d*np.ones(num_guessing_points), color='black', linestyle='--', linewidth=2)
            #guessed_distance = hyperbolic_distance(p1, p2)
            #plt.plot(angles, guessed_distance * np.ones(num_guessing_points), color='red')
            if show:
                plt.show()

    return p2, angle_change

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
    angle_to_circle_center = np.arctan2(k, h)
    initial_guesses = [
        to_cartesian(1., angle_to_circle_center + 1),
        to_cartesian(1., angle_to_circle_center - 1)
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

    if np.abs(angles[0] - angles[1]) < 0.00001:
        print(f"WARNING: Failed to find different intersection angles. Found only at {np.degrees(angles[0])} degrees.")

    return angles

def shortest_angle_sampling(limit_angles, num_guessing_points):
    limit_angles = np.mod(limit_angles + np.pi, 2 * np.pi) - np.pi
    angle_diff = limit_angles[1] - limit_angles[0]

    if angle_diff > np.pi:
        angle_diff -= 2 * np.pi
    elif angle_diff < -np.pi:
        angle_diff += 2 * np.pi

    angles = np.linspace(0, angle_diff, num_guessing_points)
    angles += limit_angles[0]
    angles = np.mod(angles + np.pi, 2 * np.pi) - np.pi

    return angles

def hyperbolic_traverse(starting_point, step_sequence, facing_angle=np.pi/2., plot=False):
    """
    Starts from starting_point in the unit circle, follows the hyperbolic curves along step_sequence and returns the
    coordinates at the end.

    :param starting_point: Cartesian coordinates for the starting point in the unit circle.
    :param step_sequence: A string with steps ('D', 'R', 'U', 'L') to take.
    :param facing_angle: Initial facing angle.
    :param plot: Debug parameter. True if the program should plot the curves and the point.
    :returns: Coordinates at the destination.
    """
    turn_sequence = HyperbolicGrid.drul_to_brfl(step_sequence)
    brfl = ['B', 'R', 'F', 'L']
    angle_changes = [np.pi, -np.pi/2., 0., np.pi/2.]
    point = starting_point

    for step in turn_sequence:
        facing_angle += angle_changes[brfl.index(step)]
        normal = to_cartesian(1., facing_angle + np.pi/2.)
        circle = MiniMap.find_circle(point, normal)

        point, angle_diff = brute_guessing_p2(point, circle, facing_angle, plot=False, show=False)
        facing_angle += angle_diff

        if plot:
            circle_plot(circle, ax, show=False)
            scatterplot(np.array([point]), ax, show=False)

    return point


# --------------------- MATH OPERATIONS ---------------------

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

def find_z0(circle):
    c, r = circle
    center_r, center_phi = to_polar(c[0], c[1])
    (x0, y0) = to_cartesian(center_r-r, center_phi)
    z0 = z_t(x0, y0)
    return x0, y0, z0


def z_t(x_t, y_t):
    rho_squared = x_t ** 2 + y_t ** 2
    return np.arccosh(1 + (2 * rho_squared) / (1 - rho_squared))

def compute_rho_squared(z):  # Compute x^2 + y^2 from z
    return (np.cosh(z) - 1) / (np.cosh(z) + 1)

def circle_equation(vars, circle, rho_squared):
    x, y = vars
    (h, k), r = circle
    return [
        (x - h)**2 + (y - k)**2 - r**2,
        x**2 + y**2 - rho_squared
    ]

def to_cartesian(r, phi):
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return np.array([x, y])

def to_polar(x, y):
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return np.array([r, phi])


# --------------------------- PLOTTING -----------------------------

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

def top_down_plot(circle, points, ax=None, color='blue', show=True):
    # Unpack the input
    center, radius = circle
    if ax is None:
        ax = plt.gca()

    if np.isinf(radius):
        if np.isinf(center[0]):
            ax.plot([center[1], center[1]], [-1, 1], color=color, linewidth=1)
        elif np.isinf(center[1]):
            ax.plot([-1, 1], [center[0], center[0]], color=color, linewidth=1)
    else:
        plotted_circle = plt.Circle(center, radius, color=color, fill=False, linewidth=1)
        ax.add_artist(plotted_circle)

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

    if show:
        plt.show()

def square_plot(points, ax=None, show=False):
    # If no axis is provided, get the current axis
    if ax is None:
        ax = plt.gca()

    # Close the polygon by adding the first point at the end
    points = np.vstack([points, points[0, :]])
    ax.plot(points[:, 0], points[:, 1])

    # Add the unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_unit_circle = np.cos(theta)
    y_unit_circle = np.sin(theta)
    ax.plot(x_unit_circle, y_unit_circle, 'k--', label='Unit Circle')  # dotted black line for unit circle

    # Set limits and aspect ratio
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')

    # Show the plot only if requested
    if show:
        plt.show()

def scatterplot(points, ax=None, color='black', show=False):
    # If no axis is provided, get the current axis
    if ax is None:
        ax = plt.gca()

    # Plot the scatter points
    ax.scatter(points[:, 0], points[:, 1], color=color)

    # Set limits
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)

    # Show the plot only if requested
    if show:
        plt.show()

def circle_plot(circle, ax=None, show=False):
    # If no axis is provided, get the current axis
    if ax is None:
        ax = plt.gca()

    center, radius = circle

    # Add the unit circle
    theta = np.linspace(0, 2 * np.pi, 100)
    x_unit_circle = np.cos(theta)
    y_unit_circle = np.sin(theta)
    ax.plot(x_unit_circle, y_unit_circle, 'k--', label='Unit Circle')  # dotted black line for unit circle

    # Check for infinite radius (line)
    if np.isinf(radius):
        if np.isinf(center[0]):  # Vertical line
            ax.plot([center[1], center[1]], [-1, 1], color='blue', linewidth=1)
        elif np.isinf(center[1]):  # Horizontal line
            ax.plot([-1, 1], [center[0], center[0]], color='blue', linewidth=1)
    else:
        # Plot the circle
        the_plot = plt.Circle(center, radius, color='blue', fill=False, linewidth=1)
        ax.add_artist(the_plot)

    # Set limits and aspect ratio
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_aspect('equal')

    # Show the plot only if requested
    if show:
        plt.show()


# ------------------------ RUNNING ----------------------------

if __name__ == '__main__':
    fig, ax = None, None
    generate_neighbor_respecting_grid_data(num_levels=4, plot=False)
    #save_duplicates(7)


