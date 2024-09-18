import numpy as np


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


# Generate candidate points on the circle, restricted to the unit disc
def generate_candidate_points(h, k, r, num_points=1000):
    candidate_points = []
    angles = np.linspace(0, 2 * np.pi, num_points)

    for theta in angles:
        p2 = np.array([h + r * np.cos(theta), k + r * np.sin(theta)])
        if np.linalg.norm(p2) < 1:  # Only keep points inside the unit circle
            candidate_points.append(p2)

    return np.array(candidate_points)


# Function to find the best p2 by evaluating candidate points
def find_best_p2(p1, h, k, r, d):
    candidate_points = generate_candidate_points(h, k, r)

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


# Example usage:
p1 = np.array([0.06212503, -0.29136784])  # Point p1 (inside the unit circle)
h, k, r = -1.46, -2.18, 2.43  # Circle parameters: center (h, k), radius r
d = 0.01  # Desired hyperbolic distance

# Find the best p2 that satisfies both equations
p2 = find_best_p2(p1, h, k, r, d)
print("The solution p2 is:", p2)

# Verify the hyperbolic distance between p1 and p2
dist = hyperbolic_distance(p1, p2)
print("Hyperbolic distance between p1 and p2:", dist)
