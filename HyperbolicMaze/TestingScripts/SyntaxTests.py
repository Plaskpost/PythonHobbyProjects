import math

def find_on_line(point1, point2, angle3):
    r1, theta1 = point1
    r2, theta2 = point2
    theta3 = (theta2 + angle3) % 360

    # Calculate the difference in radii and the angle between the points
    delta_r = r2 - r1
    delta_theta = (theta3 - theta1) % 360

    # Convert the angle to radians
    delta_theta_radians = math.radians(delta_theta)

    # Calculate the distance to the third point using the law of cosines
    distance = math.sqrt(r1**2 + r2**2 - 2 * r1 * r2 * math.cos(delta_theta_radians))

    return distance

# Example usage:
point1 = (1, 179)  # r = 5, theta = 30 degrees
point2 = (1, 1)  # r = 8, theta = 60 degrees
angle3 = 80     # Angle of the third point with respect to point2

distance_to_third_point = find_on_line(point1, point2, angle3)
print(distance_to_third_point)
