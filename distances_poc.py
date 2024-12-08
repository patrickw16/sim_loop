import numpy as np

def calculate_distance_threshold(distances, dt, j, ego_deceleration):
    """
    Calculate the distance threshold based on the given distances, time step, 
    current index, and ego deceleration.

    Args:
        distances (list): A list of distances.
        dt (float): The time step.
        j (int): The current (time) index.
        ego_deceleration (float): The ego deceleration.

    Returns:
        float: The calculated distance threshold.
    """
    if not distances or len(distances) < 2:
        return 500  # default distance threshold

    distances_without_zeros = [item for item in distances if item != 0]
    if len(distances_without_zeros) < 2:
        return 500  # default distance threshold

    not_zero_indices = [index for index, value in enumerate(distances[0:-1]) if value != 0]
    if not not_zero_indices:
        return 500  # default distance threshold

    distances_delta = distances[-1] - distances[max(not_zero_indices)]
    time_delta = (j - max(not_zero_indices)) * dt
    delta_v = distances_delta / time_delta

    if delta_v == 0:
        return 500  # default distance threshold
    else:
        return np.square(delta_v) / (2 * ego_deceleration)

# Example usage:
distances = [50, 0, 0, 48, 44, 42, 0, 0, 38, 36, 34, 32, 0, 0, 30, 28, 26, 24, 20, 16, 14, 0, 0, 12, 11, 10]
dt = 0.1
distances_so_far = [50,0,0,48,44]
j = 4
ego_deceleration = 5.5
threshold = calculate_distance_threshold(distances_so_far, dt, j, ego_deceleration)
print("Distance Threshold:", threshold)


