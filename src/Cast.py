'''
Main class for creating the RayCaster object to be used by the Agent and performing the necessary
calculations to track hits and misses.
'''

import numpy as np

class RayCaster3D:
    def __init__(self, position, horizontal_fov, max_distance, horizontal_rays):
        self.position = np.array(position)
        self.horizontal_fov = horizontal_fov
        # Vertical FOV and rays are set for future use.
        self.vertical_fov = 0  
        self.max_distance = max_distance
        self.horizontal_rays = horizontal_rays
        self.vertical_rays = 1  # Set to 1 for now
        self.ray_angles_horizontal = np.linspace(-horizontal_fov / 2, horizontal_fov / 2, horizontal_rays)
        self.ray_angles_vertical = np.array([0])  # Single ray along the horizontal plane

    def calculate_ray_direction(self, horizontal_angle, vertical_angle):
        # Calculate the direction vector for a ray given the angles
        # Horizontal angle is all that matters for now
        direction = np.array([
            np.cos(horizontal_angle),
            np.sin(horizontal_angle),
            np.sin(vertical_angle)  # Will be zero in the current setup
        ])
        return direction / np.linalg.norm(direction)

    def angle_between(self, v1, v2):
        # Calculate the angle between two vectors
        v1_u = v1 / np.linalg.norm(v1)
        v2_u = v2 / np.linalg.norm(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    def cast_rays(self, target_position):
        # Cast rays towards the target and determine if they hit
        hits = np.zeros(self.horizontal_rays, dtype=bool)
        distances = np.full(self.horizontal_rays, self.max_distance)
        target_vector = target_position - self.position
        target_distance = np.linalg.norm(target_vector)

        for j, horizontal_angle in enumerate(self.ray_angles_horizontal):
            ray_direction = self.calculate_ray_direction(horizontal_angle, 0)  # Vertical angle is 0
            if target_distance <= self.max_distance:
                angle_diff = self.angle_between(target_vector, ray_direction)
                if angle_diff <= np.radians(self.horizontal_fov / 2):
                    hits[j] = True
                    distances[j] = target_distance

        return hits, distances

    # Additional methods can be added here for more complex behaviors and interactions.

# # Example usage
# position = [0, 0, 0]  # Starting position
# horizontal_fov = np.radians(90)  # Field of view in radians
# max_distance = 10  # Max distance the ray can detect
# horizontal_rays = 5  # Number of rays cast horizontally

# ray_caster = RayCaster3D(position, horizontal_fov, max_distance, horizontal_rays)
# target_position = [5, 5, 0]  # Target position

# hits, distances = ray_caster.cast_rays(target_position)
# print("Hits:", hits)
# print("Distances:", distances)
