'''
Main class for creating the Agent object and updating its properties.
'''
from Brain import Brain
from Cast import RayCaster3D
import numpy as np
import math

import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

MAX_DISTANCE            =       float(config['DEFAULT']['MAX_DISTANCE'])
HORIZONTAL_FOV          =       float(config['DEFAULT']['HORIZONTAL_FOV'])
HORIZONTAL_RAYS         =       int(config['DEFAULT']['HORIZONTAL_RAYS'])
THRESHOLD               =       float(config['DEFAULT']['THRESHOLD'])

class Agent:
    def __init__(self, brain=None, initial_position=np.random.uniform(-10, 10, size=3), initial_heading=np.random.uniform(0, 2*math.pi)):
        if brain is None:
            self.brain = Brain()  # Initialize a new Brain if one isn't provided
        else:
            self.brain = brain  # Use the provided Brain instance
        
        initial_position[2] = 0  # Set z-axis to 0
        self.position = np.array(initial_position)  # Similar to Unity's Transform component
        self.ray_caster = RayCaster3D(position=self.position,
                                      horizontal_fov=np.radians(HORIZONTAL_FOV), 
                                      max_distance=MAX_DISTANCE, 
                                      horizontal_rays=HORIZONTAL_RAYS)
        self.trajectory = [self.position.tolist()]  # Store the initial position
        self.heading = initial_heading
        self.has_reached_target = False


    def is_within_bounds(self, position):
            # Define bounds, could be based on the same values used to initialize position randomly
            # Currently hard set to match initial_position vector
            # Will change when z-axis is introduced
            bounds = np.array([[-10, 10], [-10, 10], [0, 0]])
            return np.all((position >= bounds[:, 0]) & (position <= bounds[:, 1]))

    def update(self, target_position):
        # Check if the target is reached and stop further updates
        if np.linalg.norm(self.position - target_position) < THRESHOLD:
            self.has_reached_target = True
            #print("Reached the target!")
            return  # End the update early
        
        if self.has_reached_target:
            return  # Skip updating if the target has already been reached
        # Use ray caster to 'observe' the target
        hits, distances = self.ray_caster.cast_rays(target_position)

        # Combine ray cast information with any other inputs your brain needs
        sensory_inputs = np.concatenate((distances, hits.astype(float)))

        # Pass sensory inputs to the brain for processing
        self.brain.step(sensory_inputs)

        # New brain output handling
        forward_velocity, turning_angle = self.brain.final_outputs[:2]  # Assume the brain provides a forward velocity and a turning angle
        
        # Update the heading
        self.heading += turning_angle

        # Convert the heading and forward velocity into a velocity vector
        velocity = np.array([
            math.cos(self.heading) * forward_velocity,
            math.sin(self.heading) * forward_velocity,
            0  # Z-axis remains unchanged
        ])

        self.position += velocity  # Update the position
        # Check if the new position is within bounds
        if not self.is_within_bounds(self.position):
            # If not within bounds, prevent movement
            self.position -= velocity  # Revert to previous position

        self.ray_caster.position = self.position  # Update ray caster position with new position
        self.trajectory.append(self.position.tolist())  # Store the new position

    def reset(self):
            # Reinitialize the position to a new random starting position
            initial_position = np.random.uniform(-10, 10, size=3)
            initial_position[2] = 0  # Set z-axis to 0
            self.position = np.array(initial_position)

            # Reset the trajectory list to only include the initial position
            self.trajectory = [self.position.tolist()]

            # Reset the ray caster's position
            self.ray_caster.position = self.position