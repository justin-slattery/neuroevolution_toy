'''
Main class for creating the Agent object and updating its properties.
'''

import numpy as np
from Brain import Brain
from Cast import RayCaster3D

import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

MAX_DISTANCE            =       float(config['DEFAULT']['MAX_DISTANCE'])
HORIZONTAL_FOV          =       float(config['DEFAULT']['HORIZONTAL_FOV'])
HORIZONTAL_RAYS         =       int(config['DEFAULT']['HORIZONTAL_RAYS'])
class Agent:
    def __init__(self, brain=None, initial_position=np.random.uniform(-10, 10, size=3)):
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

    def update(self, target_position):
        # Use ray caster to 'observe' the target
        hits, distances = self.ray_caster.cast_rays(target_position)

        # Combine ray cast information with any other inputs your brain needs
        sensory_inputs = np.concatenate((distances, hits.astype(float)))

        # Pass sensory inputs to the brain for processing
        self.brain.step(sensory_inputs)

        # Assuming brain.outputs gives you the latest output of the network and it is a list with three values
        velocity = np.array(self.brain.hist_outputs[-1])  # Convert the last output to a numpy array
        if velocity.shape == (3,):  # Ensure it is a 1D array with three elements
            velocity[2] = 0  # Ensure z-axis velocity is 0
            #print(f"Output Velocity from Network: {velocity}")
            self.position += velocity  # Update the position
        else:
            # Handle the case where velocity does not have the shape (3,)
            # You might want to log an error or raise an exception here
            print(f"Velocity has an incorrect shape: {velocity.shape}")

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