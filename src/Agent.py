import numpy as np

class Agent:
    def __init__(self, brain, initial_position):
        self.brain = brain  # This would be analogous to a custom script in Unity
        self.position = np.array(initial_position)  # Similar to Unity's Transform component
        self.trajectory = [self.position.tolist()]  # Store the initial position
        # Other properties like health, stamina, etc., can be added here

    def update(self, target_position):
        self.brain.step()  # Update the neural network
        # Movement logic here will later correspond to what you would handle in Unity's Update method
        # The actual movement will be handled by Unity's physics system

        # Assuming hist_outputs[-1] gives you the latest output of the network and it is a list with three values
        velocity = np.array(self.brain.hist_outputs[-1])  # Convert the last output to a numpy array
        if velocity.shape == (3,):  # Ensure it is a 1D array with three elements
            self.position += velocity  # Update the position
        else:
            # Handle the case where velocity does not have the shape (3,)
            # You might want to log an error or raise an exception here
            print(f"Velocity has an incorrect shape: {velocity.shape}")

        self.trajectory.append(self.position.tolist())  # Store the new position