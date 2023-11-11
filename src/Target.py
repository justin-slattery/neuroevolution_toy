'''
Main class for creating the Target object and updating its properties.
'''

import numpy as np

class Target:
    def __init__(self, initial_position=np.random.uniform(-25, 25, size=3)):
        initial_position[2] = 0  # Set z-axis to 0
        self.position = np.array(initial_position)  # Similar to Unity's Transform component
    
    # Update methods if the object has its own behavior, otherwise, this could be a static object
