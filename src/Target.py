import numpy as np

class Target:
    def __init__(self, initial_position):
        self.position = np.array(initial_position)  # Similar to Unity's Transform component
    
    # Update methods if the object has its own behavior, otherwise, this could be a static object
