'''
Main class for creating the Brain object and updating its properties.
'''
from CTRNN import CTRNN
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt

import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

RUN_DURATION              =       int(config['DEFAULT']['RUN_DURATION'])
NUM_OUTPUTS               =       int(config['DEFAULT']['NUM_OUTPUTS'])
HORIZONTAL_RAYS           =       int(config['DEFAULT']['HORIZONTAL_RAYS'])
STEP_SIZE                 =       float(config['DEFAULT']['STEP_SIZE'])

class Brain:
    def __init__(self, run_duration=RUN_DURATION, step_size=STEP_SIZE, num_rays=HORIZONTAL_RAYS, num_outputs=NUM_OUTPUTS):
        # Sets the simulation run duration
        self.run_duration = run_duration
        # Sets the number of input neurons based on rays (distance for each ray, only triggered on a hit)
        self.input_size = num_rays
        # Sets the number of output neurons (x, y, z positioning)
        self.output_size = num_outputs
        # Total network size includes both input neurons and other neurons
        self.net_size = self.input_size + self.output_size
        # Sets brain step size compared to simulation duration
        self.step_size = step_size
        # Initializes the CTRNN
        self.network = CTRNN(size=self.net_size, step_size=self.step_size)
        # Initializes the input and output history
        # List version
        self.input_history = []
        self.output_history = []
        # Array version
        # self.input_history = np.zeros((int(self.run_duration/self.step_size), self.input_size))
        # self.output_history = np.zeros((int(self.run_duration/self.step_size), self.output_size))
        self.final_outputs = []
        # Initializes the network
        # self.network.randomize_outputs(0.1, 0.2)

    def step(self, external_inputs):
        # Ensure external_inputs is only for input neurons
        if len(external_inputs) != self.input_size:
            raise ValueError(f"Expected external inputs of length {self.input_size}, got {len(external_inputs)}")
        
        # Append zero values for output neurons since they are not driven by external inputs
        network_inputs = np.concatenate((external_inputs, np.zeros(self.output_size)))

        # Step through network
        for step in range(int(self.run_duration / self.step_size)):
            self.network.euler_step(network_inputs)  # Pass the network inputs to the CTRNN
            # 11/16/2023
            # Neuron has 9 neurons but only 2 outputs
            # 0-6 are input neurons
            self.network.outputs[0] = 0.0
            self.network.outputs[1] = 0.0
            self.network.outputs[2] = 0.0
            self.network.outputs[3] = 0.0
            self.network.outputs[4] = 0.0
            self.network.outputs[5] = 0.0
            self.network.outputs[6] = 0.0
        
            # # List version
            self.input_history.append(network_inputs[:self.input_size])
            self.output_history.append(self.network.outputs[-self.output_size:])
                
            # # Check if any input value is non-zero and print
            # if np.any(external_inputs):
            #     print(f"Step {step}: Non-zero Inputs - {external_inputs}")
        # Record final (2) outputs to be fed back to Agent for movement
        self.final_outputs = self.network.outputs[-self.output_size:]

        # # Debugging code to print the weights to check connections
        # # Will scale with network size
        # # First, convert to a dense array
        # dense_weights = self.network.weights.toarray()

        # # Now, iterate and print each element in (row, column) format
        # for i in range(dense_weights.shape[0]):
        #     for j in range(dense_weights.shape[1]):
        #         print(f"({i}, {j}) {dense_weights[i, j]}")
        

# brain = Brain()
# # set value for network taus
# brain.network.taus = np.ones(brain.network.size)
# print(brain.network.taus)
# # set random value for network biasesS
# brain.network.biases = np.random.uniform(-1, 1, brain.network.size)
# print(brain.network.biases)
# # set random value for network weights
# brain.network.weights = csr_matrix(np.random.uniform(-2, 2, (brain.network.size, brain.network.size)))
# print(brain.network.weights)

# brain.step()
# brain.plot()