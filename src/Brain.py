'''
Main class for creating the Brain object and updating its properties.
'''

# # Generates the code for the agent brain
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
        # Sets the number of input neurons based on rays (distance and hit/miss for each ray)
        self.input_size = 2 * num_rays
        # Sets the number of output neurons (x, y, z positioning)
        self.output_size = num_outputs
        # Total network size includes both input neurons and other neurons
        self.net_size = self.input_size + self.output_size
        # Sets brain step size compared to simulation duration
        self.step_size = step_size
        # Initializes the CTRNN
        self.network = CTRNN(size=self.net_size, step_size=self.step_size)
        self.hist_outputs = []
        # Initializes the network
        self.network.randomize_outputs(0.1, 0.2)

    def step(self, external_inputs):
        self.hist_outputs = list(self.hist_outputs)

        # Ensure external_inputs is only for input neurons
        if len(external_inputs) != self.input_size:
            raise ValueError(f"Expected external inputs of length {self.input_size}, got {len(external_inputs)}")

        # Append zero values for output neurons since they are not driven by external inputs
        network_inputs = np.concatenate((external_inputs, np.zeros(self.output_size)))

        # Step through network
        for _ in range(int(self.run_duration / self.step_size)):
            self.network.euler_step(network_inputs)  # Pass the network inputs to the CTRNN
            # Only record the outputs (last self.output_size values)
            self.hist_outputs.append(self.network.outputs[-self.output_size:])
        
        self.hist_outputs = np.asarray(self.hist_outputs)

    def plot(self):
        # Plot oscillator output
        plt.plot(np.arange(0, self.run_duration, self.step_size), self.hist_outputs[:,0])
        #plt.plot(np.arange(0, self.run_duration, self.step_size), self.hist_outputs[:,1])
        plt.xlabel('Time')
        plt.ylabel('Neuron outputs')
        plt.show()

# brain = Brain()
# # set value for network taus
# brain.network.taus = np.ones(brain.network.size)
# print(brain.network.taus)
# # set random value for network biases
# brain.network.biases = np.random.uniform(-1, 1, brain.network.size)
# print(brain.network.biases)
# # set random value for network weights
# brain.network.weights = csr_matrix(np.random.uniform(-2, 2, (brain.network.size, brain.network.size)))
# print(brain.network.weights)

# brain.step()
# brain.plot()