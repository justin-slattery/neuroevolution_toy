# # Generates the code for the agent brain
from CTRNN import CTRNN
from scipy.sparse import csr_matrix
import numpy as np
import matplotlib.pyplot as plt

import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

RUN_DURATION              =       int(config['DEFAULT']['RUN_DURATION'])
NET_SIZE                  =       int(config['DEFAULT']['NET_SIZE'])
STEP_SIZE                 =       float(config['DEFAULT']['STEP_SIZE'])

class Brain:
    def __init__(self, run_duration=RUN_DURATION, net_size=NET_SIZE, step_size=STEP_SIZE): # step_size should never be < 0.02; should be 1 > x > 0.02
        # Sets the simulation run duration
        self.run_duration = run_duration
        # Sets network size
        self.net_size = net_size
        # Sets brain step size compared to simulation duration
        self.step_size = step_size
        # Initializes the CTRNN
        self.network = CTRNN(size=self.net_size, step_size=self.step_size)
        self.hist_outputs = []
        # Initializes the network
        self.network.randomize_outputs(.1, .2)

    def step(self):
        self.hist_outputs = list(self.hist_outputs)

        # Step through network
        for _ in range(int(self.run_duration/self.step_size)):
            self.network.euler_step([0]*self.net_size) # zero external_inputs
            self.hist_outputs.append([self.network.outputs[i] for i in range(self.net_size)])
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