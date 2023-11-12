# # imports
# import numpy as np
# import matplotlib.pyplot as plt
# # importing the CTRNN class
# from CTRNN import CTRNN

# # params
# run_duration = 250
# net_size = 2
# step_size = 0.01

# # set up network
# network = CTRNN(size=net_size,step_size=step_size)
# network.taus = [1.,1.]
# network.biases = [-2.75,-1.75]
# network.weights[0,0] = 4.5
# network.weights[0,1] = 1
# network.weights[1,0] = -1
# network.weights[1,1] = 4.5

# # initialize network
# network.randomize_outputs(0.1,0.2)

# # simulate network
# outputs = []
# for _ in range(int(run_duration/step_size)):
#     network.euler_step([0]*net_size) # zero external_inputs
#     outputs.append([network.outputs[i] for i in range(net_size)])
# outputs = np.asarray(outputs)

# # plot oscillator output
# plt.plot(np.arange(0,run_duration,step_size),outputs[:,0])
# plt.plot(np.arange(0,run_duration,step_size),outputs[:,1])
# plt.xlabel('Time')
# plt.ylabel('Neuron outputs')
# plt.show()

# # Explicitly setting network inputs and outputs
# if len(neuron_input) < self.net_size:
#     for j in range(6, self.net_size):
#         neuron_input.append(0.0001)
    
# self.network.euler_step(neuron_input)
# # Clamping neurons to bootstrap shaping
# # Green Neurons
# self.network.outputs[2] = 0.0
# self.network.outputs[3] = 0.0
# # Blue Neurons
# self.network.outputs[4] = 0.0
# self.network.outputs[5] = 0.0
# # Inter Neurons
# self.network.outputs[6] = 0.0
# self.network.outputs[7] = 0.0
# self.network.outputs[8] = 0.0
# self.network.outputs[9] = 0.0
# #self.network.outputs[10] = 0.0
# # Forward Neuron
# self.network.outputs[11] = 0.0
# # Left/Right Motor Neurons
# #self.network.outputs[12] = 0.0
# #self.network.outputs[13] = 0.0
# # Grasping Neuron
# self.network.outputs[14] = 0.0

# # Restricting full connectivity
# for k in range(self.population_size):
#     c = CTRNN.CTRNN(size=self.network_size, step_size=self.network_speed)
#     # Assign initial weights
#     # Need to clamp blue and green neurons and their 
#     # connections since they are not currently being used
#     for i in range(self.network_size):
#         for j in range(self.network_size):
#             # Prevent input-to-output connections
#             if i < 6 and j > self.network_size-5:
#                 c.weights[i, j] = 0.0001
#             # Prevent output-to-input and
#             # inter-to-input connections
#             elif i > 5 and j < 6:
#                 c.weights[i, j] = 0.0001
#             else:
#                 c.weights[i, j] = [val for val in [random.uniform(-10.0, 10.0) for i in range(1)] if val !=0][0]

# 8 logical processors on laptop
# import multiprocessing

# pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
# print(pool)