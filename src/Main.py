from GA import GeneticAlgorithm
from Agent import Agent
from Brain import Brain
from Target import Target
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

POPULATION_SIZE         =       int(config['DEFAULT']['POPULATION_SIZE'])
GENERATIONS             =       int(config['DEFAULT']['GENERATIONS'])
MUTATION_PROBABILITY    =       float(config['DEFAULT']['MUTATION_PROBABILITY'])
MUTATION_STRENGTH       =       float(config['DEFAULT']['MUTATION_STRENGTH'])
ELITISM_FRACTION        =       float(config['DEFAULT']['ELITISM_FRACTION'])
RUN_DURATION            =       int(config['DEFAULT']['RUN_DURATION'])
NET_SIZE                =       int(config['DEFAULT']['NET_SIZE'])
STEP_SIZE               =       float(config['DEFAULT']['STEP_SIZE'])
DATA_DIR                =       str(config['DEFAULT']['DATA_DIR'])

# Define the fitness function
def fitness_function(network):
    # Your code to evaluate the network's fitness goes here
    return np.random.rand()  # Example placeholder

# Usage
ga = GeneticAlgorithm(
    fitness_function=fitness_function,
    population_size=POPULATION_SIZE,
    mutation_prob=MUTATION_PROBABILITY,
    mutation_strength=MUTATION_STRENGTH,
    elite_size_ratio=0.2
)

# Run the genetic algorithm
#best_network = ga.step(num_generations=100)

# Initialize your agent and object here
brain = Brain()
# set value for network taus
brain.network.taus = np.ones(brain.network.size)
# set random value for network biases
brain.network.biases = np.random.uniform(-1, 1, brain.network.size)
# set random value for network weights
brain.network.weights = csr_matrix(np.random.uniform(-2, 2, (brain.network.size, brain.network.size)))

agent = Agent(brain=brain, initial_position=np.random.uniform(-10, 10, size=3))
target = Target(initial_position=np.random.uniform(-10, 10, size=3))

# Run the simulation
num_steps = 50  # Just an example, set this to whatever makes sense for your simulation
some_threshold = 1.0  # The distance considered close enough to "reach" the object

for _ in range(num_steps):
    agent.update(target_position=target.position)
    # Check for condition where agent reaches the object
    if np.linalg.norm(agent.position - target.position) < some_threshold:
        print("Reached the object!")
        break

# The agent's trajectory is recorded in the agent.trajectory attribute
# Now let's plot it using the trajectory data
trajectory = np.array(agent.trajectory)  # Convert to numpy array for easy slicing

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Agent Path')
ax.scatter(*target.position, color='r', label='Object')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

