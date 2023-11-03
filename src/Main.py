from GA import GeneticAlgorithm
import numpy as np

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
best_network = ga.step(num_generations=100)