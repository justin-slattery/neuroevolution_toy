'''
Main class for creating the Genetic Algorithm and its required components.
'''
from copy import deepcopy
from Agent import Agent
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np

class GeneticAlgorithm:
    def __init__(self, fitness_function, population_size, mutation_prob, mutation_strength, elitism_fraction):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.mutation_strength = mutation_strength
        self.elite_size = int(elitism_fraction * population_size)
        self.population = []  # Initialize the population with Agent objects
        self.fitness_function = fitness_function  # Fitness function that evaluates an Agent
        self.fitnesses = np.zeros(population_size)
        self.best_fitness = -np.inf
        self.best_individual = None  # This will store the best Agent object

    def populate(self):
        # Initialize agents and constrain weights
        # brains get created with agent initialization
        for _ in range(self.population_size):
            agent = Agent()
            # Assign/Constrain Weights
            for i in range(agent.brain.net_size):
                for j in range(agent.brain.net_size):
                    # Prevent input-to-output connections
                    if i < agent.brain.input_size and j > agent.brain.net_size-agent.brain.output_size-1:
                        agent.brain.network.weights[i, j] = 0.0001
                    # Prevent output-to-input connections
                    elif i > agent.brain.input_size-1 and j < agent.brain.net_size-agent.brain.output_size:
                        agent.brain.network.weights[i, j] = 0.0001
                    else:
                        agent.brain.network.weights[i, j] = np.random.uniform(-5., 5.)
                        while agent.brain.network.weights[i, j] == 0.0:
                            agent.brain.network.weights[i, j] = np.random.uniform(-5., 5.)
            # Assign Taus
            for i in range(agent.brain.net_size):
                agent.brain.network.taus[i] = 1.0
            # Assign biases
            for i in range(agent.brain.net_size):
                agent.brain.network.biases[i] = np.random.uniform(-2.5, 2.5)

            self.population.append(agent)

    def evaluate_fitness(self):
        # Number of cores/processors to use
        num_cores = 6

        # Create a multiprocessing pool using a context manager
        with Pool(processes=num_cores) as pool:
            # Map fitness_function to each agent in the population
            # and convert the result to a NumPy array
            self.fitnesses = np.array(pool.map(self.fitness_function, self.population))

        # Single processing code for reference
        # for i, agent in enumerate(self.population):
        #     self.fitnesses[i] = self.fitness_function(agent)
        # Update best fitness and Agent
        max_fitness_idx = np.argmax(self.fitnesses)
        if self.fitnesses[max_fitness_idx] > self.best_fitness:
            self.best_fitness = self.fitnesses[max_fitness_idx]
            self.best_individual = deepcopy(self.population[max_fitness_idx])

    def fitness_proportionate_selection(self):
        total_fitness = sum(self.fitnesses)
        if total_fitness == 0:
            probabilities = np.ones(self.population_size) / self.population_size
        else:
            probabilities = self.fitnesses / total_fitness
        
        selected_indices = np.random.choice(self.population_size, size=self.elite_size, p=probabilities, replace=False)
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    def mutate(self, agent):
        brain = agent.brain
        for i in range(brain.network.weights.data.shape[0]):
            if np.random.rand() < self.mutation_prob:
                brain.network.weights.data[i] += np.random.randn() * self.mutation_strength
        
        for i in range(brain.network.biases.shape[0]):
            if np.random.rand() < self.mutation_prob:
                brain.network.biases[i] += np.random.randn() * self.mutation_strength
        
        return agent

    def create_next_generation(self, elites):
        next_generation = deepcopy(elites)
        while len(next_generation) < self.population_size:
            elite_clone = deepcopy(np.random.choice(elites))
            mutated_clone = self.mutate(elite_clone)
            next_generation.append(mutated_clone)
        self.population = next_generation

    def step(self, num_generations):
        # Populate the initial population with constrained NN params
        self.populate()
        for generation in range(num_generations):
            # Reset each agent before evaluation
            for agent in self.population:
                agent.reset()
                
            self.evaluate_fitness()
            elites = self.fitness_proportionate_selection()
            self.create_next_generation(elites)
            print(f"Generation {generation}: Best Fitness = {self.best_fitness}")
        # Return the best Brain found after all generations
        return self.best_individual

