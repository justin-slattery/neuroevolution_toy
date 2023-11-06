'''
Main class for creating the Genetic Algorithm and its required components.
'''

from copy import deepcopy
from Brain import Brain
from Agent import Agent
import numpy as np

from copy import deepcopy
from Agent import Agent
import numpy as np

class GeneticAlgorithm:
    def __init__(self, fitness_function, population_size, mutation_prob, mutation_strength, elitism_fraction):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.mutation_strength = mutation_strength
        self.elite_size = int(elitism_fraction * population_size)
        self.population = [Agent() for _ in range(population_size)]  # Initialize the population with Agent objects
        self.fitness_function = fitness_function  # Fitness function that evaluates an Agent
        self.fitnesses = np.zeros(population_size)
        self.best_fitness = -np.inf
        self.best_individual = None  # This will store the best Agent object

    def evaluate_fitness(self):
        for i, agent in enumerate(self.population):
            self.fitnesses[i] = self.fitness_function(agent)
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

