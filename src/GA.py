from copy import deepcopy
from Brain import Brain
import numpy as np

class GeneticAlgorithm:
    def __init__(self, fitness_function, population_size, mutation_prob, mutation_strength, elite_size_ratio):
        self.population_size = population_size
        self.mutation_prob = mutation_prob
        self.mutation_strength = mutation_strength
        self.elite_size = int(elite_size_ratio * population_size)
        self.population = [Brain() for _ in range(population_size)]
        self.fitness_function = fitness_function
        self.fitnesses = np.zeros(population_size)
        self.best_fitness = -np.inf
        self.best_individual = None

    def evaluate_fitness(self):
        # Evaluate fitness for each individual in the population
        self.fitnesses = np.array([self.fitness_function(ind) for ind in self.population])
        # Update best fitness and individual
        max_fitness_idx = np.argmax(self.fitnesses)
        if self.fitnesses[max_fitness_idx] > self.best_fitness:
            self.best_fitness = self.fitnesses[max_fitness_idx]
            self.best_individual = deepcopy(self.population[max_fitness_idx])

    def fitness_proportionate_selection(self):
        # Select individuals based on fitness proportionate probabilities
        total_fitness = sum(self.fitnesses)
        if total_fitness == 0:
            # Avoid division by zero if fitnesses are all zero
            probabilities = np.ones(self.population_size) / self.population_size
        else:
            probabilities = self.fitnesses / total_fitness
        
        # Select indices based on the computed probabilities
        selected_indices = np.random.choice(self.population_size, size=self.elite_size, p=probabilities, replace=False)
        selected_individuals = [self.population[i] for i in selected_indices]
        return selected_individuals

    def mutate(self, individual):
        # Mutate weights
        for i in range(individual.network.weights.data.shape[0]):
            if np.random.rand() < self.mutation_prob:
                individual.network.weights.data[i] += np.random.randn() * self.mutation_strength
        
        # Mutate biases
        for i in range(individual.network.size):
            if np.random.rand() < self.mutation_prob:
                individual.network.biases[i] += np.random.randn() * self.mutation_strength
        
        # Optionally enforce constraints on weights and biases
        # ...

        return individual

    def create_next_generation(self, elites):
        next_generation = elites.copy()
        while len(next_generation) < self.population_size:
            # Choose a random elite individual to clone
            elite_clone = np.random.choice(elites)
            # Create a deep copy of the elite individual to avoid modifying the original elite
            elite_clone = deepcopy(elite_clone)
            # Mutate the cloned individual and add it to the next generation
            mutated_clone = self.mutate(elite_clone)
            next_generation.append(mutated_clone)
        # Replace the current population with the new generation
        self.population = next_generation

    def step(self, num_generations):
        for generation in range(num_generations):
            self.evaluate_fitness()
            elites = self.fitness_proportionate_selection()
            self.create_next_generation(elites)
            print(f"Generation {generation}: Best Fitness = {self.best_fitness}")

        return self.best_individual

