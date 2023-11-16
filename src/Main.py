'''
Main class for running the genetic algorithm, the simulation, and recording/plotting the results.
'''
from audioop import mul
from GA import GeneticAlgorithm
from Agent import Agent
from Target import Target
from scipy.sparse import csr_matrix
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import functools
import time

import configparser
config = configparser.ConfigParser()
config.read("./config.ini")

THRESHOLD               =       float(config['DEFAULT']['THRESHOLD'])
ENV_SIZE                =       float(config['DEFAULT']['ENV_SIZE'])
STEP_SIZE               =       float(config['DEFAULT']['STEP_SIZE'])
MUTATION_PROBABILITY    =       float(config['DEFAULT']['MUTATION_PROBABILITY'])
MUTATION_STRENGTH       =       float(config['DEFAULT']['MUTATION_STRENGTH'])
ELITISM_FRACTION        =       float(config['DEFAULT']['ELITISM_FRACTION'])
NUM_CORES               =       int(config['DEFAULT']['NUM_CORES'])
RUN_DURATION            =       int(config['DEFAULT']['RUN_DURATION'])
GENERATIONS             =       int(config['DEFAULT']['GENERATIONS'])
POPULATION_SIZE         =       int(config['DEFAULT']['POPULATION_SIZE'])
DATA_DIR                =       str(config['DEFAULT']['DATA_DIR'])
MULTI_PROCESSING        =       config['DEFAULT'].getboolean('MULTI_PROCESSING')

def format_time(seconds):
    """Converts a time in seconds to a string format: hours, minutes, seconds."""
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"

# # Run a basic version of the simulation without the GA
# # Initialize your agent and object here

# agent = Agent(brain)
# target = Target()

# # Run the simulation
# num_steps = 10  # Just an example, set this to whatever makes sense for your simulation
# some_threshold = 1.0  # The distance considered close enough to "reach" the object

# for step in range(num_steps):
#     agent.update(target_position=target.position)
#     # Check for condition where agent reaches the object
#     if np.linalg.norm(agent.position - target.position) < some_threshold:
#         print("Reached the object!")
#         break
#     elif (step == (num_steps-1)) and (not np.linalg.norm(agent.position - target.position) < some_threshold):
#         print(f"Distance to object: {np.linalg.norm(agent.position - target.position)}")

# # The agent's trajectory is recorded in the agent.trajectory attribute
# # Now let's plot it using the trajectory data
# trajectory = np.array(agent.trajectory)  # Convert to numpy array for easy slicing

# Run the GA-driven simulation
def fitness_function(agent, target):
    # Run the simulation for a fixed number of steps
    for _ in range(RUN_DURATION):
        agent.update(target_position=target.position)
        if agent.has_reached_target:
            break  # End the simulation early if the target is reached
        
    # Fitness is inversely related to the distance to the target at the last step
    fitness = 1.0 / (np.linalg.norm(agent.position - target.position) + 1e-5)
    
    return fitness

def plot_trajectory(agent_trajectory, target_position):
    trajectory = np.array(agent_trajectory)  # Convert to numpy array for easy slicing

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot environment boundaries
    x_bounds, y_bounds = [-ENV_SIZE, ENV_SIZE], [-ENV_SIZE, ENV_SIZE]
    ax.plot([x_bounds[0], x_bounds[1], x_bounds[1], x_bounds[0], x_bounds[0]],
            [y_bounds[0], y_bounds[0], y_bounds[1], y_bounds[1], y_bounds[0]],
            [0, 0, 0, 0, 0], 'gray', linestyle='--')

    # Plot the trajectory as a line
    ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], label='Agent Path')
    ax.scatter(trajectory[0, 0], trajectory[0, 1], trajectory[0, 2], color='k', s=50, label='Start')
    ax.scatter(*target_position, color='r', s=50, label='Target')

    if len(trajectory) > 1:
        direction = trajectory[-1] - trajectory[-2]
        direction = direction / np.linalg.norm(direction)
        ax.quiver(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2],
                  direction[0], direction[1], direction[2],
                  length=0.25, normalize=True, color='black', arrow_length_ratio=0.05)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

def plot_nn_io(best_agent):
    input_history = np.array(best_agent.brain.input_history)
    output_history = np.array(best_agent.brain.output_history)

    # print("Input history shape:", np.array(best_agent.brain.input_history).shape)
    # print("Output history shape:", np.array(best_agent.brain.output_history).shape)
    
    fig, axs = plt.subplots(2, 1, figsize=(10, 6))
    fig.suptitle('Best Agent Neural Network Inputs and Outputs Over Time')

    axs[0].set_title('Inputs')
    for i in range(best_agent.brain.input_size):
        axs[0].plot(input_history[:, i], label=f'Input {i+1}')
    axs[0].set_ylabel('Input Value')
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1))

    axs[1].set_title('Outputs')
    for i in range(best_agent.brain.output_size):
        axs[1].plot(output_history[:, i], label=f'Output {i+1}')
    axs[1].set_xlabel('Time Step')
    axs[1].set_ylabel('Output Value')
    axs[1].legend(loc='upper left', bbox_to_anchor=(1, 1))

    plt.tight_layout()
    plt.subplots_adjust(right=0.8)

def main():
    if MULTI_PROCESSING:
        print("Utiliizing multi-processing. Compute resources maximized. \n")
    else:
        print("Not utilizing multi-processing. Evaluation will be slower. \n")
    
    # Initialize the target
    target = Target()

    # Initialize the genetic algorithm
    ga = GeneticAlgorithm(
        fitness_function=functools.partial(fitness_function, target=target),
        population_size=POPULATION_SIZE,
        mutation_prob=MUTATION_PROBABILITY,
        mutation_strength=MUTATION_STRENGTH,
        elitism_fraction=ELITISM_FRACTION,
        multi_processing=MULTI_PROCESSING,
        num_cores=NUM_CORES
    )

    start_time = time.time()

    # Run the genetic algorithm
    best_agent = ga.step(num_generations=GENERATIONS)  # Assumes ga.step returns the best brain

    # Run the simulation for the best agent for visualization
    for run in range(RUN_DURATION):
        best_agent.update(target_position=target.position)
        if best_agent.has_reached_target:
            print("Reached the target!")
            break  # End the simulation early if the target is reached
        elif run == (RUN_DURATION-1) and (not np.linalg.norm(best_agent.position - target.position) <= THRESHOLD):
            print(f"Distance to target: {np.linalg.norm(best_agent.position - target.position)}")

    end_time = time.time()
    elapsed_time = end_time - start_time

    # Format the elapsed time into hours, minutes, and seconds
    formatted_time = format_time(elapsed_time)
    print(f"Simulation and GA took {formatted_time} to complete.")

    if best_agent.has_reached_target:
        #print("Input History:", best_agent.brain.input_history)
        # Plotting the trajectory
        plot_trajectory(best_agent.trajectory, target.position)

        # Plotting the NN Inputs and Outputs over Time
        plot_nn_io(best_agent)
        plt.show()

    else:
        print("Agent did not reach the target. No plots generated.")

if __name__ == '__main__':
    main()