import pickle
import numpy as np
from tools.functions import FastImagePlot
from main_init import neural_network, agent

human = 'rodrigo'
n_episodes = 3
show_trajectory = False
file_handler = open('trajectories_evaluation/HG_DAgger_%s.obj' % human, 'rb')
trajectories = pickle.load(file_handler)

if show_trajectory:
    # Plot observations
    observations_plot = FastImagePlot(1, np.zeros([64, 64, 1]), 64, 'Image State', vmin=0, vmax=255, filters=1)

squared_error_list = []
MSE_list = []
agent.h = 0  # Set h and feedback received to random value to avoid errors
agent.feedback_received = False

for k in range(n_episodes):
    neural_network._load_policy(id=str(k))
    # Iterate through every episode
    for i in range(len(trajectories)):
        # Iterate through the trajectory
        agent.new_episode()
        for j in range(len(trajectories[i])):
            step = trajectories[i][j]  # get human [observation, action]
            state_representation = step[0]
            action_human = step[1]
            observation = step[2]

            # Get action agent
            _, _, action_agent = agent.action(neural_network, state_representation)

            # Squared error
            squared_error = (action_agent - action_human) ** 2
            squared_error_list.append(squared_error)

            # Show trajectory
            if show_trajectory:
                observations_plot.refresh(observation[0, :, :, 1].numpy())
            #print('Episode:', i, ': Step:', j)

    # Compute mean squared error
    MSE = np.array(squared_error_list).mean()
    MSE_list.append(MSE)
print('MSE:', MSE_list)
