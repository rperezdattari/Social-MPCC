import time
import pickle
from main_init import exp_num, count_down, max_num_of_episodes, env, render, max_time_steps_episode, transition_model, \
    neural_network


"""
Main loop of the algorithm described in the paper 'Interactive Learning of Temporal Features for Control' 
"""

human = 'rodrigo'
init_time = time.time()

# Print general information
print('\nExperiment number:', exp_num)
print('Environment: Carla')
time.sleep(2)

# Count-down before training if requested
if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)

# Start training loop
pause = True
trajectories = []
episode_trajectory = []
for i_episode in range(max_num_of_episodes):
    print('Starting episode number', i_episode)

    observation = env.reset()  # reset environment at the beginning of the episode
    # Iterate over the episode
    for t in range(int(max_time_steps_episode)):
        if hasattr(env, 'joystick_info'):
            if env.joystick_info.buttons[2]:
                pause = True
            elif env.joystick_info.buttons[1]:
                pause = False
        if pause:
            continue

        # Get feedback signal
        h = env.get_feedback()

        action_agent = 0
        action_human = h
        action = action_human

        # Act
        observation, _, environment_done, _ = env.step(action, action_human, action_agent)

        state_representation = transition_model.get_state_representation(neural_network, observation)

        # Compute done
        done = environment_done

        # Append transition to episode trajectory
        episode_trajectory.append([state_representation, action, transition_model.network_input[-1]])

        # End of episode
        if done:
            if render:
                time.sleep(1)

            # Append episode trajectory to trajectories
            trajectories.append(episode_trajectory)

            # Clear episode trajectory
            episode_trajectory = []

            # Save trajectories
            file_handler = open('trajectories_evaluation/HG_DAgger_%s.obj' % human, 'wb')
            pickle.dump(trajectories, file_handler)

            print('Total time (s):', '%.3f' % (time.time() - init_time))
            break
