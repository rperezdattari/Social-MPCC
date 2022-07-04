import numpy as np
import time
import tensorflow as tf
from main_init import neural_network, transition_model, transition_model_type, agent, agent_type, exp_num, count_down, \
    max_num_of_episodes, env, render, max_time_steps_episode, human_feedback, save_results, eval_save_path, \
    render_delay, save_policy, save_transition_model, env_type


"""
Main loop of the algorithm described in the paper 'Interactive Learning of Temporal Features for Control' 
"""

# Initialize variables
total_feedback, total_time_steps, trajectories_database = [], [], []
t_total, h_counter, last_t_counter, omg_c, eval_counter = 1, 0, 0, 0, 0
human_done, evaluation, random_agent, evaluation_started = False, False, False, False

init_time = time.time()

# Print general information
print('\nExperiment number:', exp_num)
print('Environment: Carla')
print('Learning algorithm: ', agent_type)
print('Transition Model:', transition_model_type, '\n')

time.sleep(2)

# Count-down before training if requested
if count_down:
    for i in range(10):
        print(' ' + str(10 - i) + '...')
        time.sleep(1)

# Start training loop
pause = True
i_episode = 0
#summary_writer = tf.summary.create_file_writer(logdir='logs/
summary_writer = None
t_train = 0
for _ in range(max_num_of_episodes):
    print('Starting episode number', i_episode)

    if not evaluation:
        transition_model.new_episode()
        agent.new_episode()

    # Reset environment at the beginning of the episode
    observation = env.reset()

    # Ensure trajectory is not short
    if env_type == 'ROS':
        while env.n_waypoints < 20:
            observation = env.reset()

    # Reset variables at the beginning of the episode
    past_action, past_observation, episode_trajectory, h_counter = None, None, [], 0

    # Go through time steps in the episode
    for t in range(int(max_time_steps_episode)):

        if env_type == 'gym':
            # Make the environment visible and delay
            if render:
                env.render()
                time.sleep(render_delay)

            # Get feedback signal
            h, feedback_received = human_feedback.get_feedback()

            # Ask human for done
            human_done = human_feedback.ask_for_done()

        elif env_type == 'ROS':
            if hasattr(env, 'joystick_info'):
                if env.joystick_info.buttons[2]:
                    pause = True
                elif env.joystick_info.buttons[1]:
                    pause = False
            if pause:
                continue

            # Get feedback signal
            h = env.get_feedback()
            human_done = False  # TODO: implement
            feedback_received = env.feedback_received

        else:
            h = None

        # Feed h to agent
        agent.feed_h(h, feedback_received)

        # Map action from observation
        state_representation = transition_model.get_state_representation(neural_network, observation)
        action, action_human, action_agent = agent.action(neural_network, state_representation)

        # Act
        if env_type == 'gym':
            observation, _, environment_done, _ = env.step(action)
        elif env_type == 'ROS':
            observation, _, environment_done, _ = env.step(action, action_human, action_agent)

        # Add last action to transition model
        transition_model.add_action(action)

        # Append transition to database
        if not evaluation:
            if past_action is not None and past_observation is not None:
                episode_trajectory.append([past_observation, past_action, transition_model.processed_observation])  # append o, a, o' (not really necessary to store it like this)

            past_observation, past_action = transition_model.processed_observation, action

            if t % 100 == 0 or environment_done:
                trajectories_database.append(episode_trajectory)  # append episode trajectory to database
                episode_trajectory = []

        if feedback_received:  # and (not env.past_feedback_received):
            past_feedback_received = feedback_received
            h_counter += 1

        # Compute done
        done = environment_done or human_done

        # Add last step to memory buffer
        if transition_model.last_step(action) is not None and np.any(h):  # if human teleoperates, add action to database
            agent.buffer.add(transition_model.last_step(action))

        # Update weights transition model/policy
        if not evaluation and (t > 100):
            if done:
                t_total = done  # tell the interactive_agents that the episode finished

            t_train = agent.train(neural_network, transition_model, action, t_total, done, summary_writer, t_train, feedback_received, transition_model.last_step(h))
            t_total += 1

        # End of episode
        if done:  # and (t > 100):
            i_episode += 1
            print('Percentage of given feedback:', '%.3f' % ((h_counter / (t + 1e-6)) * 100))
            total_feedback.append(h_counter/(t + 1e-6))
            total_time_steps.append(t_total)
            if save_results:
                np.save(eval_save_path + str(i_episode) + '_feedback', total_feedback)
                np.save(eval_save_path + str(i_episode) + '_time', total_time_steps)

            if save_policy:
                neural_network.save_policy(id=str(i_episode + 1))

            if render:
                time.sleep(1)

            print('Total time (s):', '%.3f' % (time.time() - init_time))
            break
        #elif done and (t < 100):
        #    t_total -= t
        #    agent.buffer.buffer = agent.buffer.buffer[:-h_counter]
        #    break
