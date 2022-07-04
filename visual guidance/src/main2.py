import numpy as np
import time
from main_init2 import  agent, agent_type, exp_num, count_down, \
    max_num_of_episodes, env, render, max_time_steps_episode, human_feedback, save_results, eval_save_path, \
    render_delay, save_policy, env_type


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

    # Reset environment at the beginning of the episode
    observation = env.reset()

    # Restart agent
    agent.restart()

    # Ensure trajectory is not short
    if env_type == 'ROS':
        while env.n_waypoints < 20:
            observation = env.reset()

    # Reset variables at the beginning of the episode
    past_action, past_observation, episode_trajectory, h_counter = None, None, [], 0

    # Go through time steps in the episode
    t = 0
    while True:
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
                if env.joystick_info.buttons[2] and not pause:
                    pause = True
                    print("Paused")
                elif env.joystick_info.buttons[1]:
                    pause = False
                    print("Started")
            if pause:
                continue

            # Get feedback signal
            h = env.get_feedback()
            human_done = False  # TODO: implement
            feedback_received = env.feedback_received

        else:
            h = None

        t += 1

        # Feed h to agent
        agent.feed_h(h, feedback_received)

        if env_type == 'gym':
            observation = [observation, 0, 0]

        # Map action from observation
        if env_type == 'gym':
            action = agent.action(observation)
        elif env_type == 'ROS':
            action, action_human, action_agent = agent.action(observation)

        # Act
        if env_type == 'gym':
            observation, _, environment_done, _ = env.step(action)
        elif env_type == 'ROS':
            observation, _, environment_done, _ = env.step(action, action_human, action_agent)

        # Count feedback
        if feedback_received:  # and (not env.past_feedback_received):
            past_feedback_received = feedback_received
            h_counter += 1

        # Compute done
        done = environment_done or human_done

        # Update weights transition model/policy
        if env_type == 'ROS':
            if t > 100:
                if done:
                    t_total = done  # tell the interactive_agents that the episode finished

                t_train = agent.train(feedback_received=feedback_received,
                                      done=done)
                t_total += 1
        elif env_type == 'gym':
            if done:
                t_total = done  # tell the interactive_agents that the episode finished

            t_train = agent.train(feedback_received=feedback_received,
                                  done=done)
            t_total += 1

        # End of episode
        if done:
            if env_type == 'ROS':
                _, _, _, _ = env.step(np.array([-1]), action_human, action_agent)
            if (t > 100):
                i_episode += 1

                total_feedback.append(h_counter/(t + 1e-6))
                total_time_steps.append(t_total)
                if save_results:
                    np.save(eval_save_path + str(i_episode) + '_feedback', total_feedback)
                    np.save(eval_save_path + str(i_episode) + '_time', total_time_steps)

                if save_policy:
                    agent.neural_network.save_policy()
                    agent.save_buffer()

                if render:
                    time.sleep(1)

            print('Total time (s):', '%.3f' % (time.time() - init_time))
            break