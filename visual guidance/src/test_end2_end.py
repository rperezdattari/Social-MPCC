import os.path
import rospy
import numpy as np
import time
import argparse
from main_init2_end2end import agent, agent_type, exp_num, count_down, \
    num_of_test_episodes, env, render, max_time_steps_episode, human_feedback, save_results, eval_save_path, \
    render_delay, save_policy, env_type, config_general


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

n_collisions, n_timeouts = 0, 0
imitation_error, infeasible_steps = [], []
collision_times, stuck_times = [], []
total_n_steps = 0

now = rospy.Time.now()
t0 = dict = {'seconds': now.secs,
             'n_seconds': now.nsecs}
i_episode = 0
times = []
while i_episode < num_of_test_episodes:
    print('Starting episode number', i_episode)

    # Reset environment at the beginning of the episode
    observation = env.reset()

    # Ensure trajectory is not short
    if env_type == 'ROS':
        while env.n_waypoints < 20:
            observation = env.reset()

    # Reset variables at the beginning of the episode
    past_action, past_observation, episode_trajectory, h_counter = None, None, [], 0

    # Go through time steps in the episode
    t = 0
    while True:
        if hasattr(env, 'joystick_info'):
            if env.joystick_info.buttons[2] and not pause:
                pause = True
                print("Paused")
            elif env.joystick_info.buttons[1]:
                pause = False
                print("Started")
        if pause and False:
            continue

        # Get feedback signal
        if not config_general.getboolean('mpc'):
            h = env.get_feedback()
            human_done = False  # TODO: implement
            feedback_received = env.feedback_received
        else:
            h = 0.0
            feedback_received = True

        t += 1
        # Feed h to agent
        agent.feed_h(h, feedback_received)

        if env_type == 'gym':
            observation = [observation, 0, 0]

        # Map action from observation
        if env_type == 'gym':
            action = agent.action(observation)
        elif env_type == 'ROS':
            # time_init = time.time()
            action, action_human, action_agent = agent.action(observation)
            # time_end = time.time()
            # delta_time = time_end - time_init
            # if t > 200:
            #     times.append(delta_time)
            #     print('time:', np.mean(times))
            #     print('std:', np.sqrt(np.var(times)))

        # Act
        if env_type == 'gym':
            observation, _, environment_done, _ = env.step(action)
        elif env_type == 'ROS':
            observation, _, environment_done, _ = env.step(action, action_human, action_agent)

        # Compute done
        done = environment_done or human_done

        imitation_error.append(np.linalg.norm(action_human - action_agent))

        # End of episode
        if done:
            if env_type == 'ROS':
                _, _, _, _ = env.step(np.array([-1, 0, 0]), action_human, action_agent)
            i_episode += 1
            total_feedback.append(h_counter/(t + 1e-6))
            total_time_steps.append(t_total)

            save_path = config_general['graph_load_path']+"/results"
            if not os.path.isdir(save_path):
                os.mkdir(save_path)

            if t > 100:
                i_episode +=1

            if env.collision and (t > 100):
                t = 0
                n_collisions += env.collision
                now = rospy.Time.now()
                dict = {'seconds': now.secs,
                        'n_seconds': now.nsecs}
                collision_times.append(dict)

            if env.velocity_stuck:
                now = rospy.Time.now()
                n_timeouts += env.velocity_stuck
                dict = {'seconds': now.secs,
                        'n_seconds': now.nsecs}
                stuck_times.append(dict)

            print("****************** Statistics ********************")
            print("N collisions: {}".format(n_collisions))
            print("N timeouts: {}".format(n_timeouts))
            print("Imitation error: {}+-{}".format(np.mean(imitation_error), np.std(imitation_error)))
            print("Percentage of infeasible steps: {}+-{}".format(np.mean(infeasible_steps), np.std(infeasible_steps)))
            print("Total distance: {}".format(env.total_distance))
            if total_n_steps > 0:
                infeasible_steps.append(env.total_infeasible_steps/total_n_steps*100)

            if render:
                time.sleep(1)

            print('Total time (s):', '%.3f' % (time.time() - init_time))
            break

        total_n_steps += 1

print("****************** Statistics ********************")
print("N collisions: {}".format(n_collisions))
print("N timeouts: {}".format(n_timeouts))
print("Imitation error: {}+-{}".format(np.mean(imitation_error),np.std(imitation_error)))
print("Percentage of infeasible steps: {}+-{}".format(np.mean(infeasible_steps),np.std(infeasible_steps)))
print("Total distance: {}".format(env.total_distance))

for coll_time in collision_times:
    print("Collision happened at {}s".format(coll_time['seconds']-t0['seconds']))
for stuck_time in stuck_times:
    print("Got stuck at {}s".format(stuck_time['seconds']-t0['seconds']))

rospy.signal_shutdown("Done Testing")
