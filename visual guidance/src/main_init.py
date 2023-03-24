# Import libraries and read program args
import argparse
from tools.functions import load_config_data
parser = argparse.ArgumentParser()
parser.add_argument('--config-file', default='Carla_HGDAgger_full2', help='select file in config_files folder')
parser.add_argument('--exp-num', default='-1')
parser.add_argument('--testing', type=bool, default=False, help='test flag')
args = parser.parse_args()

config_file = args.config_file
exp_num = args.exp_num

# Load from config files
if args.testing:
    config = load_config_data('config_files/' + config_file + '_test.ini')
else:
    config = load_config_data('config_files/' + config_file + '.ini')

config_general = config['GENERAL']
config_agent = config['AGENT']
config_feedback = config['FEEDBACK']

if config_general['env_type'] == 'ROS':
    import rospy
    from carla_env import CarlaEnv
elif config_general['env_type'] == 'gym':
    import gym
    from feedback import Feedback

import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # switch to CPU
from buffer import Buffer
from interactive_agents.selector import agent_selector
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

"""
Script that initializes the variables used in the file main.py
"""

agent_type = config_agent['agent']
environment = config_general['environment']
eval_save_path = 'results/' + environment + '_' + agent_type + '/'

render = config_general.getboolean('render')
count_down = config_general.getboolean('count_down')
save_results = config_general.getboolean('save_results')
save_policy = config_agent.getboolean('save_policy')
max_num_of_episodes = config_general.getint('max_num_of_episodes')
num_of_test_episodes = config_general.getint('num_of_test_episodes')
max_time_steps_episode = float(config_general['max_time_steps_episode'])
render_delay = float(config_general['render_delay'])
env_type = config_general['env_type']

# Create Agent
agent = agent_selector(agent_type, config_agent, config_general['graph_folder_path'], config_general['graph_load_path'])

if env_type == 'gym':
    # Create gym environment
    env = gym.make(environment)  # create environment
    observation = env.reset()
    if render:
        env.render()

    # Feedback
    human_feedback = Feedback(env=env,
                              key_type=config_feedback['key_type'],
                              h_up=config_feedback['h_up'],
                              h_down=config_feedback['h_down'],
                              h_right=config_feedback['h_right'],
                              h_left=config_feedback['h_left'],
                              h_null=config_feedback['h_null'])
elif env_type == 'ROS':
    # Create ROS environment
    rospy.init_node('carla_DCOACH')
    env = CarlaEnv()
    rate = rospy.Rate(1)
    rate.sleep()  # apparently we need to run this first for the reset to work

    # Feedback
    human_feedback = None  # it's obtained through the environment

else:
    print('Environment type not known.')
    exit()

# Create saving directory if it does no exist
if save_results:
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)