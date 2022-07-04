# Import libraries and read program args
import argparse
from tools.functions import load_config_data
parser = argparse.ArgumentParser()
parser.add_argument('--config-file', default='Carla_HGDAgger_full', help='select file in config_files folder')
parser.add_argument('--exp-num', default='-1')
args = parser.parse_args()

config_file = args.config_file
exp_num = args.exp_num

# Load from config files
config = load_config_data('config_files/' + config_file + '.ini')
config_general = config['GENERAL']
config_transition_model = config['TRANSITION_MODEL']
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
from transition_model import TransitionModel
from neural_network import NeuralNetwork
from tensorflow.compat.v1 import ConfigProto, InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

"""
Script that initializes the variables used in the file main.py
"""

agent_type = config_agent['agent']
environment = config_general['environment']
transition_model_type = config_transition_model['transition_model']
eval_save_path = 'results/' + environment + '_' + agent_type + '_' + transition_model_type + '/'

render = config_general.getboolean('render')
count_down = config_general.getboolean('count_down')
save_results = config_general.getboolean('save_results')
save_policy = config_agent.getboolean('save_policy')
save_transition_model = config_transition_model.getboolean('save_transition_model')
max_num_of_episodes = config_general.getint('max_num_of_episodes')
max_time_steps_episode = float(config_general['max_time_steps_episode'])
render_delay = float(config_general['render_delay'])
env_type = config_general['env_type']

# Create Neural Network
neural_network = NeuralNetwork(policy_learning_rate=float(config_agent['learning_rate']),
                               transition_model_learning_rate=float(config_transition_model['learning_rate']),
                               lstm_hidden_state_size=config_transition_model.getint('lstm_hidden_state_size'),
                               load_policy=config_agent.getboolean('load_policy'),
                               dim_a=config_agent.getint('dim_a'),
                               network_loc=config_general['graph_folder_path'],
                               image_size=config_transition_model.getint('image_side_length'))
# Create Transition Model
transition_model = TransitionModel(training_sequence_length=config_transition_model.getint('training_sequence_length'),
                                   lstm_hidden_state_size=config_transition_model.getint('lstm_hidden_state_size'),
                                   crop_observation=config_transition_model.getboolean('crop_observation'),
                                   image_width=config_transition_model.getint('image_side_length'),
                                   show_transition_model_output=config_transition_model.getboolean('show_transition_model_output'),
                                   show_observation=config_transition_model.getboolean('show_observation'),
                                   resize_observation=config_transition_model.getboolean('resize_observation'),
                                   occlude_observation=config_transition_model.getboolean('occlude_observation'),
                                   dim_a=config_agent.getint('dim_a'),
                                   buffer_sampling_rate=config_transition_model.getint('buffer_sampling_rate'),
                                   buffer_sampling_size=config_transition_model.getint('buffer_sampling_size'),
                                   number_training_iterations=config_transition_model.getint('number_training_iterations'),
                                   train_end_episode=config_transition_model.getboolean('train_end_episode'))

# Create Agent
agent = agent_selector(agent_type, config_agent)

# Create Transition Model buffer
transition_model_buffer = Buffer(min_size=config_transition_model.getint('buffer_min_size'),
                                 max_size=config_transition_model.getint('buffer_max_size'))

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
    env.reset()

    # Feedback
    human_feedback = None  # it's obtained through the environment

else:
    print('Environment type not known.')
    exit()

# Create saving directory if it does no exist
if save_results:
    if not os.path.exists(eval_save_path):
        os.makedirs(eval_save_path)