from interactive_agents.DCOACH import DCOACH
from interactive_agents.HG_DAgger import HG_DAGGER
from interactive_agents.HG_DAgger2 import HG_DAGGER2
from interactive_agents.HG_DAgger3 import HG_DAGGER3
"""
Functions that selects the agent
"""


def agent_selector(agent_type, config_agent, graph_folder_path, graph_load_path):
    if agent_type == 'DCOACH':
        return DCOACH(dim_a=config_agent.getint('dim_a'),
                      action_upper_limits=config_agent['action_upper_limits'],
                      action_lower_limits=config_agent['action_lower_limits'],
                      e=config_agent['e'],
                      buffer_min_size=config_agent.getint('buffer_min_size'),
                      buffer_max_size=config_agent.getint('buffer_max_size'),
                      buffer_sampling_rate=config_agent.getint('buffer_sampling_rate'),
                      buffer_sampling_size=config_agent.getint('buffer_sampling_size'),
                      train_end_episode=config_agent.getboolean('train_end_episode'))

    elif agent_type == 'HG_DAgger':
        return HG_DAGGER(dim_a=config_agent.getint('dim_a'),
                         action_upper_limits=config_agent['action_upper_limits'],
                         action_lower_limits=config_agent['action_lower_limits'],
                         buffer_min_size=config_agent.getint('buffer_min_size'),
                         buffer_max_size=config_agent.getint('buffer_max_size'),
                         buffer_sampling_rate=config_agent.getint('buffer_sampling_rate'),
                         buffer_sampling_size=config_agent.getint('buffer_sampling_size'),
                         number_training_iterations=config_agent.getint('number_training_iterations'),
                         train_end_episode=config_agent.getboolean('train_end_episode'),
                         lstm_hidden_state_size=config_agent.getint('lstm_hidden_state_size'))
    elif agent_type == 'HG_DAgger2':
        return HG_DAGGER2(dim_a=config_agent.getint('dim_a'),
                          action_upper_limits=config_agent['action_upper_limits'],
                          action_lower_limits=config_agent['action_lower_limits'],
                          buffer_min_size=config_agent.getint('buffer_min_size'),
                          buffer_max_size=config_agent.getint('buffer_max_size'),
                          number_training_iterations=config_agent.getint('number_training_iterations'),
                          image_width=config_agent.getint('image_width'),
                          policy_learning_rate=float(config_agent['learning_rate']),
                          load_policy=config_agent.getboolean('load_policy'),
                          network_loc=graph_folder_path,
                          network_load_path=graph_load_path,
                          traffic_light=config_agent.getboolean('traffic_light'),
                          input_direction=config_agent.getboolean('input_direction'))
    elif agent_type == 'HG_DAgger3':
        return HG_DAGGER3(dim_a=config_agent.getint('dim_a'),
                          action_upper_limits=config_agent['action_upper_limits'],
                          action_lower_limits=config_agent['action_lower_limits'],
                          buffer_min_size=config_agent.getint('buffer_min_size'),
                          buffer_max_size=config_agent.getint('buffer_max_size'),
                          number_training_iterations=config_agent.getint('number_training_iterations'),
                          image_width=config_agent.getint('image_width'),
                          policy_learning_rate=float(config_agent['learning_rate']),
                          load_policy=config_agent.getboolean('load_policy'),
                          network_loc=graph_folder_path)
    else:
        raise NameError('Not valid network.')
