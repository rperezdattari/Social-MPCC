from interactive_agents.HG_DAgger import HG_DAGGER

"""
Functions that selects the agent
"""


def agent_selector(agent_type, config_agent, graph_folder_path, graph_load_path):
    if agent_type == 'HG_DAgger':
        return HG_DAGGER(dim_a=config_agent.getint('dim_a'),
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
    else:
        raise NameError('Not valid network.')
