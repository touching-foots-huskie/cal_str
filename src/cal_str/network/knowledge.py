# knowledge storage:
knowledge_dict = dict()
knowledge_dict['environment'] = dict()
knowledge_dict['map_list'] = dict()
knowledge_dict['action_dim'] = dict()
knowledge_dict['action_map'] = dict()

# setting for different environment:
# ball
knowledge_dict['environment']['ball'] = 'Ball-v0'
knowledge_dict['map_list']['ball'] = [[0, 1, 2, 3, 4, 5, 6]]  # ob mapping
knowledge_dict['action_dim']['ball'] = [2, 2]  # ai, ..., all
knowledge_dict['action_map']['ball'] = [[0, 1]]  # action mapping

# ball, ob
knowledge_dict['environment']['ball,ob'] = 'Ball-v1'
knowledge_dict['map_list']['ball,ob'] = [[0, 1, 2, 3, 4, 5, 6], [0, 1, 2, 3, 7, 8, 9]]  # ob mapping
knowledge_dict['action_dim']['ball,ob'] = [2, 2, 2]  # ai, ..., all
knowledge_dict['action_map']['ball,ob'] = [[0, 1], [0, 1]]  # action mapping

# reacher
knowledge_dict['environment']['reacher'] = 'Reacher-v2'
knowledge_dict['map_list']['reacher'] = [[i for i in range(11)]]  # ob mapping
knowledge_dict['action_dim']['reacher'] = [2, 2]  # ai, ..., all
knowledge_dict['action_map']['reacher'] = [[0, 1]]  # action mapping


def network_config(config):
    # index:
    if config['update_name'] is None:
        config['index'] = 0
    else:
        config['index'] = (config['env_type'].split(',')).index(config['update_name'])

    if config['run_type'] == 'train':
        config['a_names'] = config['env_type']
        config['c_names'] = config['env_type']
        config['c_activate'] = True
        config['t_activate'] = False

        # trainable
        config['a_trainable'] = ''
        config['c_trainable'] = ''

        # saveable
        config['a_param_save'] = ''
        config['c_param_save'] = ''

        # restore
        config['a_param_restore'] = ''
        config['c_param_restore'] = ''

        for i in range(len(config['env_type'].split(','))):
            if i != config['index']:
                config['a_trainable'] += 'False,'
                config['c_trainable'] += 'False,'

                config['a_param_save'] += 'False,'
                config['c_param_save'] += 'False,'

                # restore when not train
                config['a_param_restore'] += 'True,'
                config['c_param_restore'] += 'True,'
            else:
                config['a_trainable'] += 'True,'
                config['c_trainable'] += 'True,'

                config['a_param_save'] += 'True,'
                config['c_param_save'] += 'True,'

                if config['continue']:
                    config['a_param_restore'] += 'True,'
                    config['c_param_restore'] += 'True,'
                else:
                    config['a_param_restore'] += 'False,'
                    config['c_param_restore'] += 'False,'

        # trainable
        config['a_trainable'] = config['a_trainable'][:-1]
        config['c_trainable'] = config['c_trainable'][:-1]

        # saveable
        config['a_param_save'] = config['a_param_save'][:-1]
        config['c_param_save'] = config['c_param_save'][:-1]

        # restore
        config['a_param_restore'] = config['a_param_restore'][:-1]
        config['c_param_restore'] = config['c_param_restore'][:-1]
    # environment config:
    config['suppress_ratio'] = 1.0
    # map structure
    config['environment'] = knowledge_dict['environment'][config['env_type']]
    config['map_list'] = knowledge_dict['map_list'][config['env_type']]
    config['action_dim'] = knowledge_dict['action_dim'][config['env_type']]
    config['action_map'] = knowledge_dict['action_map'][config['env_type']]