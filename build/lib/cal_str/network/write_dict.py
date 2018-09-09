import collections

def write_dict(path, name='caln'):
    if name == 'caln':
        saver_dict = collections.OrderedDict()
        base_list = ['fish', 'ball', 'reacher']
        addition_list = ['ob', None]
        part_list = ['action', 'value', 'activation']
        # combination:
        for base in base_list:
            for addition in addition_list:
                for part in part_list:
                    if addition != None:
                        saver_dict['{}_{}_{}'.format(base, addition, part)] = 0
                    else:
                        saver_dict['{}_{}_{}'.format(base, base, part)] = 0

        for i, name in enumerate(saver_dict.keys()):
            # detailed location
            saver_dict[name] = '{}/train_log/{}'.format(path, str(i))
        return saver_dict
    else:
        pass
