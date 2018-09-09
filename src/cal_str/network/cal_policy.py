# cal_policy is the cascade policy structure of gym version
import os
import tensorflow as tf
import baselines.common.tf_util as U
import cal_str.network.core_net as br

from cal_str.network.write_dict import write_dict
from baselines.common.distributions import make_pdtype
from baselines.common.mpi_running_mean_std import RunningMeanStd


class CalPolicy(object):

    recurrent = False

    def __init__(self, name, *arg, **kwargs):
        with tf.variable_scope(name):
            self.scope = tf.get_variable_scope().name
            self._init(*arg, **kwargs)

    def _init(self, config, ob_space, ac_space):
        self.config = config
        self.update_name = config['update_name']
        self.base_name = config['a_names'].split(',')[0]

        current_path = os.getcwd()
        self.save_path_dict = write_dict(current_path)
        self.index = config['index']

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        # get ob:
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))
        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        # normalize:
        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        # deliver ob to each part
        self.obs = []
        for i, map_list in enumerate(config['map_list']):
            _ob = []
            for _i in map_list:
                _ob.append(obz[:, _i:(_i+1)])
            self.obs.append(tf.concat(_ob, axis=-1, name='ob_{}'.format(i)))

        # ac network build:
        self.means = []

        self.vpred = 0
        self.vpreds = dict()
        self.param_dict = dict()
        self.saver_dict = dict()

        self.build_c_network()
        # get final output
        pdparam = self.build_a_network()
        self.pd = self.pdtype.pdfromflat(pdparam)
        self.state_in = []
        self.state_out = []

        stochastic = tf.placeholder(dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def build_a_network(self):
        # anet setup:
        _mean = 0
        for i, name in enumerate(self.config['a_names'].split(',')):
            # different attrib has different dim of action
            _action_dim = self.config['action_dim'][i]
            with tf.variable_scope(name) as scope:
                # if reuse attribute
                if name in self.config['a_names'].split(',')[:i]:
                    scope.reuse_variables()
                if i == 0:
                    mean, logstd = br.actor_net(self.obs[0], _action_dim)
                    _mean = mean  # mean of first attrib

                else:
                    cs = tf.concat([self.obs[i], _mean], axis=-1)  # concat with the first attrib
                    mean, logstd = br.actor_net(cs, _action_dim)

                # refresh std
                if i == self.index:
                    self.logstd = logstd

                # refresh mean
                self.means.append(mean)
                # param cache
                self.param_dict['{}_action'.format(name)] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                              '{}/{}/actor'.format(self.scope, name))
                self.saver_dict['{}_action'.format(name)] = tf.train.Saver(self.param_dict['{}_action'.format(name)])

        # concat| complex construction
        _means = [tf.constant(0.0) for i in range(self.config['action_dim'][-1])]
        for _i, a_mean in enumerate(self.means):
            for _index in self.config['action_map'][_i]:
                _means[_index] += a_mean[:, _index:_index+1]

        # final output
        final_mean = tf.concat(_means, axis=-1)
        # construct whole_mean
        pdparam = tf.concat([final_mean, final_mean * 0.0 + self.logstd], axis=1)
        return pdparam

    def build_c_network(self):
        if self.config['c_activate']:
            for i, name in enumerate(self.config['c_names'].split(',')):
                with tf.variable_scope(name) as scope:
                    if name in self.config['c_names'].split(',')[:i]:
                        scope.reuse_variables()
                    if i == 0:
                        vpred = br.critic_net(self.obs[0])
                    else:
                        # we don't need to concern with mean, which is too detail to converge
                        vpred = br.critic_net(self.obs[i])

                    self.vpreds[name] = vpred
                    # value is already added
                    self.vpred = vpred + self.vpred

                self.param_dict['{}_value'.format(name)] = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                                              '{}/{}/critic'.format(self.scope, name))
                self.saver_dict['{}_value'.format(name)] = tf.train.Saver(self.param_dict['{}_value'.format(name)])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        # find train part:
        if self.config['a_trainable'] and self.config['c_trainable']:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}/{}'.format(self.scope, self.update_name))
        elif self.config['a_trainable'] and not(self.config['c_trainable']):
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}/{}/actor'.format(self.scope, self.update_name))
        elif not(self.config['a_trainable']) and self.config['c_trainable']:
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, '{}/{}/critic'.format(self.scope, self.update_name))
        else:
            return None

    def get_initial_state(self):
        return []

    def save(self):
        # get param_save:
        self.config['param_save'] = {}
        # a:
        for i, value in enumerate(self.config['a_param_save'].split(',')):
            self.config['param_save']['{}_action'.format(self.config['a_names'].split(',')[i])] = \
                True if value == 'True' else False
        # c
        if self.config['c_activate']:
            for i, value in enumerate(self.config['c_param_save'].split(',')):
                self.config['param_save']['{}_value'.format(self.config['c_names'].split(',')[i])] \
                    = True if value == 'True' else False

        # t
        if self.config['t_activate']:
            for i, value in enumerate(self.config['t_param_save'].split(',')):
                self.config['param_save']['{}_activation'.format(self.config['t_names'].split(',')[i]
                                                                             )] = True if value == 'True' else False
        # save
        for name in self.param_dict.keys():
            if self.config['param_save'][name]:
                self.saver_dict[name].save(tf.get_default_session(),
                                           self.save_path_dict['{}_{}'.format(self.base_name, name)])
                print('{} saved'.format(name))

    def restore(self):
        self.config['param_restore'] = {}
        # a:
        for i, value in enumerate(self.config['a_param_restore'].split(',')):
            self.config['param_restore'][
                '{}_action'.format(self.config['a_names'].split(',')[i])] = True if value == 'True' else False
        # c
        if self.config['c_activate']:
            for i, value in enumerate(self.config['c_param_restore'].split(',')):
                self.config['param_restore'][
                    '{}_value'.format(self.config['c_names'].split(',')[i])] = True if value == 'True' else False

        # t
        if self.config['t_activate']:
            for i, value in enumerate(self.config['t_param_restore'].split(',')):
                self.config['param_restore']['{}_activation'.format(
                    self.config['t_names'].split(',')[i])] = True if value == 'True' else False

        for name in self.param_dict.keys():
            if self.config['param_restore'][name]:
                self.saver_dict[name].restore(tf.get_default_session(),
                                              self.save_path_dict['{}_{}'.format(self.base_name, name)])
                print('{} restored'.format(name))
