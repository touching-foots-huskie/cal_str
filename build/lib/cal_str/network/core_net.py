# $File: core_net.py
# $Author: Harvey Chang
import numpy as np
import tensorflow as tf
from baselines.common import tf_util as U


def actor_net(obs_ph, act_dim, num_hid_layers=2, hid_size=64):
    with tf.variable_scope('actor'):
        last_out = obs_ph
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i' % (i + 1),
                                                  kernel_initializer=U.normc_initializer(1.0)))

        mean = tf.layers.dense(last_out, act_dim, name='final', kernel_initializer=U.normc_initializer(0.01))
        logstd = tf.get_variable(name="logstd", shape=[1, act_dim], initializer=tf.zeros_initializer())
        return mean, logstd
      
        
def critic_net(obs_ph, num_hid_layers=2, hid_size=64):
    with tf.variable_scope('critic'):
        last_out = obs_ph
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(tf.layers.dense(last_out, hid_size, name='fc%i' % (i + 1),
                                                  kernel_initializer=U.normc_initializer(1.0)))

        vpred = tf.layers.dense(last_out, 1, name='final', kernel_initializer=U.normc_initializer(1.0))[:, 0]
        return vpred


def activate_net(obs_ph):
    with tf.variable_scope('activate'):
        obs_dim = obs_ph.shape.as_list()[-1]  
        hid1_size = obs_dim * 10  
        hid3_size = 10  
        hid2_size = int(np.sqrt(hid1_size * hid3_size))
        out = tf.layers.dense(obs_ph, hid1_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / obs_dim)), name="h1")
        out = tf.layers.dense(out, hid2_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid1_size)), name="h2")
        out = tf.layers.dense(out, hid3_size, tf.tanh,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid2_size)), name="h3")
        out = tf.layers.dense(out, 1,
                              kernel_initializer=tf.random_normal_initializer(
                                  stddev=np.sqrt(1 / hid3_size)), name='output')
        out = tf.squeeze(out)
        return out

