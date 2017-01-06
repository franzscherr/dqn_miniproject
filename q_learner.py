#!/usr/bin/python
# __________________________________________________________________________________________________
# Q-Learner implementation
#

import tensorflow as tf
import numpy as np

from convolution import ConvolutionPart
from fully_connected import FCPart

class Q_Learner:
    def __init__(self, n_actions, input_shape=None):
        conv_size = [5, 5]
        filter_sizes = [10, 10]
        pool_size = [2, 2]
        conv_init_arg = (conv_size, filter_sizes, pool_size)

        layer_sizes = [40, 20, n_actions]
        do_normalization = True
        fc_init_arg = (layer_sizes, do_normalization)

        self.conv_part = ConvolutionPart(*conv_init_arg)
        self.fc_part = FCPart(*fc_init_arg)
        self.n_actions = n_actions
        self.input_shape = input_shape

        # Target Q function
        self.t_conv_part = ConvolutionPart(*conv_init_arg)
        self.t_fc_part = FCPart(*fc_init_arg)

    def add_to_graph(self):
        with tf.name_scope('q'):
            self.state_holder = tf.placeholder(dtype=tf.float32, shape=self.input_shape,
                    name='input')
            self.action_holder = tf.placeholder(dtype=tf.int32, shape=[-1], name='action')
            self.conv_out = self.conv_part.add_to_graph(self.state_holder)
            self.q_out = self.fc_part.add_to_graph(self.conv_out)

        with tf.name_scope('target_q'):
            self.next_state_holder = \
                    tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
            self.t_conv_out = self.t_conv_part.add_to_graph(self.next_state_holder)
            self.t_q_out = self.t_fc_part.add_to_graph(self.t_conv_out)

        with tf.name_scope('train_q'):
            self.reward_holder = tf.placeholder(dtype=tf.float32, shape=[-1], name='reward_holder')
            self.is_done_holder = tf.placeholder(dtype=tf.int32, shape=[-1], name='is_done_holder')

            y = self.reward_holder + self.gamma * tf.reduce_max(self.t_q_out, axis=1) * \
                    (1 - self.is_done_holder)
            y_no_update = tf.stop_gradient(y)

            action_one_hot = tf.one_hot(self.action_holder, self.n_actions)

            self.loss = tf.reduce_mean((y - self.q_out * action_one_hot) ** 2, name='loss')
            # TODO: Let main.py set up the optimization strategy?

        with tf.name_scope('target_update'):
            self.t_conv_part.add_assign_weights('ref1', self.conv_part)
            self.t_fc_part.add_assign_weights('ref1', self.fc_part)

    def run_target_q_update(self, sess):
        self.t_conv_part.run_assign_weights('ref1', sess)
        self.t_fc_part.run_assign_weights('ref1', sess)

    def q_values(self, state, sess):
        return sess.run(q_out, feed_dict={self.state_holder: state})
