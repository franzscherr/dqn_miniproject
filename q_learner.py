#!/usr/bin/python
# __________________________________________________________________________________________________
# Q-Learner implementation
#

import tensorflow as tf
import numpy as np

from convolution import ConvolutionPart
from fully_connected import FCPart

# __________________________________________________________________________________________________
class QLearner:
    def __init__(self, n_actions, input_shape=None, gamma=0.99, load_from=None):
        conv_size = [5, 5]
        filter_sizes = [10, 10]
        pool_size = [2, 2]
        conv_init_arg = (input_shape[0], conv_size, filter_sizes, pool_size, load_from)

        layer_sizes = [80, 20, n_actions]
        do_normalization = False
        fc_init_arg = (input_shape[0], layer_sizes, do_normalization, load_from)

        self.gamma = gamma
        self.n_batch = input_shape[0]

        self.conv_part = ConvolutionPart(*conv_init_arg)
        self.fc_part = FCPart(*fc_init_arg)
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.input_shape[0] = None

        # Target Q function
        self.t_conv_part = ConvolutionPart(*conv_init_arg)
        self.t_fc_part = FCPart(*fc_init_arg)

    def add_to_graph(self):
        with tf.name_scope('q'):
            self.state_holder = tf.placeholder(dtype=tf.float32, shape=self.input_shape,
                    name='input')
            self.action_holder = tf.placeholder(dtype=tf.int32, shape=[self.n_batch], name='action')
            self.conv_out = self.conv_part.add_to_graph(self.state_holder)
            self.q_out = self.fc_part.add_to_graph(self.conv_out)

        with tf.name_scope('target_q'):
            self.next_state_holder = \
                    tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
            self.t_conv_out = self.t_conv_part.add_to_graph(self.next_state_holder)
            self.t_q_out = self.t_fc_part.add_to_graph(self.t_conv_out)

        with tf.name_scope('train_q'):
            self.reward_holder = tf.placeholder(dtype=tf.float32, shape=[self.n_batch], name='reward_holder')
            self.is_done_holder = tf.placeholder(dtype=tf.int32, shape=[self.n_batch], name='is_done_holder')
            is_done_float = tf.to_float(self.is_done_holder)

            y = self.reward_holder + self.gamma * tf.reduce_max(self.t_q_out, axis=1) * \
                    (1 - is_done_float)
            y_no_update = tf.stop_gradient(y)

            action_one_hot = tf.one_hot(self.action_holder, self.n_actions)

            self.loss = tf.reduce_mean((y - tf.reduce_sum(self.q_out * action_one_hot, axis=1)) \
                    ** 2, name='loss')
            # TODO: Let main.py set up the optimization strategy?

        with tf.name_scope('target_update'):
            self.t_conv_part.add_assign_weights('ref1', self.conv_part)
            self.t_fc_part.add_assign_weights('ref1', self.fc_part)

    def run_target_q_update(self, sess):
        self.t_conv_part.run_assign_weights('ref1', sess)
        self.t_fc_part.run_assign_weights('ref1', sess)

    def q_values(self, state, sess):
        return sess.run(self.q_out, feed_dict={self.state_holder: state})

    def save_weights(self, file_name, sess):
        f = file_name
        if type(file_name) == str:
            f = open(file_name, 'wb')
        self.conv_part.save_weights(f, sess)
        self.fc_part.save_weights(f, sess)
