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
    def __init__(self, n_actions, model, target_model, input_shape=None, gamma=0.99):
        self.gamma = gamma
        self.n_batch = input_shape[0]
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.input_shape[0] = None
        self.model = model
        self.target_model = target_model

    def add_to_graph(self):
        with tf.name_scope('q'):
            self.state_holder = tf.placeholder(dtype=tf.float32, shape=self.input_shape,
                    name='input')
            self.action_holder = tf.placeholder(dtype=tf.int32, shape=[self.n_batch], name='action')
            self.value, self.advantage = self.model.add_to_graph(self.state_holder)

        with tf.name_scope('target_q'):
            self.next_state_holder = \
                    tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
            self.t_value, self.t_advantage = self.target_model.add_to_graph(self.next_state_holder)

        with tf.name_scope('train_q'):
            self.reward_holder = tf.placeholder(dtype=tf.float32, shape=[self.n_batch], name='reward_holder')
            self.is_done_holder = tf.placeholder(dtype=tf.int32, shape=[self.n_batch], name='is_done_holder')
            self.is_done_float = tf.to_float(self.is_done_holder)

            real_q = self.advantage + self.value
            self.q_out = real_q

            real_t_q = self.t_value + self.t_advantage
            self.t_q_out = real_t_q

            self.y = self.reward_holder + self.gamma * tf.reduce_max(real_t_q, axis=1) * \
                    (1 - self.is_done_float)
            self.y_no_update = tf.stop_gradient(self.y)

            self.action_one_hot = tf.one_hot(self.action_holder, self.n_actions)
            self.expected = tf.reduce_sum(real_q * self.action_one_hot, axis=1)
            self.diff = self.y_no_update - self.expected

            self.loss = tf.reduce_sum(self.diff ** 2, name='loss')

        with tf.name_scope('target_update'):
            self.target_model.add_assign_weights('ref1', self.model)

    def run_target_q_update(self, sess):
        self.target_model.run_assign_weights('ref1', sess)

    def q_values(self, state, sess):
        return sess.run(self.q_out, feed_dict={self.state_holder: state})
