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
            self.q_out = self.model.add_to_graph(self.state_holder)

        with tf.name_scope('target_q'):
            self.next_state_holder = \
                    tf.placeholder(dtype=tf.float32, shape=self.input_shape, name='input')
            self.t_q_out = self.target_model.add_to_graph(self.next_state_holder)

        with tf.name_scope('train_q'):
            self.reward_holder = tf.placeholder(dtype=tf.float32, shape=[self.n_batch], name='reward_holder')
            self.is_done_holder = tf.placeholder(dtype=tf.int32, shape=[self.n_batch], name='is_done_holder')
            is_done_float = tf.to_float(self.is_done_holder)
            
            value = self.q_out[:, -1]
            value = tf.reshape(value, [-1, 1])
            advantage = self.q_out[:, :-1]
            advantage = advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)

            real_q = advantage + value

            t_value = self.t_q_out[:, -1]
            t_value = tf.reshape(t_value, [-1, 1])
            t_advantage = self.t_q_out[:, :-1]
            t_advantage = t_advantage - tf.reduce_mean(t_advantage, axis=1, keep_dims=True)

            real_t_q = t_value + t_advantage

            y = self.reward_holder + self.gamma * tf.reduce_max(real_t_q, axis=1) * \
                    (1 - is_done_float)
            y_no_update = tf.stop_gradient(y)

            action_one_hot = tf.one_hot(self.action_holder, self.n_actions)

            self.loss = tf.reduce_mean((y - tf.reduce_sum(real_q * action_one_hot, axis=1)) \
                    ** 2, name='loss')
            # TODO: Let main.py set up the optimization strategy?

        with tf.name_scope('target_update'):
            self.target_model.add_assign_weights('ref1', self.model)

    def run_target_q_update(self, sess):
        self.target_model.run_assign_weights('ref1', sess)

    def q_values(self, state, sess):
        return sess.run(self.q_out, feed_dict={self.state_holder: state})
