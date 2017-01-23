#!/usr/bin/python
# __________________________________________________________________________________________________
# Testing only
# First parameter is directory to load the model

import tensorflow as tf
import numpy as np
import gym

from q_learner import QLearner
from fully_connected import FCPart
from model import Model

# __________________________________________________________________________________________________
# Learning parameters
params = {}
# Used parameters
params['n_train_iterations']      = 5000
params['n_test_iterations']       = 4
params['n_batch']                 = 32
params['update_frequency']        = 54
params['max_episode_length']      = 5000
params['learning_rate']           = 0.0001
params['gamma']                   = 0.985
params['eps']                     = 0.4
params['eps_max']                 = 0.9994
params['eps_min']                 = 0.15
params['print_interval']          = 5

# We use a small amount of dropout to have a better generalization
params['keep_prob_begin'] = 0.9
params['keep_prob_end'] = 1.0
params['keep_prob'] = params['keep_prob_begin']
# Unused parameters
params['learning_rate_decay']     = 1
params['learning_rate_min']       = 0.002

params['temperature_begin'] = 0.6
params['temperature_end'] = 5.0
params['temperature'] = params['temperature_begin']

# __________________________________________________________________________________________________
# Load parameters from configuration (if existent, or create one)
import sys
import pickle as pl
import os
session_path = None

session_path = './trained'
assert(os.path.exists(session_path))
if os.path.exists(session_path + '/config.pkl'):
    with open(session_path + '/config.pkl', 'rb') as f:
        params = pl.load(f)

# __________________________________________________________________________________________________
# Environment to play
env = gym.make('Pong-v0')

n_actions = env.action_space.n
train_observation_shape = [6]
train_observation_shape.insert(0, params['n_batch'])

observation_shape = train_observation_shape[:]
observation_shape[0] = 1

# __________________________________________________________________________________________________
# Model
file_name = session_path + '/model.npz'

class DropoutModel(Model):
    def __init__(self, fc_sizes, keep_holder, input_shape=None, load_from=None):
        # Define the model to have a separate branch of advantage and value
        fc1_arg = (input_shape[0], fc_sizes[:-1], False, load_from, False, 'fc1')
        fc_advantage_arg = (input_shape[0], [30, n_actions], False, load_from, True, 'fc_advantage')
        fc_value_arg = (input_shape[0], [30, 1], False, load_from, True, 'fc_value')

        self.n_batch = input_shape[0]
        self.keep_holder = keep_holder

        self.fc1 = FCPart(*fc1_arg)
        self.fc_advantage = FCPart(*fc_advantage_arg)
        self.fc_value = FCPart(*fc_value_arg)
        self.input_shape = input_shape
        self.input_shape[0] = None

    def add_to_graph(self, input_tensor):
        interm = self.fc1.add_to_graph(input_tensor)
        interm_dp = tf.nn.dropout(interm, self.keep_holder)
        advantage = self.fc_advantage.add_to_graph(interm_dp)
        advantage = advantage - tf.reduce_mean(advantage, axis=1, keep_dims=True)
        value = self.fc_value.add_to_graph(interm_dp)
        return value, advantage

    def add_assign_weights(self, key, rhs):
        self.fc1.add_assign_weights(key, rhs.fc1)
        self.fc_advantage.add_assign_weights(key, rhs.fc_advantage)
        self.fc_value.add_assign_weights(key, rhs.fc_value)

    def run_assign_weights(self, key, sess):
        self.fc1.run_assign_weights(key, sess)
        self.fc_advantage.run_assign_weights(key, sess)
        self.fc_value.run_assign_weights(key, sess)

    def save_weights(self, file_name, sess):
        pass

keep_holder = tf.placeholder_with_default(1.0, shape=None)

model_args = ([30, n_actions + 1], keep_holder, train_observation_shape, file_name)
target_model_args = ([30, n_actions + 1], keep_holder, train_observation_shape, file_name)

# Double Q-learning requires two models (target and action)
model = DropoutModel(*model_args)
target_model = DropoutModel(*target_model_args)

q_learner = QLearner(n_actions, model, target_model, train_observation_shape, params['gamma'])
q_learner.add_to_graph()

# Actual training operation
learning_rate_holder = tf.placeholder(dtype=tf.float32)
train_step = tf.train.AdamOptimizer(learning_rate_holder).minimize(q_learner.loss)

# Softmax
q_holder = tf.placeholder(dtype=tf.float32)
q_dist = tf.nn.softmax(q_holder)

# __________________________________________________________________________________________________
# Policy
def policy(q_values, strategy='epsgreedy', **kwargs):
    # q_values 1 x n_actions
    # Epsilon greedy
    if strategy == 'epsgreedy':
        eps = kwargs.get('eps', 0.1)
        if np.random.rand() < eps:
            return np.random.randint(n_actions)
        return np.argmax(q_values[:, :])
    # Boltzmann action selection
    elif strategy == 'boltzmann':
        temperature = kwargs.get('temperature', 1.0)
        dist = sess.run(q_dist, feed_dict={q_holder: q_values / temperature})
        return np.random.choice(n_actions, p=dist[0])
    # Complete random
    else:
        return np.random.randint(n_actions)

# __________________________________________________________________________________________________
# Experience handling
import random

class Experience:
    def __init__(self, capacity=100):
        self.experience = []
        self.capacity = capacity

    def add(self, transition):
        if len(self.experience) >= self.capacity:
            self.experience.pop(np.random.randint(self.capacity))
        self.experience.append(transition)

    def sample(self):
        t = random.choice(self.experience)
        return t

    def sample_batch(self, num):
        samples = []
        for i in range(num):
            t = self.sample()
            samples.append(t)
        return map(list, zip(*samples))

# __________________________________________________________________________________________________
# Preprocessing improved version
def extract(observation):
    crop = np.mean(observation, axis=2)[34:194, :]
    ball = np.unravel_index(np.argmax(np.logical_and(crop > 235, crop < 237)), crop.shape)
    self = np.unravel_index(np.argmax(np.logical_and(crop > 123, crop < 124)), crop.shape)
    opp = np.unravel_index(np.argmax(np.logical_and(crop > 138, crop < 140)), crop.shape)
    return ball, self, opp

def extract_rel(observation):
    ball, self, opp = extract(observation)
    return ball[1] - self[1], ball[0] - self[0], opp[0] - self[0]

def prepro(observation, prev_observation):
    bx, by, oy = extract_rel(observation)
    pbx, pby, poy = extract_rel(prev_observation)
    return np.array([bx, by, oy, pbx, pby, poy])

# __________________________________________________________________________________________________
# Create Session
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# __________________________________________________________________________________________________
# Test loop
try:
    while True:
        observation = env.reset()
        prev_observation = observation
        state = prepro(observation, prev_observation)

        for j in range(params['max_episode_length']):
            env.render()

            q_values = q_learner.q_values(np.reshape(state, observation_shape), sess)
            a_t = policy(q_values, strategy='epsgreedy', eps=0, observation=state)

            prev_observation = observation

            observation, reward, done, _ = env.step(a_t)
            new_state = prepro(observation, prev_observation)

            state = new_state

            if done:
                break
except KeyboardInterrupt:
    pass

sess.close()
