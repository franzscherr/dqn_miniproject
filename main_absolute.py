#!/usr/bin/python
# __________________________________________________________________________________________________
# Main file that implements Q-Learning with experience replay

import pdb
import tensorflow as tf
import numpy as np
import gym

from q_learner import QLearner
from fully_connected_tanh import FCPart
from model import Model
# from pong_tools import prepro

# __________________________________________________________________________________________________
# Learning parameters
params = {}
params['n_train_iterations']      = 5000
params['n_test_iterations']       = 4
params['n_batch']                 = 32
params['update_frequency']        = 54
params['max_episode_length']      = 5000
params['learning_rate']           = 0.00008
params['learning_rate_decay']     = 1
params['learning_rate_min']       = 0.002
params['gamma']                   = 0.985
params['eps']                     = 0.4
params['eps_max']                 = 0.9994
params['eps_min']                 = 0.15
params['print_interval']          = 5

params['keep_prob_begin'] = 0.9
params['keep_prob_end'] = 1.0
params['keep_prob'] = params['keep_prob_begin']

params['temperature_begin'] = 0.6
params['temperature_end'] = 5.0
params['temperature'] = params['temperature_begin']

# __________________________________________________________________________________________________
# Load parameters from configuration (if existent, or create one)
import sys
import pickle as pl
import os
session_path = None

# __________________________________________________________________________________________________
# EnvLog helps to log the training process
class EnvLog:
    def __init__(self):
        # per training sample
        self.loss = []
        self.learning_rate = []
        self.temperature = []

        # per trajectory
        self.duration = []
        self.reward = []

    def add_after_sample(self, loss, learning_rate, temperature):
        self.loss.append(loss)
        self.learning_rate.append(learning_rate)
        self.temperature.append(temperature)

    def add_after_trajectory(self, reward, duration):
        self.reward.append(reward)
        self.duration.append(duration)

    def save(self, f):
        pl.dump((self.duration, self.reward, self.loss, self.learning_rate, self.temperature), f)

    def load(self, f):
        self.duration, self.reward, self.loss, self.learning_rate, self.temperature = pl.load(f)

elog = EnvLog()

if len(sys.argv) > 1:
    session_path = sys.argv[1]
    if not os.path.exists(session_path):
        os.mkdir(session_path)
    if os.path.exists(session_path + '/config.pkl'):
        with open(session_path + '/config.pkl', 'rb') as f:
            params = pl.load(f)
        with open(session_path + '/log.pkl', 'rb') as f:
            elog.load(f)
    else:
        with open(session_path + '/config.pkl', 'wb') as f:
            pl.dump(params, f)

# __________________________________________________________________________________________________
# Environment to play
env = gym.make('Pong-v0')

n_actions = env.action_space.n
# train_observation_shape = list(env.observation_space.shape)
train_observation_shape = [6]

# For first simple testing, input is two adjacent frames
# train_observation_shape[-1] = 2
train_observation_shape.insert(0, params['n_batch'])

observation_shape = train_observation_shape[:]
observation_shape[0] = 1

# __________________________________________________________________________________________________
# Model
file_name = session_path + '/model.npz'

# model_args = ([5,5], [4,4], [4,4], [30, n_actions], train_observation_shape, False, file_name)
# model = ConvolutionalModel(*model_args)
# target_model = ConvolutionalModel(*model_args)

class DropoutModel(Model):
    def __init__(self, fc_sizes, keep_holder, input_shape=None, load_from=None):
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
        f = file_name
        if type(file_name) == str:
            f = open(file_name, 'wb')
        d1 = self.fc1.saveable_weights_dict(f, sess)
        d2 = self.fc_advantage.saveable_weights_dict(f, sess)
        d3 = self.fc_value.saveable_weights_dict(f, sess)
        np.savez_compressed(f, **{**d1, **d2, **d3})

class SimpleModel(Model):
    def __init__(self, fc_sizes, input_shape=None):
        fc_init_arg = (input_shape[0], fc_sizes, False)

        self.n_batch = input_shape[0]

        self.fc_part = FCPart(*fc_init_arg)
        self.input_shape = input_shape
        self.input_shape[0] = None

    def add_to_graph(self, input_tensor):
        q_out = self.fc_part.add_to_graph(input_tensor)
        return q_out

    def add_assign_weights(self, key, rhs):
        self.fc_part.add_assign_weights(key, rhs.fc_part)

    def run_assign_weights(self, key, sess):
        self.fc_part.run_assign_weights(key, sess)

keep_holder = tf.placeholder_with_default(1.0, shape=None)

model_args = ([30, n_actions + 1], keep_holder, train_observation_shape, file_name)
target_model_args = ([30, n_actions + 1], keep_holder, train_observation_shape, file_name)

# model = SimpleModel(*model_args)
# target_model = SimpleModel(*model_args)
model = DropoutModel(*model_args)
target_model = DropoutModel(*target_model_args)

q_learner = QLearner(n_actions, model, target_model, train_observation_shape, params['gamma'])
q_learner.add_to_graph()

learning_rate_holder = tf.placeholder(dtype=tf.float32)
train_step = tf.train.AdamOptimizer(learning_rate_holder).minimize(q_learner.loss)
# train_step = tf.train.GradientDescentOptimizer(learning_rate_holder).minimize(q_learner.loss)

# __________________________________________________________________________________________________
# sess = tf.Session('grpc://10.0.0.6:49354')
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# __________________________________________________________________________________________________
# Experience handling
# TODO: Make this entirely in tensorflow
import random

class Experience:
    def __init__(self, capacity=100):
        self.experience_fi = []
        self.experience_in = []
        self.capacity = capacity

    def add(self, transition):
        if np.random.rand() < 0.5:#transition[2] != 0:
            if len(self.experience_fi) >= self.capacity:
                self.experience_fi.pop(np.random.randint(self.capacity))
            self.experience_fi.append(transition)
        else:
            if len(self.experience_in) >= self.capacity:
                self.experience_in.pop(np.random.randint(self.capacity))
            self.experience_in.append(transition)
        # if len(self.experience) >= self.capacity:
            # # if not transition[4] and np.random.rand() > 0.003:
                # # return
            # self.experience.pop(np.random.randint(self.capacity))
        # self.experience.append(transition)

    def sample(self):
        t = None
        while t == None:
            if np.random.rand() < 0.5 and len(self.experience_fi) > 0:
                t = random.choice(self.experience_fi)
            else:
                t = random.choice(self.experience_in)
        return t

    def sample_batch(self, num):
        samples = []
        for i in range(num):
            t = self.sample()
            # for j in range(50):
                # if t[4]:
                    # break
            samples.append(t)
        return map(list, zip(*samples))

def preprocess(previous, current):
    # a = np.sum(previous, axis=2, keepdims=True)
    # b = np.sum(current, axis=2, keepdims=True)
    # a = (a - np.mean(a))
    # b = (b - np.mean(b))
    # a = a / np.max(a)
    # b = b / np.max(b)
    # return np.concatenate([a, b], axis=(len(previous.shape) - 1))
    return current

q_holder = tf.placeholder(dtype=tf.float32)
q_dist = tf.nn.softmax(q_holder)

def policy(q_values, strategy='epsgreedy', **kwargs):
    observations = kwargs.get('observation', None)
    if observations != None and np.random.rand() < 0.0:
        if observations[2 + 3] > 0:
            return 1
        else:
            return -1
    # q_values 1 x n_actions
    if strategy == 'epsgreedy':
        eps = kwargs.get('eps', 0.1)
        if np.random.rand() < eps:
            return np.random.randint(n_actions)
        return np.argmax(q_values[:, :])
    elif strategy == 'boltzmann':
        temperature = kwargs.get('temperature', 1.0)
        # e = np.exp((q_values[0, :]) / temperature)
        # dist = e / np.sum(e)
        dist = sess.run(q_dist, feed_dict={q_holder: q_values / temperature})
        return np.random.choice(n_actions, p=dist[0])
    else:
        return np.random.randint(n_actions)


def extract(observation):
    crop = np.mean(observation, axis=2)[34:194, :]
    ball = np.unravel_index(np.argmax(np.logical_and(crop > 235, crop < 237)), crop.shape)
    self = np.unravel_index(np.argmax(np.logical_and(crop > 123, crop < 124)), crop.shape)
    opp = np.unravel_index(np.argmax(np.logical_and(crop > 138, crop < 140)), crop.shape)
    return ball, self, opp

def extract_rel(observation):
    ball, self, opp = extract(observation)
    return ball[1] - self[1], ball[0] - self[0], opp[0] - self[0]

def extract_abs(observation):
    ball, self, opp = extract(observation)
    return ball[1], ball[0], opp[0]

def prepro(observation, prev_observation):
    bx, by, oy = extract_abs(observation)
    pbx, pby, poy = extract_abs(prev_observation)
    return np.array([bx, by, oy, pbx, pby, poy])


if True:
    # __________________________________________________________________________________________________
    # Train loop - Sample trajectories - Update Q-Function
    try:
        experience = Experience(1300)
        loss_list = []
        reward_list = []
        duration_list = []
        sch = 40
        render = False

        for i in range(params['n_train_iterations']):
            observation = env.reset()
            prev_observation = observation
            # state = preprocess(prev_observation, observation)
            state = prepro(observation, prev_observation)
            total_loss = 0
            total_reward = 0
            total_action = 0

            for j in range(params['max_episode_length']):
                if params['eps'] > params['eps_min']:
                    params['eps'] -= (params['eps_max'] - params['eps_min']) / params['n_train_iterations']

                if params['learning_rate'] > params['learning_rate_min']:
                    params['learning_rate'] *= params['learning_rate_decay']

                q_values = q_learner.q_values(np.reshape(state, observation_shape), sess)
                a_t = policy(q_values, strategy='epsgreedy', eps=params['eps'], observation=state)
                # a_t = policy(q_values, strategy='boltzmann', temperature=params['temperature'])
                total_action += a_t

                prev_observation = observation
                if render:
                    env.render()

                observation, reward, done, _ = env.step(a_t)
                total_reward += reward

                # new_state = preprocess(prev_observation, observation)
                new_state = prepro(observation, prev_observation)

                experience.add(tuple([state, a_t, reward, new_state, done]))

                if (i + j) > 32:
                    state_batch, action_batch, reward_batch, next_state_batch, is_done_batch = \
                            experience.sample_batch(params['n_batch'])

                    # Needs to be improved, suggests the training of this sample batch
                    _, loss, action_one_hot, y, q, q_target, diff, expected, value, advantage = sess.run([
                        train_step, q_learner.loss, q_learner.action_one_hot, q_learner.y, 
                        q_learner.q_out, q_learner.t_q_out, q_learner.diff, q_learner.expected,
                        q_learner.value, q_learner.advantage], feed_dict={
                        keep_holder: params['keep_prob'],
                        learning_rate_holder: params['learning_rate'],
                        q_learner.state_holder: state_batch,
                        q_learner.action_holder: action_batch,
                        q_learner.reward_holder: reward_batch,
                        q_learner.next_state_holder: next_state_batch,
                        q_learner.is_done_holder: is_done_batch})
                    total_loss += loss
                    elog.add_after_sample(loss, params['learning_rate'], params['temperature'])

                if render and done:
                    render = False
                    pdb.set_trace()
                if done and i > sch:
                    pdb.set_trace()

                state = new_state

                if (i + j) % params['update_frequency'] == 0:
                    # update target Q function weights
                    q_learner.run_target_q_update(sess)

                if done:
                    break
            if params['keep_prob'] < params['keep_prob_end']:
                params['keep_prob'] += (params['keep_prob_end'] - params['keep_prob_begin']) / params['n_train_iterations']
            # if params['temperature'] > params['temperature_end']: 
                # params['temperature'] -= (params['temperature_begin'] - params['temperature_end']) / params['n_train_iterations']
                # if params['temperature'] < params['temperature_end']:
                    # params['temperature'] = params['temperature_end']
            elog.add_after_trajectory(total_reward, j + 1)

            total_loss /= j
            total_action /= j
            reward_list.append(total_reward)
            duration_list.append(j)
            loss_list.append(total_loss)
            if i % params['print_interval'] == 0:
                print('loss {:8g} +/- {:4.2f} | reward {:8.2f} +/- {:4.2f} | duration {:5f} +/- {:3f}'
                        .format(np.mean(loss_list), np.sqrt(np.var(loss_list)), np.mean(reward_list),
                            np.sqrt(np.var(reward_list)), np.mean(duration_list), np.sqrt(np.var(duration_list))))
                # print('average loss in trajectory {:5d}: {:10g} | reward: {} | avg action: {}'
                        # .format(i, total_loss, total_reward, total_action))
                loss_list = []
                duration_list = []
                reward_list = []
                if file_name:
                    with open(file_name, 'wb') as f:
                        model.save_weights(f, sess)
                with open(session_path + '/config.pkl', 'wb') as f:
                    pl.dump(params, f)
                with open(session_path + '/log.pkl', 'wb') as f:
                    elog.save(f)
    except KeyboardInterrupt:
        # save parameters
        if file_name:
            with open(file_name, 'wb') as f:
                model.save_weights(f, sess)
        with open(session_path + '/config.pkl', 'wb') as f:
            pl.dump(params, f)
        with open(session_path + '/log.pkl', 'wb') as f:
            elog.save(f)

# __________________________________________________________________________________________________
# Test loop
try:
    for i in range(params['n_test_iterations']):
        observation = env.reset()
        prev_observation = observation
        # state = preprocess(prev_observation, observation)
        state = prepro(observation, prev_observation)

        for j in range(params['max_episode_length']):
            env.render()

            q_values = q_learner.q_values(np.reshape(state, observation_shape), sess)
            a_t = policy(q_values, strategy='epsgreedy', eps=0, observation=state)

            prev_observation = observation

            observation, reward, done, _ = env.step(a_t)
            # new_state = preprocess(prev_observation, observation)
            new_state = prepro(observation, prev_observation)

            state = new_state

            if done:
                break
except KeyboardInterrupt:
    pass

sess.close()