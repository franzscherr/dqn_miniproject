#!/usr/bin/python
# __________________________________________________________________________________________________
# Main file that implements Q-Learning with experience replay

import tensorflow as tf
import numpy as np
import gym

from q_learner import QLearner
from fully_connected import FCPart
from model import Model

# __________________________________________________________________________________________________
# Learning parameters
n_train_iterations      = 50
n_test_iterations       = 10
n_batch                 = 16
update_frequency        = 6
max_episode_length      = 200
learning_rate           = 9e-3
learning_rate_decay     = 0.9
learning_rate_min       = 4e-3
gamma                   = 0.99
eps                     = 0.9
eps_decay               = 0.85
eps_min                 = 0.01
print_interval          = 10

# __________________________________________________________________________________________________
# Environment to play
# env = gym.make('Pong-v0')
env = gym.make('CartPole-v0')

n_actions = env.action_space.n
train_observation_shape = list(env.observation_space.shape)

# For first simple testing, input is two adjacent frames
# train_observation_shape[-1] = 2
train_observation_shape.insert(0, n_batch)

observation_shape = train_observation_shape[:]
observation_shape[0] = 1

# __________________________________________________________________________________________________
# Model
file_name = 'pong.npz'

# model_args = ([5,5], [4,4], [4,4], [30, n_actions], train_observation_shape, False, file_name)
# model = ConvolutionalModel(*model_args)
# target_model = ConvolutionalModel(*model_args)

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

model_args = ([20, 20, n_actions + 1], train_observation_shape)
model = SimpleModel(*model_args)
target_model = SimpleModel(*model_args)

q_learner = QLearner(n_actions, model, target_model, train_observation_shape, gamma)
q_learner.add_to_graph()

# train_step = tf.train.AdamOptimizer(learning_rate).minimize(q_learner.loss)
learning_rate_holder = tf.placeholder(dtype=tf.float32)
train_step = tf.train.GradientDescentOptimizer(learning_rate_holder).minimize(q_learner.loss)

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
        self.experience = []
        self.capacity = capacity

    def add(self, transition):
        if len(self.experience) >= self.capacity:
            self.experience.pop(np.random.randint(self.capacity))
        self.experience.append(transition)

    def sample(self):
        return random.choice(self.experience)

    def sample_batch(self, num):
        samples = []
        for i in range(num):
            samples.append(self.sample())
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

def policy(q_values, eps):
    # q_values 1 x n_actions
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    return np.argmax(q_values[:, :-1])

# __________________________________________________________________________________________________
# Train loop - Sample trajectories - Update Q-Function
try:
    experience = Experience(10000)
    loss_list = []
    reward_list = []
    duration_list = []

    for i in range(n_train_iterations):
        observation = env.reset()
        prev_observation = observation
        state = preprocess(prev_observation, observation)
        total_loss = 0
        total_reward = 0
        total_action = 0

        for j in range(max_episode_length):
            if eps > eps_min:
                eps *= eps_decay

            if learning_rate > learning_rate_min:
                learning_rate *= learning_rate_decay

            q_values = q_learner.q_values(np.reshape(state, observation_shape), sess)
            a_t = policy(q_values, eps)
            total_action += a_t

            prev_observation = observation

            observation, reward, done, _ = env.step(a_t)
            total_reward += reward

            new_state = preprocess(prev_observation, observation)

            experience.add(tuple([state, a_t, reward, new_state, done]))

            state_batch, action_batch, reward_batch, next_state_batch, is_done_batch = \
                    experience.sample_batch(n_batch)

            # Needs to be improved, suggests the training of this sample batch
            _, loss = sess.run([train_step, q_learner.loss], feed_dict={
                learning_rate_holder: learning_rate,
                q_learner.state_holder: state_batch,
                q_learner.action_holder: action_batch,
                q_learner.reward_holder: reward_batch,
                q_learner.next_state_holder: next_state_batch,
                q_learner.is_done_holder: is_done_batch})

            total_loss += loss

            state = new_state

            if (i + j) % update_frequency == 0:
                # update target Q function weights
                q_learner.run_target_q_update(sess)

            if done:
                break
        total_loss /= j
        total_action /= j
        reward_list.append(total_reward)
        duration_list.append(j)
        loss_list.append(total_loss)
        if i % print_interval == 0:
            print('loss {:8g} +/- {:4.2f} | reward {:8.2f} +/- {:4.2f} | duration {:5f} +/- {:3f}'
                    .format(np.mean(loss_list), np.sqrt(np.var(loss_list)), np.mean(reward_list),
                        np.sqrt(np.var(reward_list)), np.mean(duration_list), np.sqrt(np.var(duration_list))))
            # print('average loss in trajectory {:5d}: {:10g} | reward: {} | avg action: {}'
                    # .format(i, total_loss, total_reward, total_action))
            loss_list = []
            duration_list = []
            reward_list = []
except KeyboardInterrupt:
    # save parameters
    if file_name and False:
        with open(file_name, 'wb') as f:
            q_learner.save_weights(f, sess)

# __________________________________________________________________________________________________
# Test loop
try:
    for i in range(n_test_iterations):
        observation = env.reset()
        prev_observation = observation
        state = preprocess(prev_observation, observation)

        for j in range(max_episode_length):
            env.render()

            q_values = q_learner.q_values(np.reshape(state, observation_shape), sess)
            a_t = policy(q_values, 0)

            prev_observation = observation

            observation, reward, done, _ = env.step(a_t)
            new_state = preprocess(prev_observation, observation)

            state = new_state

            if done:
                break
except KeyboardInterrupt:
    pass

sess.close()
