#!/usr/bin/python
# __________________________________________________________________________________________________
# Main file that implements Q-Learning with experience replay

import tensorflow as tf
import numpy as np
import gym

from q_learner import QLearner

# __________________________________________________________________________________________________
# Learning parameters
n_train_iterations      = 200
n_test_iterations       = 10
n_batch                 = 8
update_frequency        = 100
max_episode_length      = 1000
learning_rate           = 3e-3
gamma                   = 0.999
eps                     = 0.2
eps_decay               = 0.9999
eps_min                 = 0.08
print_interval          = 2

# __________________________________________________________________________________________________
# Environment to play
env = gym.make('Breakout-v0')

n_actions = env.action_space.n
train_observation_shape = list(env.observation_space.shape)

# For first simple testing, input is two adjacent frames
train_observation_shape[-1] *= 2
train_observation_shape.insert(0, n_batch)

observation_shape = train_observation_shape[:]
observation_shape[0] = 1

# __________________________________________________________________________________________________
# Model
file_name = 'saved_weights.npz'

q_learner = QLearner(n_actions, train_observation_shape, gamma, file_name)
q_learner.add_to_graph()

train_step = tf.train.AdamOptimizer(learning_rate).minimize(q_learner.loss)

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
    return np.concatenate([previous, current], axis=(len(previous.shape) - 1))

def policy(q_values, eps):
    # q_values 1 x n_actions
    if np.random.rand() < eps:
        return np.random.randint(n_actions)
    return np.argmax(q_values)

# __________________________________________________________________________________________________
# Train loop - Sample trajectories - Update Q-Function
try:
    experience = Experience(100)

    for i in range(n_train_iterations):
        observation = env.reset()
        prev_observation = observation
        state = preprocess(prev_observation, observation)
        total_loss = 0
        total_reward = 0

        for j in range(max_episode_length):
            if eps > eps_min:
                eps *= eps_decay

            q_values = q_learner.q_values(np.reshape(state, observation_shape), sess)
            a_t = policy(q_values, eps)

            prev_observation = observation

            observation, reward, done, _ = env.step(a_t)
            total_reward += reward

            new_state = preprocess(prev_observation, observation)

            experience.add(tuple([state, a_t, reward, new_state, done]))

            state_batch, action_batch, reward_batch, next_state_batch, is_done_batch = \
                    experience.sample_batch(n_batch)

            # Needs to be improved, suggests the training of this sample batch
            _, loss = sess.run([train_step, q_learner.loss], feed_dict={
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
        if i % print_interval == 0:
            print('average loss in trajectory {:5d}: {:10g} | reward: {}'
                    .format(i, total_loss / print_interval, total_reward))
except KeyboardInterrupt:
    # save parameters
    if file_name:
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
