#!/usr/bin/python
# __________________________________________________________________________________________________
# Main file that implements Q-Learning with experience replay

import tensorflow as tf
import numpy as np
import gym

# __________________________________________________________________________________________________
# Learning parameters
n_train_iterations      = 1
n_test_iterations       = 100
n_batch                 = 1
update_frequency        = 10
max_episode_length      = 200
learning_rate           = 1e-3
gamma                   = 0.999
eps                     = 0.1
eps_decay               = 0.999
eps_min                 = 0.01

# __________________________________________________________________________________________________
# Environment to play
env = gym.make('Breakout-v0')

sess = tf.Session()

# __________________________________________________________________________________________________
# Experience handling
# TODO: Make this entirely in tensorflow
import random

class Experience:
    def __init___(self, capacity):
        self.experience = set()
        self.capacity = capacity

    def add(self, transition):
        if len(self.experience) >= self.capacity:
            self.experience.remove(random.choice(self.experience))
        self.experience.add(transition)

    def sample(self):
        return random.choice(self.experience)

    def sample_batch(self, num):
        samples = []
        for i in range(num):
            samples.append(self.sample())
        return map(list, zip(*samples))

def preprocess(previous, current):
    np.concatenate([previous, current], axis=(len(previous.shape) - 1))

# __________________________________________________________________________________________________
# Train loop - Sample trajectories - Update Q-Function
try:
    experience = Experience(100)

    for i in range(n_train_iterations):
        observation = env.reset()
        prev_observation = observation
        state = preprocess(prev_observation, observation)

        for j in range(max_episode_length):
            q = sess.run(dqn, feed_dict={'state': state})
            a_t = policy(q, eps)

            prev_observation = observation

            observation, reward, done, _ = env.step(a_t)
            new_state = preprocess(prev_observation, observation)

            experience.add((state, a_t, reward, new_state, done))

            state_batch, action_batch, reward_batch, next_state_batch, is_done_batch = \
                    experience.sample_batch(batch_size)

            # Needs to be improved, suggests the training of this sample batch
            sess.run(learn_step, feed_dict={
                'state_batch': state_batch,
                'action_batch': action_batch,
                'reward_batch': reward_batch,
                'next_state_batch': next_state_batch,
                'is_done_batch': is_done_batch})

            state = new_state

            if i % update_frequency == 0:
                # update target Q function weights
                pass

            if done:
                break
except KeyboardInterrupt:
    pass
    # save parameters

# __________________________________________________________________________________________________
# Test loop
for i in range(n_test_iterations):
    pass

sess.close()
