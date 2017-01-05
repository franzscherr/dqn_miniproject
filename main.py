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
n_batch                 = 5
learning_rate           = 1e-3
gamma                   = 0.999
eps                     = 0.1
eps_decay               = 0.999
eps_min                 = 0.01

# __________________________________________________________________________________________________
# Environment to play
env = gym.make('Breakout-v0')

# __________________________________________________________________________________________________
# Train loop - Sample trajectories - Update Q-Function
try:
    for i in range(n_train_iterations):
        pass
except KeyboardInterrupt:
    # save parameters

# __________________________________________________________________________________________________
# Test loop
for i in range(n_test_iterations):
    pass
