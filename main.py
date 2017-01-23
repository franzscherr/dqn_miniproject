#!/usr/bin/python
# __________________________________________________________________________________________________
# Main file that implements Q-Learning with experience replay

import pdb
import tensorflow as tf
import numpy as np
import gym
import sys

from q_learner import QLearner
from fully_connected import FCPart
from model import Model
#from pong_tools import prepro
from convolutional_model import ConvolutionalModel
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# __________________________________________________________________________________________________
# Learning parameters
n_train_iterations      = 1000
n_test_iterations       = 4
n_batch                 = 1
update_frequency        = 32
max_episode_length      = 700
learning_rate           = 0.005
learning_rate_decay     = 0.993
learning_rate_min       = 4e-4
gamma                   = 0.98
eps                     = 0.9
eps_decay               = 0.85
eps_min                 = 0.1
print_interval          = 10

keep_prob_begin = 0.7
keep_prob_end = 1.0

temperature_begin = 1000.0
temperature_end = 0.2

# __________________________________________________________________________________________________
# Environment to play
env = gym.make('Pong-v0')

n_actions = env.action_space.n
# train_observation_shape = list(env.observation_space.shape)
train_observation_shape = [6]

# For first simple testing, input is two adjacent frames
# train_observation_shape[-1] = 2
train_observation_shape.insert(0, n_batch)

observation_shape = train_observation_shape[:]
observation_shape[0] = 1

# __________________________________________________________________________________________________
# Model
file_name = 'pong_prepro.npz'
#file_name = 'pong_preproMNist.npz'

# model_args = ([5,5], [4,4], [4,4], [30, n_actions], train_observation_shape, False, file_name)
# model = ConvolutionalModel(*model_args)
# target_model = ConvolutionalModel(*model_args)

class DropoutModel(Model):
    def __init__(self, fc_sizes, keep_holder, input_shape=None):
        fc1_arg = (input_shape[0], fc_sizes[:-1], False, None, False)
        fc2_arg = (input_shape[0], fc_sizes[-1:], False)

        self.n_batch = input_shape[0]
        self.keep_holder = keep_holder

        self.fc1 = FCPart(*fc1_arg)
        self.fc2 = FCPart(*fc2_arg)
        self.input_shape = input_shape
        self.input_shape[0] = None

    def add_to_graph(self, input_tensor):
        interm = self.fc1.add_to_graph(input_tensor)
        interm_dp = tf.nn.dropout(interm, self.keep_holder)
        q_out = self.fc2.add_to_graph(interm_dp)
        return q_out

    def add_assign_weights(self, key, rhs):
        self.fc1.add_assign_weights(key, rhs.fc1)
        self.fc2.add_assign_weights(key, rhs.fc2)

    def run_assign_weights(self, key, sess):
        self.fc1.run_assign_weights(key, sess)
        self.fc2.run_assign_weights(key, sess)

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

model_args = ([30, 60, n_actions + 1], keep_holder, train_observation_shape)
target_model_args = ([30, 60, n_actions + 1], keep_holder, train_observation_shape)

#Use a convolutional model

environmentObservationPlaceHolder = tf.placeholder(dtype=tf.float32, shape=(n_batch, 210, 160, 6))
expectedFeatureOutput = tf.placeholder(dtype=tf.float32, shape=(n_batch, 6))
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1,28,28,1])
#expectedFeatureOutput = tf.placeholder(dtype=tf.float32, shape=(n_batch, 10))
model_args = (n_batch, [5, 5, 5], [6, 12, 32], (2, 2), [60, 15, 6], None, False, file_name)
model = ConvolutionalModel(*model_args)
q_out = model.add_to_graph(environmentObservationPlaceHolder)
#q_out = model.add_to_graph(x_image)

error = tf.reduce_sum((expectedFeatureOutput - q_out) ** 2)
train = tf.train.AdamOptimizer(learning_rate).minimize(error)
#train = expectedFeatureOutput - tf.reshape(q_out, [-1])

keep_prob = tf.placeholder(tf.float32)
y_ = tf.placeholder(tf.float32, shape=[None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(q_out, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(q_out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# model = SimpleModel(*model_args)
# target_model = SimpleModel(*model_args)
#model = DropoutModel(*model_args)
#target_model = DropoutModel(*target_model_args)

#q_learner = QLearner(n_actions, model, target_model, train_observation_shape, gamma)
#q_learner.add_to_graph()

#train_step = tf.train.AdamOptimizer(learning_rate).minimize(q_learner.loss)
#learning_rate_holder = tf.placeholder(dtype=tf.float32)
# train_step = tf.train.GradientDescentOptimizer(learning_rate_holder).minimize(q_learner.loss)

# __________________________________________________________________________________________________
# sess = tf.Session('grpc://10.0.0.6:49354')
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# __________________________________________________________________________________________________
# Experience handling
# TODO: Make this entirely in tensorflow
#import random

class Experience:
    def __init__(self, capacity=1000):
        self.experience = []
        self.capacity = capacity

    def add(self, transition):
        if len(self.experience) >= self.capacity:
            self.experience.pop(np.random.randint(self.capacity))
        self.experience.append(transition)

    def sample(self):
        index = np.random.randint(len(self.experience))
        return self.experience[index]
        #return np.random.choice(self.experience)

    def sample_batch(self, num):
        samples = []
        for i in range(num):
            samples.append(self.sample())
        return map(list, zip(*samples))

#def preprocess(previous, current):
    # a = np.sum(previous, axis=2, keepdims=True)
    # b = np.sum(current, axis=2, keepdims=True)
    # a = (a - np.mean(a))
    # b = (b - np.mean(b))
    # a = a / np.max(a)
    # b = b / np.max(b)
    # return np.concatenate([a, b], axis=(len(previous.shape) - 1))
#    return current

#def policy(q_values, strategy='epsgreedy', **kwargs):
#    # q_values 1 x n_actions
#    if strategy == 'epsgreedy':
#        eps = kwargs.get('eps', 0.1)
#        if np.random.rand() < eps:
#            return np.random.randint(n_actions)
#        return np.argmax(q_values[:, :-1])
#    elif strategy == 'boltzmann':
#        temperature = kwargs.get('temperature', 1.0)
#        e = np.exp(q_values[0, :-1] / temperature)
#        dist = e / np.sum(e)
#        return np.random.choice(n_actions, p=dist)
#    else:
#        return np.random.randint(n_actions)

#

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
# Train loop - Sample trajectories - Update Q-Function
try:
    experience = Experience(5000)
    loss_list = []
    reward_list = []
    duration_list = []

    #keep_prob = keep_prob_begin
    #temperature = temperature_begin

    for i in range(n_train_iterations):
        observation = env.reset()
        prev_observation = observation
        
        for j in range(0, 50):
            observation, _, _, _ = env.step(np.random.randint(n_actions))  
        prev_observation = observation

        # state = preprocess(prev_observation, observation)
        state = prepro(observation, prev_observation)
        total_loss = 0
        total_reward = 0
        total_action = 0

        for j in range(max_episode_length):

	    #Use the preprocessing step as a supervised output for the cnn
	

            #if eps > eps_min:
            #    eps *= eps_decay

            #if learning_rate > learning_rate_min:
            #    learning_rate *= learning_rate_decay

            #q_values = q_learner.q_values(np.reshape(state, observation_shape), sess)
            # a_t = policy(q_values, eps)
            #a_t = policy(q_values, strategy='boltzmann', temperature=temperature)
            #total_action += a_t

            prev_observation = observation

            observation, reward, done, _ = env.step(np.random.randint(n_actions))
            total_reward += reward

            # new_state = preprocess(prev_observation, observation)
            #print(observation)
            #print(prev_observation)
            new_state = prepro(observation, prev_observation)

            experience.add(tuple([observation, prev_observation, new_state]))

            oberservationBatch, prev_observationBatch, new_stateBatch = experience.sample_batch(n_batch)

            # Needs to be improved, suggests the training of this sample batch
            #_, loss = sess.run([train_step, q_learner.loss], feed_dict={
            #    keep_holder: keep_prob,
            #    learning_rate_holder: learning_rate,
            #    q_learner.state_holder: state_batch,
            #    q_learner.action_holder: action_batch,
            #    q_learner.reward_holder: reward_batch,
            #    q_learner.next_state_holder: next_state_batch,
            #    q_learner.is_done_holder: is_done_batch})
            
            #print(new_state)
            #print([observation, prev_observation].shape)
            observationReshape = np.reshape(oberservationBatch, (n_batch, 210, 160, 3))
            prev_observationReshape = np.reshape(prev_observationBatch, (n_batch, 210, 160, 3))
            appendedObservation = np.append(observationReshape, prev_observationReshape, axis=3)
            #print(appendedObservation.shape)
            #print(np.reshape(sess.run(q_out, feed_dict={environmentObservationPlaceHolder:appendedObservation}), -1))

            #for z in range(0, 10):
#            networkOutput = np.reshape(sess.run(q_out, #feed_dict={environmentObservationPlaceHolder:appendedObservation}), -1)
            #print(networkOutput)
            #print(new_stateBatch)

            
            loss = sess.run(error, feed_dict={expectedFeatureOutput:new_stateBatch,
environmentObservationPlaceHolder:appendedObservation})
            total_loss += loss

            sess.run(train, feed_dict={expectedFeatureOutput:new_stateBatch,
environmentObservationPlaceHolder:appendedObservation})

            #print(total_loss)
            state = new_state
            
#            batch = mnist.train.next_batch(50)
#            if j%100 == 0:
#                train_accuracy = sess.run(accuracy, feed_dict={
#                    x:batch[0], y_: batch[1], keep_prob: 1.0})
#                print("step {:d}, training accuracy {:6.4f}".format(j, train_accuracy))
                
#            test_accuracy = sess.run(accuracy, feed_dict={
#                x:mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0})
#            print("test accuracy {:6.4f}".format(test_accuracy))
#            input()
 #           sess.run(train_step, feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
                

            #print('loss {:8g}'.format(loss))
            
#            if file_name and (j % 100) == 0:
#                with open(file_name, 'wb') as f:
#                    model.save_weights(f, sess)

            #if (i + j) % update_frequency == 0:
                # update target Q function weights
            #    q_learner.run_target_q_update(sess)

            if done:
                break
        #if keep_prob < keep_prob_end:
        #    keep_prob += (keep_prob_end - keep_prob_begin) / n_train_iterations
        #if temperature > temperature_end:
        #    temperature -= (temperature_begin - temperature_end) / n_train_iterations
        #    if temperature < temperature_end:
        #        temperature = temperature_end

        total_loss /= j
        #total_action /= j
        #reward_list.append(total_reward)
        duration_list.append(j)
        loss_list.append(total_loss)
   
        print('loss {:8g} +/- {:4.2f}'.format(np.mean(loss_list), np.sqrt(np.var(loss_list))))
            # print('average loss in trajectory {:5d}: {:10g} | reward: {} | avg action: {}'
                    # .format(i, total_loss, total_reward, total_action))
        loss_list = []
        duration_list = []
        reward_list = []
        if file_name:
            with open(file_name, 'wb') as f:
                model.save_weights(f, sess)

except KeyboardInterrupt:
    pass
    # save parameters
    if file_name:
        with open(file_name, 'wb') as f:
            model.save_weights(f, sess)


aaaaaa
