#!/usr/bin/python
# __________________________________________________________________________________________________
# Main file that implements Q-Learning with experience replay

import pdb
import tensorflow as tf
import numpy as np
import gym
import sys

from fully_connected import FCPart
from model import Model
#from pong_tools import prepro
from convolutional_model import ConvolutionalModel
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# __________________________________________________________________________________________________
# Learning parameters
trainMNist = True
n_train_iterations      = 1000
n_test_iterations       = 4
n_batch                 = 1
max_episode_length      = 700
learning_rate           = 0.005
learning_rate_decay     = 0.993
learning_rate_min       = 4e-4

# __________________________________________________________________________________________________
# Environment to play
env = gym.make('Pong-v0')

n_actions = env.action_space.n
train_observation_shape = [6]

# For first simple testing, input is two adjacent frames
train_observation_shape.insert(0, n_batch)

observation_shape = train_observation_shape[:]
observation_shape[0] = 1

# __________________________________________________________________________________________________
# Model
if trainMNist:
    file_name = 'pong_prepro.npz'
    #file_name = 'pong_preproMNist.npz'
else:
    file_name = 'pong_prepro.npz'

environmentObservationPlaceHolder = tf.placeholder(dtype=tf.float32, shape=(n_batch, 210, 160, 6))

if trainMNist:
    expectedFeatureOutput = tf.placeholder(dtype=tf.float32, shape=(n_batch, 10))
else:
    expectedFeatureOutput = tf.placeholder(dtype=tf.float32, shape=(n_batch, 6))
    
x = tf.placeholder(tf.float32, shape=[None, 784])
x_image = tf.reshape(x, [-1,28,28,1])
#

if trainMNist:
    model_args = (n_batch, [5, 5, 5], [6, 12, 32], (2, 2), [60, 15, 10], None, False, file_name)
    model = ConvolutionalModel(*model_args)
    q_out = model.add_to_graph(x_image)
else:
    model_args = (n_batch, [5, 5, 5], [6, 12, 32], (2, 2), [60, 15, 6], None, False, file_name)
    model = ConvolutionalModel(*model_args)
    q_out = model.add_to_graph(environmentObservationPlaceHolder)

error = tf.reduce_sum((expectedFeatureOutput - q_out) ** 2)
train = tf.train.AdamOptimizer(learning_rate).minimize(error)
y_ = tf.placeholder(tf.float32, shape=[None, 10])
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(q_out, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(q_out,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.Session()

sess.run(tf.global_variables_initializer())

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

try:
    experience = Experience(5000)
    loss_list = []
    reward_list = []
    duration_list = []

    for i in range(n_train_iterations):
        observation = env.reset()
        prev_observation = observation
        
        for j in range(0, 50):
            observation, _, _, _ = env.step(np.random.randint(n_actions))  
        prev_observation = observation

        state = prepro(observation, prev_observation)
        total_loss = 0
        total_reward = 0
        total_action = 0

        for j in range(max_episode_length):

            prev_observation = observation

            observation, reward, done, _ = env.step(np.random.randint(n_actions))
            total_reward += reward
            
            new_state = prepro(observation, prev_observation)

            experience.add(tuple([observation, prev_observation, new_state]))

            oberservationBatch, prev_observationBatch, new_stateBatch = experience.sample_batch(n_batch)
            
            observationReshape = np.reshape(oberservationBatch, (n_batch, 210, 160, 3))
            prev_observationReshape = np.reshape(prev_observationBatch, (n_batch, 210, 160, 3))
            appendedObservation = np.append(observationReshape, prev_observationReshape, axis=3)

            state = new_state
            
            if trainMNist:
            
                batch = mnist.train.next_batch(50)
                if j%100 == 0:
                    train_accuracy = sess.run(accuracy, feed_dict={
                        x:batch[0], y_: batch[1]})
                    print("step {:d}, training accuracy {:6.4f}".format(j, train_accuracy))
                sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
            else:
                loss = sess.run(error, feed_dict={expectedFeatureOutput:new_stateBatch,
environmentObservationPlaceHolder:appendedObservation})
                
                total_loss += loss

                sess.run(train, feed_dict={expectedFeatureOutput:new_stateBatch,
    environmentObservationPlaceHolder:appendedObservation})
            
            if file_name and (j % 100) == 0:
                with open(file_name, 'wb') as f:
                    model.save_weights(f, sess)

            if done:
                break
   
        if trainMNist:
            test_accuracy = sess.run(accuracy, feed_dict={
                x:mnist.test.images, y_: mnist.test.labels})
            print("test accuracy {:6.4f}".format(test_accuracy))
        else:
            total_loss /= j
            loss_list.append(total_loss)
            print('loss {:8g} +/- {:4.2f}'.format(np.mean(loss_list), np.sqrt(np.var(loss_list))))

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
