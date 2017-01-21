#!/usr/bin/python
# __________________________________________________________________________________________________
# Fully connected network
#

import tensorflow as tf
import numpy as np

from serializable_weights import SerializableWeights

# __________________________________________________________________________________________________
# FCPart is meant to represent the later fully connected layers in DQN
#
class FCPart(SerializableWeights):
    def normalize(self, x):
        mean = tf.reduce_mean(x)
        t = x - mean
        normalized = t / tf.reduce_mean(t ** 2)
        return normalized

    def preprocess_input(self, x):
        if self.do_normalize:
            y = self.normalize(x)
        else:
            return x
        return y

    def __init__(self, n_batch, layer_sizes, do_normalize=False, load_from=None, last_part=True, name='fc'):
        super(FCPart, self).__init__(load_from)
        self.n_batch = n_batch
        self.last_part = last_part
        self.do_normalize = do_normalize
        self.layer_sizes = layer_sizes
        self.name=name

    def add_to_graph(self, input_tensor):
        shape = input_tensor.get_shape().as_list()

        input_size = 1
        for dim_size in shape[1:]:
            input_size *= dim_size
        
        prev_layer = tf.reshape(input_tensor, [-1, input_size])
        prev_size = input_size

        # inner layers
        for i in range(len(self.layer_sizes)):
            with tf.name_scope('fc_{:d}'.format(i)):
                layer_size = self.layer_sizes[i]

                # set up weights
                W = self.weight_variable([prev_size, layer_size], '{}_weights{}'.format(self.name, i))
                b = self.bias_variable([layer_size], 0.1, '{}_bias{}'.format(self.name, i))

                h = tf.matmul(prev_layer, W) + b

                # activation (not for output)
                if not self.last_part or i != len(self.layer_sizes) - 1:
                    h = tf.nn.relu(h, name='activation')
                    # h = tf.nn.tanh(h, name='activation')

                    if self.do_normalize:
                        gamma = tf.Variable(1.0, name='gamma')
                        beta = tf.Variable(0.0, name='beta')

                        mean = tf.reduce_mean(h, axis=1, keep_dims=True)
                        h = h - mean
                        h = h / tf.sqrt(tf.reduce_mean(h ** 2, axis=1, keep_dims=True))

                        h = gamma * h + beta

                prev_size = layer_size
                prev_layer = h
        return prev_layer

if __name__ == '__main__':
    fcp = FCPart([10, 10, 5], True)

    inp = tf.constant(1.0, shape=(1, 5, 5, 3))
    q = fcp.add_to_graph(inp)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(q))
