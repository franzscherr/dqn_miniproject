#!/usr/bin/python
# __________________________________________________________________________________________________
# Convolutional processing
#

import tensorflow as tf
import numpy as np

from serializable_weights import SerializableWeights

# __________________________________________________________________________________________________
# ConvolutionPart is meant to represent the first layers that process the image input
#
class ConvolutionPart(SerializableWeights):
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def normalize(self, x):
        mean = tf.reduce_mean(x)
        t = x - mean
        normalized = t / tf.sqrt(tf.reduce_mean(t ** 2))
        return normalized

    def preprocess_input(self, x):
        if self.do_normalize:
            y = self.normalize(x)
        else:
            return x
        return y

    def __init__(self, n_batch, filter_sizes, n_filters, pooling_size=None, load_from=None):
        super(ConvolutionPart, self).__init__(load_from)
        self.n_batch = n_batch
        self.do_normalize = False
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.pooling_size = None
        if pooling_size:
            self.pooling_size = [1, 2, 2, 1]
            self.pooling_size[1:3] = pooling_size
        assert(len(filter_sizes) == len(n_filters))

    def add_to_graph(self, image_input):
        # TODO: How to do normalization for convolution?
        # TODO: Different configurations, especially for PADDING
        # shape is batch_size x img_ x img_ x channels
        shape = image_input.get_shape().as_list()
        
        f_n_previous = shape[-1]
        prev_layer = image_input
        img_size = np.array(shape[1:3])

        for i in range(len(self.filter_sizes)):
            with tf.name_scope('conv_{:d}'.format(i)):
                f_size = self.filter_sizes[i]
                f_n = self.n_filters[i]

                # set up weights
                W_conv = self.weight_variable([f_size, f_size, f_n_previous, f_n], 'conv_weights{}'.format(i))
                b_conv = self.bias_variable([f_n], 0.1, 'conv_bias{}'.format(i))

                # convolute and ReLU
                h = tf.nn.relu(self.conv2d(prev_layer, W_conv) + b_conv, 
                        name='activation')
                if self.pooling_size:
                    h = tf.nn.max_pool(h, ksize=self.pooling_size, strides=self.pooling_size, padding='SAME')
                    img_size = img_size / np.array(self.pooling_size[1:3])

                f_n_previous = f_n
                prev_layer = h

        # -- output convoluted size
        self.output_size = img_size
        return prev_layer

if __name__ == '__main__':
    vp = ConvolutionPart([5, 5], [32, 64], [2, 2])

    img = tf.constant(1.0, shape=[1, 64, 128, 3])
    convout = vp.add_to_graph(img)
