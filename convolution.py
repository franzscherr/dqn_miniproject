#!/usr/bin/python
# __________________________________________________________________________________________________
# Convolutional processing
#

import tensorflow as tf
import numpy as np

# __________________________________________________________________________________________________
# ConvolutionPart is meant to represent the first layers that process the image input
#
class ConvolutionPart:
    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

    def weight_variable(self, shape, name=None):
        # TODO: NN SLIDES INITIALIZATION, is this correct for convolutional
        # layers?
        # PRO: m + n is input + output, which is exactly this
        i_max = np.sqrt(6 / np.sum(shape))
        i_min = -i_max
        initial = tf.random_uniform(shape, i_min, i_max)
        if name:
            return tf.Variable(initial, name=name)
        return tf.Variable(initial)

    def bias_variable(self, shape, name):
        # RELU aware - initialization
        initial = tf.constant(0.1, shape=shape)
        if name:
            return tf.Variable(initial, name=name)
        return tf.Variable(initial)

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

    def __init__(self, filter_sizes, n_filters, pooling_size=None):
        self.do_normalize = False
        self.filter_sizes = filter_sizes
        self.n_filters = n_filters
        self.pooling_size = None
        if pooling_size:
            self.pooling_size = [1, 2, 2, 1]
            self.pooling_size[1:3] = pooling_size
        assert(len(filter_sizes) == len(n_filters))

    def process_image(self, image_input):
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
                w_name = 'weights'.format(i)
                b_name = 'bias'.format(i)

                # set up weights
                W_conv = self.weight_variable([f_size, f_size, f_n_previous, f_n], w_name)
                b_conv = self.bias_variable([f_n], b_name)

                # convolute and ReLU
                h = tf.nn.relu(self.conv2d(prev_layer, W_conv) + b_conv, 
                        name='activation'.format(i))
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
    convout = vp.process_image(img)
