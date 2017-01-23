#!/usr/bin/python
# __________________________________________________________________________________________________
# Standard class supporting serializable weights through numpy

import tensorflow as tf
import numpy as np

class SerializableWeights:
    def add_assign_weights(self, key, rhs):
        assert(len(self.weight_list) == len(rhs.weight_list))
        assert(len(self.bias_list) == len(rhs.bias_list))

        self.assign_ops[key] = []
        self_variables = self.weight_list + self.bias_list
        rhs_variables = rhs.weight_list + rhs.bias_list
        for i in range(len(self_variables)):
            self.assign_ops[key].append(tf.assign(self_variables[i], rhs_variables[i]))

    def run_assign_weights(self, key, sess):
        for assign_op in self.assign_ops[key]:
            sess.run(assign_op)

    def saveable_weights_dict(self, file_object, sess):
        variable_dict = {}
        for weight in self.weight_list + self.bias_list:
            key = weight.name
            if key.count('/'):
                key = key[key.rindex('/') + 1:]
            if key.count(':'):
                key = key[:key.index(':')]
            variable_dict[key] = sess.run(weight)
        # save
        return variable_dict

    def weight_variable(self, shape, name):
        i_max = np.sqrt(6 / np.sum(shape))
        i_min = -i_max
        # smaller initialization proved to be better
        i_max = 0.05
        i_min = -0.05
        try:
            initial = tf.constant(self.loaded[name], shape=shape)
            print('loaded {}'.format(name))
        except:
            if self.loaded:
                print('Warning: load_from given but unable to load {}'.format(name))
            initial = tf.random_uniform(shape, i_min, i_max)
        var = tf.Variable(initial, name=name)
        self.weight_list.append(var)
        return var

    def bias_variable(self, shape, value, name):
        # RELU aware - initialization
        # value=0.1
        try:
            initial = tf.constant(self.loaded[name], shape=shape)
            print('loaded {}'.format(name))
        except:
            if self.loaded:
                print('Warning: load_from given but unable to load {}'.format(name))
            initial = tf.constant(value, shape=shape)
        bias = tf.Variable(initial, name=name)
        self.bias_list.append(bias)
        return bias

    def __init__(self, load_from=None, mmap=None):
        self.assign_ops = {}
        self.weight_list = []
        self.bias_list = []
        self.loaded = None
        try:
            self.loaded = np.load(load_from, mmap_mode=mmap)
        except:
            self.loaded = None
