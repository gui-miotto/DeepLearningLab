from typing import List
from functools import reduce
import tensorflow as tf

class Layer:
    def __init__(self):
        # Layer's output node
        self.out = None
    
    def __call__(self):
        return self.out

    def build(self, x: tf.Tensor, train_mode: tf.Tensor):
        """ 
        Given the input x, builds the part of the graph correspondent to the layer.
        """
        raise NotImplementedError

class Input(Layer):
    def __init__(self, input_shape: List[int]):
        super().__init__()
        self.input_shape = [None] + input_shape
    
    def build(self, x, train_mode: tf.Tensor):
        self.out = tf.placeholder('float', self.input_shape, name='Network_Input')

class Linear(Layer):
    def __init__(self, n_units: int):
        super().__init__()
        self.n_units = n_units

    def build(self, x: tf.Tensor, train_mode: tf.Tensor):
        n_inputs = x.shape[1].value
        n_outputs = self.n_units
        with tf.name_scope('Linear_Layer'):
            self.w = tf.Variable(tf.random_normal([n_inputs, n_outputs]), name='Linear_weights')
            self.b = tf.Variable(tf.random_normal([n_outputs]), name='Linear_biases')
            self.out = tf.add(tf.matmul(x, self.w), self.b, name='Linear_output')

class Conv2d(Layer):
    def __init__(self, filters: int, kernel_size: int, stride: int=1, pad: int=0):
        super().__init__()
        self.filters = filters
        self.ksize = kernel_size
        self.stride = stride
        self.paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0,0]])

    def build(self, x: tf.Tensor, train_mode: tf.Tensor):
        in_filters = x.shape[-1].value
        strd = [1, self.stride, self.stride, 1]
        with tf.name_scope('Conv_Layer'):
            x_pad= tf.pad(x, self.paddings, 'CONSTANT')
            self.w = tf.Variable(tf.random_normal([self.ksize, self.ksize, in_filters, self.filters]), name='Conv_weights')
            self.b = tf.Variable(tf.random_normal([self.filters]), name='Conv_biases')
            self.out = tf.add(tf.nn.conv2d(x_pad, self.w, strides=strd, padding='SAME'), self.b, name='Conv_output')

class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def build(self, x: tf.Tensor, train_mode: tf.Tensor):
        inp_shape = x.get_shape().as_list()
        features = reduce(lambda a, b: a * b, inp_shape[1:])
        self.out = tf.reshape(x, shape=[-1, features], name='Flatten')

class Dropout(Layer):
    def __init__(self, drop_probability):
        super().__init__()
        self.keep_prob = 1.0 - drop_probability

    def build(self, x: tf.Tensor, train_mode: tf.Tensor):
        self.out = tf.cond(
            train_mode, 
            lambda: tf.nn.dropout(x, self.keep_prob),
            lambda: x,
            name='Dropout_cond')

class ReLU(Layer):
    def __init__(self):
        super().__init__()

    def build(self, x: tf.Tensor, train_mode: tf.Tensor):
        self.out = tf.nn.relu(x)

       
