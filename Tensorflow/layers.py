import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers
import math
import numpy as np




def flatten(x):
    return tf.contrib.layers.flatten(x)



def deconv(x, output_shape, kernel_size=(3, 3), padding='SAME', stride=(1, 1), name="",activation=None):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], output_shape[-1], x.get_shape()[-1]]

        w = get_deconv_filter(kernel_shape)
        deconv = tf.nn.conv2d_transpose(x, w, tf.stack(output_shape), strides=stride, padding=padding)

        b = tf.get_variable('biases_deconv', [output_shape[-1]], initializer=tf.constant_initializer(0.01))
        out = tf.nn.bias_add(deconv, b)
    if activation != None:
        out = activation(out)
    return out


def get_deconv_filter(f_shape):
    """
    Bilinear Deconv filters
    :param f_shape:
    :return weights:
    """
    width = f_shape[0]
    heigh = f_shape[0]
    f = math.ceil(width / 2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear

    init = tf.constant_initializer(value=weights, dtype=tf.float32)
    return tf.get_variable(name="up_filter", initializer=init, shape=weights.shape)





def lstm(x,num_units,time_steps,name='lstm',return_states=False):
    with tf.name_scope(name):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
        outputs, states = tf.nn.dynamic_rnn(lstm_cell, x, dtype=tf.float32)
        if return_states:
            return outputs,states #return all output and states

        return outputs