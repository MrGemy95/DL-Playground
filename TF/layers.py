import tensorflow as tf
from tensorflow.contrib.layers.python.layers import initializers









def lstm(x,num_units,time_steps,output_dim):
    x = tf.unstack(x, time_steps , 1)
    print(x.get_shape())
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units, forget_bias=1.0)
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    print (outputs ,states)  #test
    print (outputs[-1])  #test

    return outputs, states        #return all output and states




def conv(x,
         num_filters,
         kernel_size,
         stride,
         initializer=tf.contrib.layers.xavier_initializer(),
         activation_fn=tf.nn.relu,return_params=False,
         padding='VALID',
         name='conv'):
    with tf.variable_scope(name):
        stride = [ stride[0], stride[1]]
        kernel_shape = [kernel_size[0], kernel_size[1], x.get_shape()[-1], num_filters]

        w = tf.get_variable('w', kernel_shape, tf.float32, initializer=initializer)
        print(stride)
        conv = tf.nn.convolution(x, w, padding,stride)

        b = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, b)

    if activation_fn != None:
        out = activation_fn(out)
    if return_params:
        return out, w, b
    else:
        return out
def max_pool(x,
           kernel_size,
           stride,
           padding='VALID',
           name='pool2d'):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [1,kernel_size[0], kernel_size[1],1]
        pool = tf.nn.max_pool(x,ksize=kernel_shape, strides=stride, padding=padding)
    return pool

def flatten(x):
    return tf.contrib.layers.flatten(x)

def dropout():
    pass

def dense(input_, num_units, activation_fn=None, w_initializer= tf.contrib.layers.xavier_initializer(),
          b_initializer= tf.constant_initializer(0.0),return_params=False, name='dense'):
    shape = input_.get_shape().as_list()

    with tf.variable_scope(name):
        w = tf.get_variable('w', [shape[1], num_units], tf.float32,
                            initializer=w_initializer)
        b = tf.get_variable('bias', [num_units],
                            initializer=b_initializer)

        out = tf.nn.bias_add(tf.matmul(input_, w), b)

        if activation_fn != None:
            out = activation_fn(out)
        if return_params:
            return out, w, b
        else:
            return out