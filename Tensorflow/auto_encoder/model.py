# From tensorflow official tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from Tensorflow.layers import conv, max_pool, dense, flatten
from Tensorflow.base_model import BaseModel






sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
