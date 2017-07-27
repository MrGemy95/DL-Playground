#From tensorflow official tutorial

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./TF/MNIST_data/", one_hot=True)


#model
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
out = tf.matmul(x, W) + b
y = tf.placeholder(tf.float32, [None, 10])
cross_entropy= tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out,labels=y))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})

#training
sess = tf.InteractiveSession()     #set itself as a defult
tf.global_variables_initializer().run()
for _ in range(10000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y: batch_ys})


correct_prediction = tf.equal(tf.argmax(out,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))