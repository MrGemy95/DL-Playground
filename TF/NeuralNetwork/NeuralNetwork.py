# From tensorflow official tutorial

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from TF.layers import conv, max_pool, dense, flatten

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# model
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
d1 = dense(x, num_units=1000,name="dense1")
d1 = dense(x, num_units=500,name="dense2")
d2 = dense(d1, num_units=10,name="dense3")

#loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=d2, labels=y))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# training
sess = tf.InteractiveSession()  # set itself as a defult
tf.global_variables_initializer().run()
for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(100)
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={
            x: batch_x, y: batch_y})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

print("Test Acc : ",sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}), "% \nExpected to get around 90-95% Acc")
