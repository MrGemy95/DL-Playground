# From tensorflow official tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from TF.layers import conv, max_pool, dense, flatten

mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)

# define the placeholder to pass the data and the ground_truth
x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])
x_image = tf.reshape(x, [-1, 28, 28, 1])


# network_architecture
h_conv1 = conv(x_image, num_filters=32, kernel_size=[3, 3], stride=[2, 2], name='coeenv1')
h_pool1 = max_pool(h_conv1, kernel_size=[2, 2], stride=[2, 2])
flat = flatten(h_pool1)
d1 = dense(flat, num_units=512, activation_fn=tf.nn.relu, name="densee2")
d2 = dense(d1, num_units=10)

#loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=d2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# training
for i in range(20000):
    batch_x, batch_y = mnist.train.next_batch(50)

    # visualizing accuracy
    if i % 100 == 0:
        train_accuracy = accuracy.eval(feed_dict={x: batch_x, y: batch_y})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y})

print("test accuracy %g" %  sess.run(accuracy, feed_dict={
    x: mnist.test.images, y: mnist.test.labels}))
