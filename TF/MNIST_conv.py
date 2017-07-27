#From tensorflow official tutorial
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./TF/MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, shape=[None, 784])
y = tf.placeholder(tf.float32, shape=[None, 10])

from TF.layers import conv2d,max_pool,dense,flatten
x_image = tf.reshape(x, [-1,28,28,1])
h_conv1,w1,b1=conv2d(x_image,output_dim=32,kernel_size=[3,3],stride=[2,2],name='coeenv1')
h_pool1 = max_pool(h_conv1,kernel_size=[2,2],stride=[2,2])
flat=flatten(h_pool1)
d1,w2,b2=dense(flat,output_dim=512,activation_fn=tf.nn.relu,name="densee2")
d2,w3,b3=dense(d1,output_dim=10)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


print("sess.run(node3): ",sess.run(h_pool1,feed_dict={x: mnist.train.next_batch(50)[0]}).shape)


cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=d2))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(d2,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(20000):
  batch = mnist.train.next_batch(50)
  if i%100 == 0:
    train_accuracy = accuracy.eval(feed_dict={
        x:batch[0], y: batch[1]})
    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y: batch[1]})

print("test accuracy %g"%accuracy.eval(feed_dict={
    x: mnist.test.images, y: mnist.test.labels}))