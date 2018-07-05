import tensorflow as tf
import numpy as np

'''
This is an example of making a rnn using the low level tensorflow API


'''
n_inputs = 3
n_neurons = 5

#inputs
X0 = tf.placeholder(tf.float32, [None, n_inputs])
X1 = tf.placeholder(tf.float32, [None, n_inputs])

#weight matricies:
Wx = tf.Variable(tf.random_normal(shape=[n_inputs, n_neurons],dtype=tf.float32))
Wy = tf.Variable(tf.random_normal(shape=[n_neurons, n_neurons],dtype=tf.float32))
b = tf.Variable(tf.zeros([1, n_neurons]), dtype=tf.float32)

#outputs:
Y0 = tf.tanh(tf.matmul(X0, Wx) + b)
Y1 = tf.tanh(tf.matmul(Y0, Wy) + tf.matmul(X1, Wx) + b)

Init = tf.global_variables_initializer()

#minibatch
x0_batch = np.array([[0,1,2],[3,4,5],[6,7,8],[9,0,1]]) #t = 0
x1_batch = np.array([[9,8,7],[0,0,0],[6,5,4],[3,2,1]]) #t = 1

with tf.Session() as sess:
    Init.run()
    Y0_val , Y1_val = sess.run([Y0, Y1],feed_dict={X0:x0_batch ,X1:x1_batch})

print(Y0_val)
print(Y1_val)
