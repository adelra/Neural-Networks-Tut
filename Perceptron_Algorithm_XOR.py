#!/usr/bin/env python

"""XOR Perceptron"""

__author__ = "Adel Rahimi, Sharif University of Technology"
__email__ = "Rahimi[dt]adel[at_sign]gmail[dot]com"

import tensorflow as tf

T, F = 1., -1.
bias = 1.

input = [
    [T, T],
    [T, F],
    [F, T],
    [F, F],
]

output = [
    [F],
    [T],
    [T],
    [F],
]

w1 = tf.Variable(tf.random_normal([2, 2]))
b1 = tf.Variable(tf.zeros([2]))

w2 = tf.Variable(tf.random_normal([2, 1]))
b2 = tf.Variable(tf.zeros([1]))

out1 = tf.tanh(tf.add(tf.matmul(input, w1), b1))
out2 = tf.tanh(tf.add(tf.matmul(out1, w2), b2))

error = tf.subtract(output, out2)
mse = tf.reduce_mean(tf.square(error))
train = tf.train.GradientDescentOptimizer(0.01).minimize(mse)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    err, target = 1, 0.01
    epoch, max_epochs = 0, 5000
    while err > target and epoch < max_epochs:
        epoch += 1
        err, _ = sess.run([mse, train])
        print('epoch:', epoch, 'mse:', err)
