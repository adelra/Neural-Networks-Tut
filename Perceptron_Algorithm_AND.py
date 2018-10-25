#!/usr/bin/env python

""" Perceptron algorithm - AND operator """

__author__ = "Adel Rahimi, Sharif University of Technology"
__email__ = "Rahimi[dt]adel[at_sign]gmail[dot]com"

import tensorflow as tf

T, F = 1., -1.
bias = 1.
input_x = [
    [T, T, bias],
    [T, F, bias],
    [F, T, bias],
    [F, F, bias],
]

input_y = [
    [T],
    [F],
    [F],
    [F],
]

w = tf.Variable(tf.random_normal([3, 1]))


# step(x) = { 1 if x > 0
#  -1 otherwise }
def step(x):
    is_greater = tf.greater(x, 0)
    as_float = tf.to_float(is_greater)
    doubled = tf.multiply(as_float, 2)
    return tf.subtract(doubled, 1)


out = step(tf.matmul(input_x, w))
error = tf.subtract(input_y, out)
mse = tf.reduce_mean(tf.square(error))
delta = tf.matmul(input_x, error, transpose_a=True)
train = tf.assign(w, tf.add(w, delta))

init = tf.global_variables_initializer()
with tf.Session() as sess:
    # initialize all the variables
    sess.run(init)
    err, target = 1, 0
    epoch, max_epochs = 0, 10
    while err > target and epoch < max_epochs:
        epoch += 1
        err, _ = sess.run([mse, train])
        print('epoch:', epoch, 'mse:', err)
