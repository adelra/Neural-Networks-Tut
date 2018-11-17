#!/usr/bin/env python

"""Feedforward neural network to predict MNIST classes.
This network uses two hidden layers with ReLU activation functions and 500 hidden units.
Reaches the Accuracy of ~60% after 500 epochs and ~75% after 900 epochs
"""

__author__ = "Adel Rahimi, Sharif University of Technology"
__email__ = "Rahimi[dt]adel[at_sign]gmail[dot]com"

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST Data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
epochs = 900
batch_size = 100
display_step = 1

# Defining Parameters
hlayer_units_1 = 500  # hidden units in the first layer
hlayer_units_2 = 500  # hidden units in the second layer
number_input = 784
number_classes = 10  # Our final softmax classes - 0-9 digits

x = tf.placeholder("float", [None, number_input])
y = tf.placeholder("float", [None, number_classes])

initializer = tf.random_normal_initializer(stddev=0.1)
W_1 = tf.get_variable("Hidden_layer1_W", shape=[number_input, hlayer_units_1], initializer=initializer)
b_1 = tf.get_variable("Hidden_layer1_b", shape=[hlayer_units_1], initializer=initializer)
W_2 = tf.get_variable("Hidden_layer2_W", shape=[hlayer_units_1, hlayer_units_2], initializer=initializer)
b_2 = tf.get_variable("Hidden_layer2_b", shape=[hlayer_units_2], initializer=initializer)
W_output = tf.get_variable("Output_W", shape=[hlayer_units_2, number_classes], initializer=initializer)
b_output = tf.get_variable("Output_b", shape=[number_classes], initializer=initializer)

hidden = tf.nn.relu(tf.matmul(x, W_1) + b_1)
hidden2 = tf.nn.relu(tf.matmul(hidden, W_2) + b_2)

# calculating output
output = tf.nn.softmax(tf.matmul(hidden2, W_output) + b_output)

# defining loss function
# returns a scalar Tensor representing the mean loss value.
loss = tf.losses.softmax_cross_entropy(y, output)

# accuracy calculation
# t.equal returns the truth value of (x == y) element-wise.
prediction = tf.equal(tf.argmax(y, 1), tf.argmax(output, 1))
accuracy = 100 * tf.reduce_mean(tf.cast(prediction, tf.float32))

# training with GradientDescentOptimizer and learning_rate of the variable learning_rate
train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.Session() as sess:
    # variable initialization
    sess.run(tf.global_variables_initializer())
    print('trainable params:')
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)
    for epoch in range(epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        acc, _, loss_ = sess.run([accuracy, train, loss], feed_dict={x: batch_x, y: batch_y})
        print('Epoch: {} Accuracy: {:.3f}, Loss: {}'.format(epoch, acc, loss_))
    print('Final Accuracy: {:.3f}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                                                        y: mnist.test.labels})))
