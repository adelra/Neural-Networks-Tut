#!/usr/bin/env python

"""Iris Data classification."""

__author__ = "Adel Rahimi, Sharif University of Technology"
__email__ = "Rahimi[dt]adel[at_sign]gmail[dot]com"


import pandas as pd
import tensorflow as tf


def inputs(data):
    iris_data = pd.read_table(data, sep=',', header=None)
    iris_data_one_hot_encoded = pd.get_dummies(iris_data)
    iris_data_one_hot_encoded.head()
    train_data = iris_data_one_hot_encoded.sample(frac=0.8, random_state=200)
    test_data = iris_data_one_hot_encoded.drop(train_data.index)
    iris_train_input_data = train_data.filter([0, 1, 2, 3])
    iris_train_label_data = train_data.filter(
        ['4_Iris-setosa', '4_Iris-versicolor', '4_Iris-virginica'])
    iris_test_input_data = test_data.filter([0, 1, 2, 3])
    iris_test_label_data = test_data.filter(
        ['4_Iris-setosa', '4_Iris-versicolor', '4_Iris-virginica'])
    return iris_train_input_data, iris_train_label_data, iris_test_input_data, iris_test_label_data


iris_train_input_data, iris_train_label_data, iris_test_input_data, iris_test_label_data = inputs('iris.data.txt')
x = tf.placeholder(tf.float32, [None, 4])
W = tf.Variable(tf.zeros([4, 3]))
b = tf.Variable(tf.zeros([3]))

yhat = tf.nn.softmax(tf.matmul(x, W) + b)
y = tf.placeholder(tf.float32, [None, 3])
cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y, 1), logits=yhat))
init = tf.global_variables_initializer()
lr_list = [0.001, 0.01, 0.1, 1]
display_step = 1
correct_prediction = tf.equal(tf.argmax(yhat, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for lr in lr_list:
        for epoch in range(3):

            # Usually send batches to the training step. But since the dataset is small sending all
            sess.run(tf.train.GradientDescentOptimizer(lr).minimize(cross_entropy),
                     feed_dict={x: iris_train_input_data, y: iris_train_label_data})
            if (epoch + 1) % display_step == 0:
                acc = accuracy.eval({x: iris_test_input_data, y: iris_test_label_data})
                print("lr:", lr, "epoch: ", epoch, "acc: ", acc)
