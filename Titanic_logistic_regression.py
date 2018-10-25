#!/usr/bin/env python

"""Titanic data logistic regression."""

__author__ = "Adel Rahimi, Sharif University of Technology"
__email__ = "Rahimi[dt]adel[at_sign]gmail[dot]com"

import numpy as np
import pandas as pd
import sklearn.model_selection as sk
import tensorflow as tf


def inputs(data):
    while True:
        for _ in range(batch_size):
            global input_x, input_y
            train = pd.read_csv(data)
            age = train["Age"].fillna(train['Age'].mean())
            age = np.array(age)
            gender = train['Sex'].replace(['female', 'male'], [0, 1])
            Pclass = np.zeros([train['Pclass'].__len__(), train['Pclass'].max()])
            for index, item in enumerate(train['Pclass']):
                Pclass[index, item - 1] = 1
            input_x = np.column_stack((age, gender, Pclass))
            input_y = np.array(train['Survived']).reshape((891, 1))
            arr = sk.train_test_split(input_x, input_y, test_size=0.33, random_state=42)
            return arr


# setting the Parameters
learning_rate = 2
training_epochs = 20
batch_size = 100  # 1, 5, 10, 100
display_step = 1
print("Input shape: ", inputs("Titanic.csv")[0].shape)
X_train, X_test, y_train, y_test = inputs("Titanic.csv")
# setting inputs
x = tf.placeholder(tf.float32, [None, 5])  # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 1], name="y")  # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([5, 1]))
b = tf.Variable(tf.zeros([1, ]), name='b')

# model
pred = tf.nn.sigmoid(tf.matmul(x, W) + b)  # sigmoid

# Minimize error
# cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
# GD Optimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize variables
init = tf.global_variables_initializer()
correct_prediction = tf.equal(pred, y)
# setting accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(input_x.shape[0] / batch_size)
        for i in range(total_batch):
            _, c, prediction = sess.run([optimizer, cost, pred], feed_dict={x: X_train,
                                                                            y: y_train})
            # average loss
            avg_cost += c / total_batch
        # print logs on each epoch
        if (epoch + 1) % display_step == 0:
            print("Epoch: ", '%04d' % (epoch + 1), "cost: ", "{:.9f}".format(avg_cost))
            acc = accuracy.eval({x: X_test, y: y_test})
            print(acc)
            if acc >= 0.6:
                break
    confusion = tf.confusion_matrix(labels=y_train, predictions=tf.argmax(prediction, 1))
    print('Confusion Matrix: \n\n', tf.Tensor.eval(confusion, feed_dict=None, session=None))
    print("Training Finished!")

    print("Final Accuracy:", accuracy.eval({x: X_test, y: y_test}))


def evaluate(x):
    output = tf.matmul(x, W) + b
    if output >= 0.5:
        output += 1
    else:
        output += 0
    return output

# TODO: confusion matrix
