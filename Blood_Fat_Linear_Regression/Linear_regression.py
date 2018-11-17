#!/usr/bin/env python

"""Q2.py: Linear Regression algorithm only using numpy."""

__author__ = "Adel Rahimi, Sharif University of Technology"
__email__ = "Rahimi[dt]adel[at_sign]gmail[dot]com"

import numpy as np


# reading the input
def inputs(fileName):
    global weight, age
    global blood_fat_content
    print("Loading data:", fileName)
    data = np.loadtxt(fileName, delimiter='  ', skiprows=36)
    # weight and age are two 2d arrays
    weight = (data[:, 2])
    age = (data[:, 3])
    blood_fat_content = np.array(data[:, 4], dtype=float)
    return weight, age, blood_fat_content


# calling input funciotn
inputs("blood fat.txt")
m = len(age)
x0 = np.ones(m)
X = np.array([x0, weight, age]).T
# X = weight_age.T
B = np.array([0, 0, 0])
Y = np.array(blood_fat_content)
alpha = 0.0000001


def cost_function(X, Y, B):
    m = len(Y)
    J = np.sum((X.dot(B) - Y) ** 2) / (2 * m)
    return J


inital_cost = cost_function(X, Y, B)
print("Initial Cost", inital_cost)


def loss(X, Y):
    Y_pred = inference(X)
    rmse = np.sqrt(sum((Y - Y_pred) ** 2))
    return rmse


# The train function
def train(X, Y, B, alpha, iterations):
    cost_hist = [0] * iterations
    m = len(Y)

    for iteration in range(iterations):
        h = X.dot(B)
        # The Difference between b w hypothesis and ground truth Y
        loss = h - Y
        # Gradient
        gradient = X.T.dot(loss) / m
        # Changing Values of B using Gradient
        B = B - alpha * gradient
        # New Cost Value
        cost = cost_function(X, Y, B)
        cost_hist[iteration] = cost

    return B, cost_hist


newB, cost_hist = train(X, Y, B, alpha, 10000)


def inference(x):
    # our new calculation of yhat equals => B + newB * x
    yhat = newB[0] + newB[1] * x[0] + newB[2] * x[1]
    return yhat


# Final Cost of new B
print("Final Loss", cost_hist[-1])
for weight in range(60, 101):
    print("yhat for age 25 and weight:", weight, "\t", inference([weight, 25]))
