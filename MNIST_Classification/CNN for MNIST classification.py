import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""CNN neural network to predict MNIST classes.
The network architecture are as follows:
 64 Conv 5x5 stride=1 ReLU
Max Pooling 2x2 stride=2 
64 Conv 5x5 stride=1 ReLU
Max Pooling 2x2 stride=2 
FC 512 units ReLU
Softmax 10 units

Reaches the Accuracy of 94% after 900 epochs.
"""

mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# Parameters
learning_rate = 0.01
epochs = 200
batch_size = 100
display_step = 1

# Network Parameters
num_input = 784  # image shape= 28*28
num_classes = 10  # 0-9 digits

# x shape = [batch, height, width, channel]
# height and width for MNIST is 28x28 and since the images are grayscale the channel is 1
x = tf.placeholder("float", [None, 28, 28, 1])
y = tf.placeholder("float", [None, num_classes])


# train_X = data.train.images.reshape(-1, 28, 28, 1)
# test_X = data.test.images.reshape(-1,28,28,1)

def convolution(x, W, b, stride):
    '''
    convolution function
    :param x: input tensor
    :param W: connecting weight
    :param b: bias
    :param stride: number of steps for convolution
    :return: relu output of the calculation of the convolution
    '''
    x = tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool(x, k, stride):
    '''
    Maxpool function
    :param x: input tensor
    :param k: kernel size
    :param stride: stride
    :return: output of the maxpooling as a tensor
    '''
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, stride, stride, 1], padding='SAME')


# Weights
weight_conv1 = tf.get_variable('wc1', shape=(5, 5, 1, 64))
weight_conv2 = tf.get_variable('wc2', shape=(5, 5, 64, 128))
weight_fc = tf.get_variable('wd', shape=(4 * 4 * 128, 500))
weight_out = tf.get_variable('wo', shape=(500, num_classes))

# Bias
bias_conv1 = tf.get_variable('B0', shape=64)
bias_conv2 = tf.get_variable('B1', shape=128)
bias_fc = tf.get_variable('B3', shape=500)
bias_out = tf.get_variable('B4', shape=10)

'''
64 Conv 5x5 stride=1 ReLU
Max Pooling 2x2 stride=2 
64 Conv 5x5 stride=1 ReLU
Max Pooling 2x2 stride=2 
FC 512 units ReLU
Softmax 10 units

'''

conv1 = convolution(x, weight_conv1, bias_conv1, stride=1)
conv1 = maxpool(conv1, k=2, stride=3)
conv2 = convolution(conv1, weight_conv2, bias_conv2, stride=1)
conv2 = maxpool(conv2, k=2, stride=3)
fc = tf.reshape(conv2, [-1, weight_fc.get_shape().as_list()[0]])
fc = tf.add(tf.matmul(fc, weight_fc), bias_fc)
fc = tf.nn.relu(fc)
out = tf.add(tf.matmul(fc, weight_out), bias_out)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)
correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))

# calculate accuracy across all the given images and average them out.
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('./Output', sess.graph)
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
    for i in range(epochs):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape(-1, 28, 28, 1)

        # Run optimization op (backprop).
        # Calculate batch loss and accuracy
        opt = sess.run(optimizer, feed_dict={x: batch_x,
                                             y: batch_y})
        loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
                                                          y: batch_y})
        print("Iter " + str(i) + ", Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))
        test_X, test_y = mnist.test.next_batch(batch_size)
        test_X = test_X.reshape(-1, 28, 28, 1)
        # Calculate accuracy for all 10000 mnist test images
        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
    print("Training Finished!")
