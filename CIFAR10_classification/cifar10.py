import pickle

import matplotlib.pyplot as plt
import tensorflow as tf
import math
import numpy as np
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
cifar10_dataset_folder_path = 'cifar-10-batches-py'
tf.reset_default_graph()
batch_size = 500


def load_cfar10_test(cifar10_dataset_folder_path, batch_id, batch_size):
    """
    Loading Cifar10 test dataset
    :param cifar10_dataset_folder_path: Folder in which the cifar10 test dataset exists
    :param batch_id: the batch id for the cifar10 test data in this case it will be test_batch
    :param batch_size: size of our batches
    :return: features and labels
    """
    with open(cifar10_dataset_folder_path + "/" + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    while True:
        for st in range(0, len(features), batch_size):
            end = min(st + batch_size, len(features))
            yield features[st:end], labels[st:end]


def get_weights_from_layer(layer, input_x):
    '''
    Get the weights from the selected activation layer
    :param layer: The layer indicated from which the weights will be extracted
    :param input_x: The input image for the layer
    :return:
    '''
    units = sess.run(layer, feed_dict={x:input_x})
    plot_weight(units)

def plot_weight(units):
    """
    Visualize weights from selected layer
    :param units: tensor of the units
    :return: matplotlib plot
    """
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")




def load_cifar10_batch(cifar10_dataset_folder_path, batch_id, batch_size):
    '''
    Funciton to load Cifar10 in batches
    :param cifar10_dataset_folder_path: The directory in which cifar10 exists
    :param batch_id: Batch ID for cifar10
    :param batch_size: Batch size for input
    :return: features and labels for cifar10
    '''
    with open(cifar10_dataset_folder_path + '/data_batch_' + str(batch_id), mode='rb') as file:
        # note the encoding type is 'latin1'
        batch = pickle.load(file, encoding='latin1')

    features = batch['data'].reshape((len(batch['data']), 3, 32, 32)).transpose(0, 2, 3, 1)
    labels = batch['labels']
    while True:
        for start in range(0, len(features), batch_size):
            end = min(start + batch_size, len(features))
            yield features[start:end], labels[start:end]

# hyper parameters
learning_rate = 0.0001
epochs = 20
display_step = 1


# to plot a sample image from cifar10
# __ = next(load_cifar10_batch(cifar10_dataset_folder_path, 1, 1))
# image = __[0]
# image = image.reshape([32, 32, 3])
# plt.imshow(image)
# plt.show()


# x shape = [batch, height, width, channel]
# height and width for MNIST is 28x28 and since the images are grayscale the channel is 1
x = tf.placeholder("float", [None, 32, 32, 3])
y = tf.placeholder(shape=[None, ], name='labels', dtype=tf.int64)





def convolution(x, W, b, stride):
    '''
    we will define a conv2d function to do our convolutions
    :param x: input x
    :param W: weight
    :param b: bias
    :param stride: stride for the convolution
    :return: The output of relu on convolution
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


initializer = tf.random_normal_initializer(stddev=0.1)

# Weights
weight_conv1 = tf.get_variable('wc1', shape=(5, 5, 3, 64), initializer=initializer)
weight_conv2 = tf.get_variable('wc2', shape=(5, 5, 64, 128), initializer=initializer)
weight_fc = tf.get_variable('wd', shape=(4 * 4 * 128, 500), initializer=initializer)
weight_out = tf.get_variable('wo', shape=(500, len(labels)), initializer=initializer)

# Bias
bias_conv1 = tf.get_variable('B0', shape=64, initializer=initializer)
bias_conv2 = tf.get_variable('B1', shape=128, initializer=initializer)
bias_fc = tf.get_variable('B3', shape=500, initializer=initializer)
bias_out = tf.get_variable('B4', shape=10, initializer=initializer)

'''
64 Conv 5x5 stride=1 ReLU
Max Pooling 2x2 stride=2 
64 Conv 5x5 stride=1 ReLU
Max Pooling 2x2 stride=2 
FC 512 units ReLU
Softmax 10 units
'''
keep_prob = 0.5
conv1 = convolution(x, weight_conv1, bias_conv1, stride=1)
dropout_conv1 = tf.nn.dropout(conv1, keep_prob=keep_prob)
conv1_maxpool = maxpool(dropout_conv1, k=2, stride=3)

conv2 = convolution(conv1_maxpool, weight_conv2, bias_conv2, stride=1)
dropout_conv2 = tf.nn.dropout(conv2, keep_prob=keep_prob)
conv2_maxpool = maxpool(dropout_conv2, k=2, stride=3)
fc = tf.reshape(conv2_maxpool, [-1, weight_fc.get_shape().as_list()[0]])
fc = tf.add(tf.matmul(fc, weight_fc), bias_fc)
fc = tf.nn.relu(fc)
out = tf.add(tf.matmul(fc, weight_out), bias_out)

cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=out, labels=y))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(out, 1, output_type=tf.int64), y)
saver = tf.train.Saver()

# calculate accuracy across all the given images and average them out.
accuracy = 100 * tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    # variable initialization
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    for epoch in range(epochs):
        n_batches = 5
        for batch_number in range(1, n_batches + 1):
            datagen = load_cifar10_batch(cifar10_dataset_folder_path, 1, batch_size)
            # for batch_features, batch_labels in load_preprocess_training_batch(batch_i, batch_size):
            # print(next(load_cifar10_batch(cifar10_dataset_folder_path,batch_id=1,batch_size=batch_size))[0].shape)
            data = next(datagen)
            X_batch = data[0]
            y_batch = data[1]
            for i in range(10000 // batch_size):
                try:
                    # , x_test, y_test
                    X_batch = X_batch.reshape(-1, 32, 32, 3)
                    feed_dict = {x: X_batch, y: y_batch}
                    opt = sess.run(optimizer, feed_dict=feed_dict)
                    loss, acc = sess.run([cost, accuracy], feed_dict=feed_dict)
                    print("Iter " + str(epoch) + ", Loss= " + \
                          "{:.6f}".format(loss) + ", Training Accuracy= " + \
                          "{:.5f}".format(
                              acc))
                    print("batch number:", batch_number)
                except StopIteration:
                    break
        epoch += 1
    print('Training Finished')
    save_path = saver.save(sess, "model/model.ckpt")
    print("Model saved in path: %s" % save_path)

    test_datagen = load_cfar10_test(cifar10_dataset_folder_path, 'test_batch', 1000)
    data = next(test_datagen)
    X_batch = data[0]
    y_batch = data[1]
    X_batch = X_batch.reshape(-1, 32, 32, 3)
    feed_dict = {x: X_batch, y: y_batch}
    loss, acc = sess.run([cost, accuracy], feed_dict=feed_dict)
    print("Loss= " + \
          "{:.6f}".format(loss) + ", Testing Accuracy= " + \
          "{:.5f}".format(
              acc))
    print(y_batch[0])
    test_item = X_batch[0].reshape([1,32,32,3])
    get_weights_from_layer(conv1, test_item)
    for i in range(0,10001):
      if y_batch[i] == 7:
        print(i)
        x_horse = X_batch[i].reshape([1,32,32,3])
        y_horse = y_batch[i]
        break
    get_weights_from_layer(conv1, x_horse)
