
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from compas_ml.helpers import labels_to_onehot

from numpy import array
from numpy import float32
from numpy import newaxis
from numpy.random import choice

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'convolution',
]


def convolution(training_data, training_labels, testing_data, testing_labels, classes, fdim, features, neurons,
                steps, batch, path, multi_layer=False):

    """ Pixel based Neural Network with Convolution layers.

    Parameters
    ----------
    training_data : list
        Training data of length m.
    training_labels : list
        Training labels of length m.
    testing_data : list
        Testing data of length p.
    testing_labels : list
        Testing labels of length p.
    classes : int
        Number of classes.
    fdim : int
        Filter size in x and y.
    features : int
        Number of features per convolution layer.
    neurons : int
        Number of neurons.
    steps : int
        Number of analysis steps.
    batch : int
        Batch size of images per step.
    path : str
        Model directory.

    Returns
    -------
    None

    """

    print('***** Session started *****')

    training_data = array(training_data, dtype=float32)
    testing_data  = array(testing_data, dtype=float32)

    dims = training_data.shape

    if len(dims) == 3:
        m, dimx, dimy = dims
        channels = 1
        training_data = training_data[:, :, :, newaxis]
        testing_data  = testing_data[:, :, :, newaxis]

    elif len(dims) == 4:
        m, dimx, dimy, channels = dims

    training_labels = array(labels_to_onehot(labels=training_labels, classes=classes))
    testing_labels  = array(labels_to_onehot(labels=testing_labels, classes=classes))


    def train():

        session = tf.InteractiveSession()

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, dimx, dimy, channels])
            y = tf.placeholder(tf.float32, [None, classes])

        def weight_variable(shape):

            return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

        def bias_variable(shape):

            return tf.Variable(tf.constant(0.01, shape=shape))

        def conv2d(x, W):

            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2x2(x):

            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        def conv_layer(input, shape, name, activation=tf.nn.relu):

            with tf.name_scope(name):

                with tf.name_scope('weights'):
                    weights = weight_variable(shape)

                with tf.name_scope('biases'):
                    biases = bias_variable([shape[-1]])

                return activation(conv2d(input, weights) + biases)

        def full_layer(input, size, name):

            with tf.name_scope(name):

                in_size = int(input.get_shape()[1])
                with tf.name_scope('weights'):
                    weights = weight_variable([in_size, size])

                with tf.name_scope('biases'):
                    biases = bias_variable([size])

                return tf.matmul(input, weights) + biases

        keep_prob = tf.placeholder(tf.float32)

        if not multi_layer:

            # x:          -1, dimx, dimy, channels
            # conv1:      -1, dimx, dimy, features
            # conv1_pool: -1, dimx/2, dimy/2, features
            # weights1:   fdim, fdim, channels, features
            conv1 = conv_layer(x, shape=[fdim, fdim, channels, features], name='conv_1')
            conv1_pool = max_pool_2x2(conv1)

            # conv2:      -1, dimx/2, dimy/2, 2*features
            # conv2_pool: -1, dimx/4, dimy/4, 2*features
            # conv2_flat: -1, dimx/4 * dimy/4 * 2*features
            # weights2:   fdim, fdim, features, 2*features
            conv2 = conv_layer(conv1_pool, shape=[fdim, fdim, features, 2 * features], name='conv_2')
            conv2_pool = max_pool_2x2(conv2)
            conv2_flat = tf.reshape(conv2_pool, [-1, int(0.25 * 0.25 * dimx * dimy * 2 * features)])

        else:

            conv1_1 = conv_layer(x_, shape=[fdim, fdim, channels, features])
            conv1_2 = conv_layer(conv1_1, shape=[fdim, fdim, features, features])
            conv1_3 = conv_layer(conv1_2, shape=[fdim, fdim, features, features])
            conv1_pool = max_pool_2x2(conv1_3)
            conv1_drop = tf.nn.dropout(conv1_pool, keep_prob=keep_prob)

            conv2_1 = conv_layer(conv1_drop, shape=[fdim, fdim, features, 2 * features])
            conv2_2 = conv_layer(conv2_1, shape=[fdim, fdim, 2 * features, 2 * features])
            conv2_3 = conv_layer(conv2_2, shape=[fdim, fdim, 2 * features, 2 * features])
            conv2_pool = max_pool_2x2(conv2_3)
            conv2_drop = tf.nn.dropout(conv2_pool, keep_prob=keep_prob)
            conv2_flat = tf.reshape(conv2_drop, [-1, int(0.25 * 0.25 * dimx * dimy * 2 * features)])

        # full1: -1, neurons
        full1 = tf.nn.relu(full_layer(conv2_flat, size=neurons, name='full_1'))
        full1_drop = tf.nn.dropout(full1, keep_prob=keep_prob)

        # logits: -1, classes
        logits = full_layer(full1_drop, size=classes, name='logits')

        with tf.name_scope('loss'):
            with tf.name_scope('total'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
        tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        training_writer = tf.summary.FileWriter('{0}/training/'.format(path), session.graph)
        testing_writer  = tf.summary.FileWriter('{0}/testing/'.format(path))

        tf.global_variables_initializer().run()

        def feed_dict(train):

            if train:
                select  = choice(m, batch, replace=False)
                x_batch = training_data[select, :, :, :]
                y_batch = training_labels[select, :]
                k = 0.5

            else:
                x_batch = testing_data
                y_batch = testing_labels
                k = 1.0

            return {x: x_batch, y: y_batch, keep_prob: k}

        for i in range(steps):

            if (i + 1) % 10 == 0:
                summary, acc = session.run([merged, accuracy], feed_dict=feed_dict(False))
                testing_writer.add_summary(summary, i + 1)
                print('Accuracy Step %s: %s' % (i + 1, acc))

            else:
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()
                summary, _ = session.run([merged, train_step], feed_dict=feed_dict(True), options=options,
                                         run_metadata=run_metadata)
                training_writer.add_run_metadata(run_metadata, 'Step: %03d' % (i + 1))
                training_writer.add_summary(summary, i + 1)

        training_writer.close()
        testing_writer.close()


    def main(_):

        if tf.gfile.Exists(path):
            tf.gfile.DeleteRecursively(path)
        tf.gfile.MakeDirs(path)

        train()

        print('***** Session finished *****')

    tf.app.run(main=main)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    # MNIST
    # ------------------------------------------------------------------------------

    # from scipy.misc import imread

    # from os import listdir

    # folder = '/home/al/compas_ml/data/mnist/'

    # training_data   = []
    # testing_data    = []
    # training_labels = []
    # testing_labels  = []

    # for i in ['testing', 'training']:
    #     for j in range(10):

    #         prefix = '{0}/{1}/{2}'.format(folder, i, j)
    #         files  = listdir(prefix)[:100]

    #         for file in files:

    #             image = imread('{0}/{1}'.format(prefix, file))
    #             if i == 'training':
    #                 training_data.append(image)
    #                 training_labels.append(j)
    #             else:
    #                 testing_data.append(image)
    #                 testing_labels.append(j)

    # path = '/home/al/temp/'

    # convolution(training_data, training_labels, testing_data, testing_labels, classes=10, fdim=5, features=32,
    #             neurons=1024, steps=200, batch=200, path=path)


    # ------------------------------------------------------------------------------
    # CIFAR10
    # ------------------------------------------------------------------------------

    from scipy.misc import imread

    from os import listdir

    import json

    folder = '/home/al/compas_ml/data/cifar10/'

    training_data   = []
    testing_data    = []
    training_labels = []
    testing_labels  = []

    with open(folder + 'columns.json', 'r') as f:
        labels = json.load(f)

    for i in ['testing', 'training']:

        prefix = '{0}/{1}/'.format(folder, i)

        for file in listdir(prefix):

            image = imread('{0}/{1}'.format(prefix, file))
            j = labels[file.split('_')[1][:-4]]

            if i == 'training':
                training_data.append(image)
                training_labels.append(j)

            else:
                testing_data.append(image)
                testing_labels.append(j)

    path = '/home/al/temp/'

    convolution(training_data, training_labels, testing_data, testing_labels, fdim=5, features=30, classes=10,
                steps=500, batch=300, neurons=500, multi_layer=0, path=path)
