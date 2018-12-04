
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from compas_ml.utilities import labels_to_onehot

from numpy import array
from numpy import float32
from numpy import newaxis
from numpy import reshape
from numpy.random import choice

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'pixel',
]


def pixel(training_data, training_labels, testing_data, testing_labels, classes, steps, batch, path, neurons):

    """ Pixel based Neural Network.

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
    steps : int
        Number of analysis steps.
    batch : int
        Batch size of images per step.
    path : str
        Model directory.
    neurons : int
        Number of neurons.

    Returns
    -------
    None

    """

    print('***** Session started *****')

    training_data = array(training_data, dtype=float32)
    testing_data  = array(testing_data, dtype=float32)

    dims = training_data.shape
    p    = testing_data.shape[0]

    if len(dims) == 3:
        channels = 1
        m, dimx, dimy = dims
        training_data = training_data[:, :, :, newaxis]
        testing_data  = testing_data[:, :, :, newaxis]

    elif len(dims) == 4:
        m, dimx, dimy, channels = dims

    length = dimx * dimy * channels

    training_labels = array(labels_to_onehot(labels=training_labels, classes=classes))
    testing_labels  = array(labels_to_onehot(labels=testing_labels, classes=classes))


    def train():

        session = tf.InteractiveSession()

        with tf.name_scope('input'):
            x = tf.placeholder(tf.float32, [None, length])
            y = tf.placeholder(tf.float32, [None, classes])

        def weight_variable(shape):

            return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

        def bias_variable(shape):

            return tf.Variable(tf.constant(0.01, shape=shape))

        def variable_summaries(var):

            with tf.name_scope('summaries'):
                mean = tf.reduce_mean(var)
                tf.summary.scalar('mean', mean)

            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))

            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

        def nn_layer(input, input_dim, output_dim, name, activation=tf.nn.relu):

            with tf.name_scope(name):

                with tf.name_scope('weights'):
                    weights = weight_variable([input_dim, output_dim])
                    variable_summaries(weights)

                with tf.name_scope('biases'):
                    biases = bias_variable([output_dim])
                    variable_summaries(biases)

                with tf.name_scope('preactivation'):
                    preactivate = tf.matmul(input, weights) + biases
                    tf.summary.histogram('preactivations', preactivate)

                activations = activation(preactivate, name='activation')
                tf.summary.histogram('activations', activations)

                return activations, weights

        layer1, weights1 = nn_layer(x, length, neurons, 'layer1')

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropped = tf.nn.dropout(layer1, keep_prob)

        logits, weights2 = nn_layer(dropped, neurons, classes, 'layer2', activation=tf.identity)

        weights = tf.matmul(weights1, weights2)
        weights_shaped  = tf.reshape(weights, [dimx, dimy, channels, classes])
        weights_classes = tf.split(weights_shaped, classes, axis=3)

        for i in range(classes):

            weights_split    = weights_classes[i]
            weights_channels = tf.split(weights_split, channels, axis=2)

            for j in range(channels):

                image = tf.expand_dims(tf.squeeze(weights_channels[j], axis=3), axis=0)
                tf.summary.image('class{0}-channel{1}'.format(i, j), image)

        with tf.name_scope('loss'):

            with tf.name_scope('total'):
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits))

        tf.summary.scalar('loss', loss)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

        with tf.name_scope('accuracy'):

            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))

            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        tf.summary.scalar('accuracy', accuracy)

        merged          = tf.summary.merge_all()
        training_writer = tf.summary.FileWriter('{0}/training/'.format(path), session.graph)
        testing_writer  = tf.summary.FileWriter('{0}/testing/'.format(path))

        tf.global_variables_initializer().run()

        def feed_dict(train):

            if train:
                select  = choice(m, batch, replace=False)
                x_batch = reshape(training_data[select, :, :, :], (batch, length))
                y_batch = training_labels[select, :]
                k = 0.5

            else:
                x_batch = reshape(testing_data, (p, length))
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

    # -----------------------------------------------------------------------------------------------
    # MNIST
    # -----------------------------------------------------------------------------------------------

    # from imageio import imread

    # from os import listdir

    # path = '/home/al/compas_ml/data/mnist/'

    # training_data   = []
    # testing_data    = []
    # training_labels = []
    # testing_labels  = []

    # for i in ['testing', 'training']:
    #     for j in range(10):

    #         prefix = '{0}/{1}/{2}'.format(path, i, j)
    #         files  = listdir(prefix)

    #         for file in files:

    #             image = imread('{0}/{1}'.format(prefix, file))

    #             if i == 'training':
    #                 training_data.append(image)
    #                 training_labels.append(j)

    #             else:
    #                 testing_data.append(image)
    #                 testing_labels.append(j)

    # path = '/home/al/temp/'

    # pixel(training_data, training_labels, testing_data, testing_labels, classes=10, steps=200, batch=200, path=path,
    #       neurons=1024)

    # -----------------------------------------------------------------------------------------------
    # Odd Even
    # -----------------------------------------------------------------------------------------------

    from compas_ml.utilities import integers_from_csv
    from compas_ml.utilities import strings_from_csv

    from numpy import zeros

    from matplotlib import pyplot as plt


    folder = '/home/al/compas_ml/data/oddeven/'

    training_data_str = strings_from_csv(file='{0}training_data.csv'.format(folder))
    testing_data_str  = strings_from_csv(file='{0}testing_data.csv'.format(folder))
    training_labels   = integers_from_csv(file='{0}training_labels.csv'.format(folder))
    testing_labels    = integers_from_csv(file='{0}testing_labels.csv'.format(folder))

    m = len(training_data_str)
    p = len(testing_data_str)

    training_data = [0] * m
    testing_data  = [0] * p

    for i in range(m):

        text  = [j if j != '-' else '10' for j in training_data_str[i]]
        ints  = [int(j) for j in text]
        image = zeros((11, 10))

        for j in range(10):
            image[ints[j], j] = 1

        training_data[i] = image

    for i in range(p):

        text  = [j if j != '-' else '10' for j in testing_data_str[i]]
        ints  = [int(j) for j in text]
        image = zeros((11, 10))

        for j in range(10):
            image[ints[j], j] = 1

        testing_data[i] = image

    # plt.imshow(testing_data[60])
    # plt.show()

    path = '/home/al/temp/'

    # print(training_data[0].shape)

    pixel(training_data, training_labels, testing_data, testing_labels, classes=2, steps=100, batch=200, path=path,
          neurons=1024)
