
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from numpy import asarray
from numpy import float32
from numpy.random import choice


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'recurrent',
]


def recurrent(training_data, training_labels, testing_data, testing_labels, n_vectors, l_vector, classes, neurons,
              steps, batch):

    """ Recurrent Neural Network.

    Parameters
    ----------
    training_data : array
        Training data of size (m x dimx x dimy).
    training_labels : array
        Training labels of size (m x classes).
    testing_data : array
        Testing data of size (n x dimx x dimy).
    testing_labels : array
        Testing labels of size (n x classes).
    n_vectors : int
        Number of vectors in each sequence.
    l_vector : int
        Length of each vector.
    classes : int
        Number of classes.
    neurons : int
        Number of neurons.
    steps : int
        Number of analysis steps.
    batch : int
        Batch size of images per step.

    Returns
    -------
    None

    """

    print('***** Session started *****')

    training_data = asarray(training_data, dtype=float32)
    training_labels = asarray(training_labels, dtype=float32)

    x_ = tf.placeholder(tf.float32, shape=[None, n_vectors, l_vector])
    y_ = tf.placeholder(tf.float32, shape=[None, classes])
    m = training_data.shape[0]

    rnn_cell = tf.contrib.rnn.BasicRNNCell(neurons)
    outputs, _ = tf.nn.dynamic_rnn(rnn_cell, x_, dtype=tf.float32)

    Wl = tf.Variable(tf.truncated_normal([neurons, classes], mean=0, stddev=0.01))
    bl = tf.Variable(tf.truncated_normal([classes], mean=0, stddev=0.01))
    output = tf.matmul(outputs[:, -1, :], Wl) + bl

    diff = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_)
    cross_entropy = tf.reduce_mean(diff)
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(output, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        for i in range(steps):

            select = choice(m, batch, replace=False)
            x_batch = training_data[select, :, :]
            y_batch = training_labels[select, :]

            if (i + 1) % 10 == 0:
                acc = session.run(accuracy, feed_dict={x_: x_batch, y_: y_batch})
                print('Step: {0}: Accuracy: {1:.1f}'.format(i + 1, 100 * acc))

            session.run(train_step, feed_dict={x_: x_batch, y_: y_batch})

        acc = session.run(accuracy, feed_dict={x_: testing_data, y_: testing_labels})
        print('Testing accuracy: {1:.1f}'.format(i + 1, 100 * acc))

        print('***** Session finished *****')


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    from numpy import array
    from os import listdir
    from scipy.misc import imread

    path = '/home/al/compas_ml/data/mnist/'

    training_data   = []
    testing_data    = []
    training_labels = []
    testing_labels  = []

    for j in ['testing', 'training']:
        for i in range(10):
            files = listdir('{0}/{1}/{2}'.format(path, j, i))
            for file in files:
                image = imread('{0}/{1}/{2}/{3}'.format(path, j, i, file))
                dimx, dimy = image.shape
                binary = [0] * 10
                binary[i] = 1
                if j == 'training':
                    training_data.append(image)
                    training_labels.append(binary)
                else:
                    testing_data.append(image)
                    testing_labels.append(binary)

    training_data = array(training_data)[:, :, :]
    testing_data = array(testing_data)[:, :, :]

    recurrent(training_data, training_labels, testing_data, testing_labels, n_vectors=dimx, l_vector=dimy,
              neurons=500, classes=10, steps=3000, batch=300)
