
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
    'softmax_regression',
]


def softmax_regression(training_data, training_labels, testing_data, testing_labels, classes, length, steps, batch):

    """ Softmax Regression Classifier.

    Parameters
    ----------
    training_data : array
        Training data of size (m x length).
    training_labels : array
        Training labels of size (m x classes).
    testing_data : array
        Testing data of size (n x length).
    testing_labels : array
        Testing labels of size (n x classes).
    classes : int
        Number of classes.
    length : int
        Number of pixels per image.
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

    x = tf.placeholder(tf.float32, [None, length])
    W = tf.Variable(tf.zeros([length, classes]))
    m = training_data.shape[0]

    y_true = tf.placeholder(tf.float32, [None, classes])
    y_pred = tf.matmul(x, W)

    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    cross_entropy = tf.reduce_mean(diff)
    gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        for i in range(steps):

            select = choice(m, batch, replace=False)
            x_batch = training_data[select, :]
            y_batch = training_labels[select, :]

            session.run(gd_step, feed_dict={x: x_batch, y_true: y_batch})

            if (i + 1) % 10 == 0:
                print('Step: {0}'.format(i + 1))

        acc = session.run(accuracy, feed_dict={x: testing_data, y_true: testing_labels})
        print('Accuracy: {0:.1f} %'.format(100 * acc))

    print('***** Session finished *****')


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    from matplotlib import pyplot as plt
    from numpy import array
    from scipy.misc import imread
    from os import listdir


    path = '/home/al/compas_ml/data/mnist/'

    training_data   = []
    testing_data    = []
    training_labels = []
    testing_labels  = []
    length = 0

    for j in ['testing', 'training']:
        for i in range(10):
            files = listdir('{0}/{1}/{2}'.format(path, j, i))
            for file in files:
                image = imread('{0}/{1}/{2}/{3}'.format(path, j, i, file))
                if not length:
                    if len(image.shape) == 2:
                        dimx, dimy = image.shape
                        length = dimx * dimy
                    else:
                        dimx, dimy, dimz = image.shape
                        length = dimx * dimy * dimz
                binary = [0] * 10
                binary[i] = 1
                if j == 'training':
                    training_data.append(image.reshape(length))
                    training_labels.append(binary)
                else:
                    testing_data.append(image.reshape(length))
                    testing_labels.append(binary)

    # plt.imshow(array(training_data).transpose())
    # plt.show()

    softmax_regression(training_data, training_labels, testing_data, testing_labels,
                       classes=10, length=length, steps=1000, batch=300)
