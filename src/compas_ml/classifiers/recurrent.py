
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

# from numpy import asarray
# from numpy import float32
# from numpy.random import choice
from numpy.random import shuffle


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'recurrent',
]


def recurrent(training_data, training_labels, testing_data, testing_labels, lengths, classes, neurons, steps,
              batch):

    """ Recurrent Neural Network.

    Parameters
    ----------
    training_data : array
        Training data of size (m x n), each entry is a string sequence.
    # training_labels : array
    #     Training labels of size (m x classes).
    # testing_data : array
    #     Testing data of size (n x dimx x dimy).
    # testing_labels : array
    #     Testing labels of size (n x classes).
    # lengths
    classes : int
        Number of classes.
    # neurons : int
    #     Number of neurons.
    steps : int
        Number of analysis steps.
    batch : int
        Batch size.

    Returns
    -------
    None

    """

    print('***** Session started *****')

    m = training_data.shape[0]

    word_index = {}
    index = 0

    for i in range(m):
        for word in training_data[i].split():
            if word not in word_index:
                word_index[word] = index
                index += 1

    index_word = {index: word for word, index in word_index.items()}
    # size = len(index_word)

    print(index_word)
    # def make_sentence_batch(batch_size, data, labels, lengths):

    #     indices = list(range(len(data)))
    #     shuffle(indices)
    #     batch = indices[:batch_size]
    #     x = [[word_index[word] for word in data[i].split()] for i in batch]
    #     y = [labels[i] for i in batch]  # slice?
    #     seqlens = [lengths[i] for i in batch]  # slice?

    #     return x, y, seqlens

    # training_data = asarray(training_data, dtype=float32)
    # training_labels = asarray(training_labels, dtype=float32)

    x_ = tf.placeholder(tf.int32, shape=[batch, steps])
    y_ = tf.placeholder(tf.float32, shape=[batch, classes])
    z_ = tf.placeholder(tf.int32, shape=[batch])

    # rnn_cell = tf.contrib.rnn.BasicRNNCell(neurons)
    # outputs, _ = tf.nn.dynamic_rnn(rnn_cell, x_, dtype=tf.float32)

    # Wl = tf.Variable(tf.truncated_normal([neurons, classes], mean=0, stddev=0.01))
    # bl = tf.Variable(tf.truncated_normal([classes], mean=0, stddev=0.01))
    # output = tf.matmul(outputs[:, -1, :], Wl) + bl

    # diff = tf.nn.softmax_cross_entropy_with_logits(logits=output, labels=y_)
    # cross_entropy = tf.reduce_mean(diff)
    # train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

    # correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(output, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as session:

        # session.run(tf.global_variables_initializer())

        # for i in range(steps):

        #     select = choice(m, batch, replace=False)
        #     x_batch = training_data[select, :, :]
        #     y_batch = training_labels[select, :]

        #     if (i + 1) % 10 == 0:
        #         acc = session.run(accuracy, feed_dict={x_: x_batch, y_: y_batch})
        #         print('Step: {0}: Accuracy: {1:.1f}'.format(i + 1, 100 * acc))

        #     session.run(train_step, feed_dict={x_: x_batch, y_: y_batch})

        # acc = session.run(accuracy, feed_dict={x_: testing_data, y_: testing_labels})
        # print('Testing accuracy: {1:.1f}'.format(i + 1, 100 * acc))

        print('***** Session finished *****')


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    from compas_ml.helpers import classes_to_onehot

    from numpy import array

    from random import randint


    n_vectors = 1000
    l_vector = 10

    even = []
    odd  = []
    lengths = []

    for i in range(n_vectors // 2):
        even_ = []
        odd_  = []
        n = randint(5, 10)
        lengths.append(n)
        for j in range(n):
            even_.append(str(randint(0, 4) * 2))
            odd_.append(str(randint(0, 4) * 2 + 1))
        for j in range(l_vector - n):
            even_.append('-')
            odd_.append('-')
        even.append(' '.join(even_))
        odd.append(' '.join(odd_))

    data = array(even + odd)
    n = int(0.8 * n_vectors)
    training_data = data[:n]
    testing_data  = data[n:]

    labels = [0] * (n_vectors // 2) + [1] * (n_vectors // 2)
    labels = classes_to_onehot(classes=labels, length=2)
    training_labels = labels[:n]
    testing_labels  = labels[n:]

    lengths *= 2

    recurrent(training_data, training_labels, testing_data, testing_labels, lengths=lengths, neurons=500,
              classes=2, steps=500, batch=100)
