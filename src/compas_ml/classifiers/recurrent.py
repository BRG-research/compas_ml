
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from compas_ml.helpers import labels_to_onehot

from numpy.random import shuffle

import tensorflow as tf


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'recurrent',
]


def recurrent(training_data, training_labels, testing_data, testing_labels, classes, neurons, steps, batch,
              dimension=64):

    """ Recurrent Neural Network.

    Parameters
    ----------
    training_data : list
        Training data of length m, each entry is a string sequence of length n.
    training_labels : list
        Training labels of length m.
    testing_data : list
        Testing data of length p, each entry is a string sequence of length n.
    testing_labels : list
        Testing labels of length p.
    classes : int
        Number of classes.
    neurons : int
        Number of neurons.
    steps : int
        Number of analysis steps.
    batch : int
        Batch size.
    dimension : int
        Dimension.

    Returns
    -------
    None

    """

    print('***** Session started *****')

    word_index = {}
    index = 0

    for sequence in training_data:
        for word in sequence.split():
            if word not in word_index:
                word_index[word] = index
                index += 1

    index_word = {index: word for word, index in word_index.items()}
    size = len(index_word)
    m = len(training_data)
    n = len(training_data[0].split())
    p = len(testing_data)

    x = tf.placeholder(tf.int32, shape=[None, n])
    y = tf.placeholder(tf.float32, shape=[None, classes])
    z = tf.placeholder(tf.int32, shape=[None])

    training_labels = labels_to_onehot(labels=training_labels, classes=2)
    testing_labels  = labels_to_onehot(labels=testing_labels, classes=2)

    training_lengths = []
    testing_lengths  = []

    for i in training_data:
        line = i.split()
        if '-' not in line:
            length = n
        else:
            length = line.index('-')
        training_lengths.append(length)

    for i in testing_data:
        line = i.split()
        if '-' not in line:
            length = n
        else:
            length = line.index('-')
        testing_lengths.append(length)


    def make_batch(batch_size, data, labels, lengths):

        indices = list(range(len(data)))
        shuffle(indices)
        batch = indices[:batch_size]
        x_ = [[word_index[word] for word in data[i].split()] for i in batch]
        y_ = [labels[i] for i in batch]
        z_ = [lengths[i] for i in batch]

        return x_, y_, z_


    with tf.name_scope('embeddings'):

        embeddings = tf.Variable(tf.random_uniform([size, dimension], -1.0, 1.0), name='embedding')
        embed = tf.nn.embedding_lookup(embeddings, x)

    with tf.variable_scope('bigru'):

        with tf.variable_scope('forward'):
            gru_fw_cell = tf.contrib.rnn.GRUCell(neurons)
            gru_fw_cell = tf.contrib.rnn.DropoutWrapper(gru_fw_cell)

        with tf.variable_scope('backward'):
            gru_bw_cell = tf.contrib.rnn.GRUCell(neurons)
            gru_bw_cell = tf.contrib.rnn.DropoutWrapper(gru_bw_cell)

        outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=gru_fw_cell, cell_bw=gru_bw_cell, inputs=embed,
                                                          sequence_length=z, dtype=tf.float32, scope='bigru')
        states = tf.concat(values=states, axis=1)

    weights = tf.Variable(tf.truncated_normal([2 * neurons, classes], mean=0, stddev=0.01))
    bias    = tf.Variable(tf.truncated_normal([classes], mean=0, stddev=0.01))
    # logits  = tf.matmul(states[1], weights + bias)
    logits  = tf.matmul(states, weights + bias)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(loss)

    correct  = tf.equal(tf.argmax(y, 1), tf.argmax(logits, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    with tf.Session() as session:

        session.run(tf.global_variables_initializer())

        for i in range(steps):

            x_batch, y_batch, z_batch = make_batch(batch, training_data, training_labels, training_lengths)

            if (i + 1) % 10 == 0:
                acc = session.run(accuracy, feed_dict={x: x_batch, y: y_batch, z: z_batch})
                print('Step: {0}: Accuracy: {1:.1f}'.format(i + 1, 100 * acc))

            session.run(train_step, feed_dict={x: x_batch, y: y_batch, z: z_batch})

        x_test, y_test, z_test = make_batch(p, testing_data, testing_labels, testing_lengths)
        acc = session.run(accuracy, feed_dict={x: x_test, y: y_test, z: z_test})
        print('Accuracy: {1:.1f}'.format(i + 1, 100 * acc))

    print('***** Session finished *****')


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    import pandas

    path = '/home/al/compas_ml/data/oddeven/'

    training_data    = list(dict(pandas.read_csv('{0}training_data.csv'.format(path))['sequence']).values())
    testing_data     = list(dict(pandas.read_csv('{0}testing_data.csv'.format(path))['sequence']).values())
    training_labels  = list(dict(pandas.read_csv('{0}training_labels.csv'.format(path))['label']).values())
    testing_labels   = list(dict(pandas.read_csv('{0}testing_labels.csv'.format(path))['label']).values())

    recurrent(training_data, training_labels, testing_data, testing_labels, steps=100, batch=100, classes=2,
              neurons=500)
