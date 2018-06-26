
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'recurrent',
]


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


with tf.name_scope('rnn_weights'):
    with tf.name_scope('W_x'):
        Wx = tf.Variable(tf.zeros([element_size, hidden_layer_size]))
        variable_summaries(Wx)
    with tf.name_scope('W_h'):
        Wh = tf.Variable(tf.zeros([hidden_layer_size, hidden_layer_size]))
        variable_summaries(Wh)
    with tf.name_scope('Bias'):
        b_rnn = tf.Variable(tf.zeros([hidden_layer_size]))
        variable_summaries(b_rnn)


def rnn_step(previous_hidden_state, x):
    current_hidden_state = tf.tanh(tf.matmul(previous_hidden_state, Wh) + tf.matmul(x, Wx) + b_rnn)
    return current_hidden_state

processed_input = tf.transpose(_inputs, perm=[1, 0, 2])
# input size batch_size, time_steps, element_size
# output size time_steps, batch_size, element_size
initial_hidden = tf.zeros([batch_size, hidden_layer_size])
all_hidden_states = tf.scan(rnn_step, processed_input, initializer=initial_hidden, name='states')


def recurrent():
    pass


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    element_size = 28  # length of each vector in the sequence
    time_steps = 28  # number of vectors in the sequence
    num_classes = 10
    batch_size = 128
    hidden_layer_size = 128

    LOG_DIR = '/home/al/temp/rnn/'

    _inputs = tf.placeholder(tf.float32, shape=[None, time_steps, element_size], name='inputs')
    y = tf.placeholder(tf.float32, shape=[None, num_classes], name='labels')
