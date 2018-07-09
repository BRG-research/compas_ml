
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from numpy import asarray
from numpy import float32
from numpy import int32
from numpy.random import choice


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'softmax',
]


def softmax(training_data, training_labels, testing_data, testing_labels, classes, length, steps, batch, model_dir,
            channels, neurons):

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
    model_dir : str
        Model directory.
    neurons : int
        Number of neurons in hidden layer.

    Returns
    -------
    None

    """

    print('***** Session started *****')

    def train():

        session = tf.InteractiveSession()

        with tf.name_scope('input'):
            x_ = tf.placeholder(tf.float32, [None, length * channels], name='x-input')
            y_ = tf.placeholder(tf.float32, [None, classes], name='y-input')

        def weight_variable(shape):
            return tf.Variable(tf.truncated_normal(shape, stddev=0.01))

        def bias_variable(shape):
            return tf.Variable(tf.constant(0.1, shape=shape))

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

        def nn_layer(input, input_dim, output_dim, name, act=tf.nn.relu):
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
                activations = act(preactivate, name='activation')
                tf.summary.histogram('activations', activations)
                return activations, weights

        hidden1, weights1 = nn_layer(x_, lengths * channels, neurons, 'layer1')

        with tf.name_scope('dropout'):
            keep_prob = tf.placeholder(tf.float32)
            tf.summary.scalar('dropout_keep_probability', keep_prob)
            dropped = tf.nn.dropout(hidden1, keep_prob)

        y_pred, weights2 = nn_layer(dropped, neurons, classes, 'layer2', act=tf.identity)

    #     weights = tf.matmul(weights1, weights2)
    #     weights_shaped = tf.reshape(weights, [dims[0], dims[1], dims[2], classes])
    #     weights_good, weights_bad = tf.split(weights_shaped, 2, axis=3)
    #     good_r, good_g, good_b = tf.split(weights_good, 3, axis=2)
    #     bad_r, bad_g, bad_b = tf.split(weights_bad, 3, axis=2)
    #     good_r_ = tf.expand_dims(tf.squeeze(good_r, axis=3), 0)
    #     good_g_ = tf.expand_dims(tf.squeeze(good_g, axis=3), 0)
    #     good_b_ = tf.expand_dims(tf.squeeze(good_b, axis=3), 0)
    #     bad_r_ = tf.expand_dims(tf.squeeze(bad_r, axis=3), 0)
    #     bad_g_ = tf.expand_dims(tf.squeeze(bad_g, axis=3), 0)
    #     bad_b_ = tf.expand_dims(tf.squeeze(bad_b, axis=3), 0)
    #     tf.summary.image('good/compression', good_r_)
    #     tf.summary.image('good/symmetry', good_g_)
    #     tf.summary.image('good/independent', good_b_)
    #     tf.summary.image('bad/compression', bad_r_)
    #     tf.summary.image('bad/symmetry', bad_g_)
    #     tf.summary.image('bad/independent', bad_b_)

        with tf.name_scope('cross_entropy'):
            diff = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_pred)
            with tf.name_scope('total'):
                cross_entropy = tf.reduce_mean(diff)
        tf.summary.scalar('cross_entropy', cross_entropy)

        with tf.name_scope('train'):
            train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

        with tf.name_scope('accuracy'):
            with tf.name_scope('correct_prediction'):
                correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_, 1))
            with tf.name_scope('accuracy'):
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        merged = tf.summary.merge_all()
        training_writer = tf.summary.FileWriter(model_dir + '/training/', session.graph)
        testing_writer  = tf.summary.FileWriter(model_dir + '/testing/')
        tf.global_variables_initializer().run()

    #     def feed_dict(train):
    #         if train or FLAGS.fake_data:
    #             select = choice(train_data.shape[0], batch, replace=False)
    #             xs = reshape(train_data[select, :, :, :], (batch, dims[0] * dims[1] * dims[2]))
    #             ys = train_labels[select, :]
    #             k = 0.5
    #         else:
    #             xs = reshape(test_data, (test_data.shape[0], dims[0] * dims[1] * dims[2]))
    #             ys = test_labels
    #             k = 1.0
    #         return {x: xs, y_: ys, keep_prob: k}

    #     for i in range(FLAGS.max_steps):
    #         if i % 10 == 0:
    #             summary, acc = session.run([merged, accuracy], feed_dict=feed_dict(False))
    #             testing_writer.add_summary(summary, i)
    #             print('Accuracy at step %s: %s' % (i, acc))
    #         else:
    #             if i % 100 == 99:
    #                 run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    #                 run_metadata = tf.RunMetadata()
    #                 summary, _ = session.run([merged, train_step],
    #                                       feed_dict=feed_dict(True),
    #                                       options=run_options,
    #                                       run_metadata=run_metadata)
    #                 training_writer.add_run_metadata(run_metadata, 'step%03d' % i)
    #                 training_writer.add_summary(summary, i)
    #                 print('Adding run metadata for', i)
    #             else:
    #                 summary, _ = session.run([merged, train_step], feed_dict=feed_dict(True))
    #                 training_writer.add_summary(summary, i)
    #     training_writer.close()
    #     testing_writer.close()


    # def main(_):
    #     if tf.gfile.Exists(FLAGS.log_dir):
    #         tf.gfile.DeleteRecursively(FLAGS.log_dir)
    #     tf.gfile.MakeDirs(FLAGS.log_dir)
    #     train()

    # parser = argparse.ArgumentParser()
    # parser.add_argument('--fake_data', nargs='?', const=True, type=bool, default=False, help='Fake data for testing.')
    # parser.add_argument('--max_steps', type=int, default=steps, help='Number of steps for training.')
    # parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate')
    # parser.add_argument('--dropout', type=float, default=0.9, help='Keep probability for training dropout.')
    # parser.add_argument('--log_dir', type=str, default=model_dir, help='Summaries log directory')

    # FLAGS, unparsed = parser.parse_known_args()

    # tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)













    # training_data   = asarray(training_data, dtype=float32)
    # testing_data    = asarray(training_data, dtype=float32)
    # training_labels = asarray(training_labels, dtype=int32)
    # testing_labels  = asarray(training_labels, dtype=int32)

    # m = training_data.shape[0]

    # y_true = tf.placeholder(tf.float32, [None, classes])
    # y_pred = tf.matmul(x, W)

    # diff = tf.nn.softmax_cross_entropy_with_logits(logits=y_pred, labels=y_true)
    # cross_entropy = tf.reduce_mean(diff)
    # gd_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    # correct_mask = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_mask, tf.float32))

    # with tf.Session() as session:

    #     session.run(tf.global_variables_initializer())

    #     for i in range(steps):

    #         select = choice(m, batch, replace=False)
    #         x_batch = training_data[select, :]
    #         y_batch = training_labels[select, :]

    #         session.run(gd_step, feed_dict={x: x_batch, y_true: y_batch})

    #         if (i + 1) % 10 == 0:
    #             print('Step: {0}'.format(i + 1))

    #     acc = session.run(accuracy, feed_dict={x: testing_data, y_true: testing_labels})
    #     print('Accuracy: {0:.1f} %'.format(100 * acc))

    print('***** Session finished *****')


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    from compas_ml.helpers import classes_to_onehot

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
                    else:
                        dimx, dimy, dimz = image.shape
                    length = dimx * dimy
                if j == 'training':
                    training_data.append(image)
                    training_labels.append(i)
                else:
                    testing_data.append(image)
                    testing_labels.append(i)

    # plt.imshow(array(training_data).transpose())
    # plt.show()

    training_labels = classes_to_onehot(classes=training_labels, length=10)
    testing_labels  = classes_to_onehot(classes=testing_labels, length=10)

    model_dir = '/home/al/temp/tf/'

    softmax(training_data, training_labels, testing_data, testing_labels,
            classes=10, length=length, steps=1000, batch=300, model_dir=model_dir, channels=1, neurons=1024)
