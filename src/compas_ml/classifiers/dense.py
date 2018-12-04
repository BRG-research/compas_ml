
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'dense',
]


def train(features, labels, batch):

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))

    return dataset.shuffle(1000).repeat().batch(batch)


def evaluate(features, labels, batch):

    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    return dataset.batch(batch)


def model(features, labels, mode, params):

    # Network

    network = tf.feature_column.input_layer(features, params['features'])

    for units in params['neurons']:
        network = tf.layers.dense(network, units=units, activation=tf.nn.relu)

    logits = tf.layers.dense(network, params['classes'], activation=None)

    # Predictions

    predictions = tf.argmax(logits, 1)

    if mode == tf.estimator.ModeKeys.PREDICT:

        predictions = {
            'class_ids':     predictions[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits':        logits,
        }

        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Loss and metrics

    loss     = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions, name='accuracy')
    metrics  = {'accuracy': accuracy}

    tf.summary.scalar('accuracy', accuracy[1])

    # Evaluation

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Training

    assert mode == tf.estimator.ModeKeys.TRAIN

    learningRate = tf.train.exponential_decay(learning_rate=0.1, global_step=tf.train.get_global_step(),
                                              decay_steps=params['steps'], decay_rate=0.95, staircase=True)
    train_op = tf.train.GradientDescentOptimizer(learningRate).minimize(loss, tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def dense(training_data, training_labels, testing_data, testing_labels, batch, steps, features, classes, neurons,
          type, path):

    """ Dense Neural Network.

    Parameters
    ----------
    training_data : dict
        Training data as a column based dictionary.
    training_labels : list
        Training labels of length m.
    testing_data : dict
        Testing data as a column based dictionary.
    testing_labels : list
        Testing labels of length p.
    batch : int
        Batch size.
    steps : int
        Number of analysis steps.
    features : list
        List of feature string names.
    classes : int
        Number of classes.
    neurons : int
        Number of neurons.
    type : str
        Type of data for features: 'float' or 'binary'.
    path : str
        Path to save log files.

    Returns
    -------
    None

    """

    print('***** Session started *****')

    # Features

    columns = []

    for key in features:

        if type == 'float':
            columns.append(tf.feature_column.numeric_column(key=key))

        elif type == 'binary':
            cat_feature = tf.feature_column.categorical_column_with_identity(key=key, num_buckets=2)
            ind_feature = tf.feature_column.indicator_column(cat_feature)
            columns.append(ind_feature)

    # Train

    params = {'features': columns, 'neurons': neurons, 'classes': classes, 'steps': steps}

    classifier = tf.estimator.Estimator(model_fn=model, params=params, model_dir=path)
    classifier.train(input_fn=lambda: train(training_data, training_labels, batch=batch), steps=steps)
    classifier.evaluate(input_fn=lambda: evaluate(testing_data, testing_labels, batch=batch))

    # Predict

    expected    = testing_labels
    predict     = {i: list(j) for i, j in dict(testing_data).items()}
    predictions = classifier.predict(input_fn=lambda: evaluate(predict, labels=None, batch=batch))

    correct = 0
    total   = 0

    for prediction, expect in zip(predictions, expected):

        class_id    = prediction['class_ids'][0]
        probability = prediction['probabilities'][class_id] * 100

        if class_id == expect:
            correct += 1

        print('Predicted:Actual = {0}:{1} @ {2:.1f}%'.format(class_id, expect, probability))

        total += 1

    print('Accuracy: {0:.1f}\%'.format(100 * correct / total))

    print('***** Session finished *****')


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    from compas_ml.utilities import strings_from_csv
    from compas_ml.utilities import dict_from_csv
    from compas_ml.utilities import integers_from_csv


    path = '/home/al/compas_ml/data/iris/'

    features        = strings_from_csv(file='{0}features.csv'.format(path))
    training_data   = dict_from_csv(file='{0}training_data.csv'.format(path), headers=features)
    testing_data    = dict_from_csv(file='{0}testing_data.csv'.format(path), headers=features)
    training_labels = integers_from_csv(file='{0}training_labels.csv'.format(path))
    testing_labels  = integers_from_csv(file='{0}testing_labels.csv'.format(path))

    dense(training_data, training_labels, testing_data, testing_labels, batch=10, steps=5000, features=features,
          classes=3, neurons=[20, 20], type='float', path='/home/al/temp/dnn/')
