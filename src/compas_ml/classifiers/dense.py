
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
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    return dataset.shuffle(1000).repeat().batch(batch)


def evaluate(features, labels, batch):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)
    dataset = tf.data.Dataset.from_tensor_slices(inputs)
    return dataset.batch(batch)


def model(features, labels, mode, params):

    # Network

    network = tf.feature_column.input_layer(features, params['columns'])
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

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions, name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # Evaluation

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # Training

    assert mode == tf.estimator.ModeKeys.TRAIN
    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def dense(training_data, training_labels, testing_data, testing_labels, classes, batch, steps, path, neurons=[10, 10],
          type='float'):

    # Features

    features = []
    for key in training_data.keys():
        if type == 'float':
            features.append(tf.feature_column.numeric_column(key=key))
        elif type == 'binary':
            cat_feature = tf.feature_column.categorical_column_with_identity(key=key, num_buckets=classes)
            ind_feature = tf.feature_column.indicator_column(cat_feature)
            features.append(ind_feature)

    # Train

    params = {
        'columns': features,
        'neurons': neurons,
        'classes': classes,
    }
    classifier = tf.estimator.Estimator(model_fn=model, params=params, model_dir=path)
    classifier.train(input_fn=lambda: train(training_data, training_labels, batch=batch), steps=steps)
    classifier.evaluate(input_fn=lambda: evaluate(testing_data, testing_labels, batch=batch))

    # Predict

    expected = list(testing_labels.values.ravel())
    predict  = {i: list(j) for i, j in dict(testing_data).items()}
    predictions = classifier.predict(input_fn=lambda: evaluate(predict, labels=None, batch=batch))

    c, d = 0, 0
    for prediction, expect in zip(predictions, expected):
        class_id = prediction['class_ids'][0]
        probability = prediction['probabilities'][class_id] * 100
        print('Testing {0} - Predicted:Actual {1}:{3} - Probability: {2:.1f}'.format(c, class_id, probability, expect))
        if class_id == expect:
            d += 1
        c += 1
    print('Final accuracy: {0:.1f}'.format(100 * d / c))


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    import pandas as pd


    # Load data

    path = '/home/al/compas_ml/data/iris/'

    columns         = list(pd.read_csv('{0}columns.csv'.format(path)).keys())
    training_data   = pd.read_csv('{0}training_data.csv'.format(path), names=columns, header=0)
    testing_data    = pd.read_csv('{0}testing_data.csv'.format(path), names=columns, header=0)
    training_labels = pd.read_csv('{0}training_labels.csv'.format(path), names=['val'], header=0)
    testing_labels  = pd.read_csv('{0}testing_labels.csv'.format(path), names=['val'], header=0)

    dense(training_data, training_labels, testing_data, testing_labels, classes=3, batch=10, steps=1000,
          path='/home/al/temp/dnn/', neurons=[10, 10], type='float')
