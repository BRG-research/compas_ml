
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Lambda
from keras.losses import mse
from keras.losses import binary_crossentropy
from keras.models import Model

import matplotlib.pyplot as plt

from numpy import array
from numpy import arange
from numpy import linspace
from numpy import round
from numpy import zeros

import argparse


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'autoencoder',
]


def autoencoder(training_data, training_labels, length, neurons, batch, epochs, latent_dim=2):

    # This is based on the Keras blog post on Variational Autoencoders
    # https://blog.keras.io/building-autoencoders-in-keras.html

    inputs   = Input(shape=(length, ), name='encoder_input')
    x        = Dense(neurons, activation='relu')(inputs)
    z_mean   = Dense(latent_dim, name='z_mean')(x)
    z_logvar = Dense(latent_dim, name='z_logvar')(x)
    z        = Lambda(sampling, output_shape=(latent_dim, ), name='z')([z_mean, z_logvar])

    encoder = Model(inputs, [z_mean, z_logvar, z], name='encoder')

    latent_inputs = Input(shape=(latent_dim, ), name='z_sampling')
    x = Dense(neurons, activation='relu')(latent_inputs)
    outputs = Dense(length, activation='sigmoid')(x)

    decoder = Model(latent_inputs, outputs, name='decoder')

    outputs = decoder(encoder(inputs)[2])
    vae = Model(inputs, outputs, name='vae_mlp')

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights')
    parser.add_argument('-m', '--mse', action='store_true')
    args = parser.parse_args()

    models = (encoder, decoder)
    data = (training_data, training_labels)

    if args.mse:
        reconstruction_loss = mse(inputs, outputs)
    else:
        reconstruction_loss = binary_crossentropy(inputs, outputs)
    reconstruction_loss *= length

    kl_loss = 1 + z_logvar - backend.square(z_mean) - backend.exp(z_logvar)
    kl_loss = backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')

    if args.weights:
        vae = vae.load_weights(args.weights)
    else:
        vae.fit(training_data, epochs=epochs, batch_size=batch, validation_data=(training_data, None))

    plot_results(models, data, batch_size=batch)


def sampling(args):

    z_mean, z_logvar = args

    batch   = backend.shape(z_mean)[0]
    dim     = backend.int_shape(z_mean)[1]
    epsilon = backend.random_normal(shape=(batch, dim))

    return z_mean + backend.exp(0.5 * z_logvar) * epsilon


def plot_results(models, data, batch_size):

    encoder, decoder = models
    x, y = data
    z_mean, _, _ = encoder.predict(x, batch_size=batch_size)

    plt.figure(figsize=(10, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()

    n = 50
    digit_size = 28

    figure = zeros((digit_size * n, digit_size * n))
    grid_x = linspace(-4, 4, n)
    grid_y = linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size, j * digit_size: (j + 1) * digit_size] = digit

    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = arange(start_range, end_range, digit_size)
    sample_range_x = round(grid_x, 1)
    sample_range_y = round(grid_y, 1)

    plt.figure(figsize=(10, 10))
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.show()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    # ------------------------------------------------------------------------------
    # MNIST
    # ------------------------------------------------------------------------------

    from numpy import float32
    from numpy import reshape

    from scipy.misc import imread

    from os import listdir

    folder = '/home/al/compas_ml/data/mnist/'

    training_data   = []
    testing_data    = []
    training_labels = []
    testing_labels  = []

    for i in ['testing', 'training']:
        for j in range(10):

            prefix = '{0}/{1}/{2}'.format(folder, i, j)
            files  = listdir(prefix)

            for file in files:

                image = imread('{0}/{1}'.format(prefix, file))
                if i == 'training':
                    training_data.append(image)
                    training_labels.append(j)
                else:
                    testing_data.append(image)
                    testing_labels.append(j)

    training_data = array(training_data, dtype=float32) / 255
    testing_data  = array(testing_data, dtype=float32) / 255

    m, dimx, dimy = training_data.shape
    length = dimx * dimy

    training_data  = reshape(training_data, [-1, length])
    testing_data   = reshape(testing_data, [-1, length])
    testing_labels = testing_labels

    autoencoder(training_data, training_labels, length=length, neurons=1000, batch=500, epochs=100)
