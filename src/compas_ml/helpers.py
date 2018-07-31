
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from matplotlib import pyplot as plt

from numpy import array
from numpy import float32
from numpy import zeros

from scipy.misc import imread


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'labels_to_onehot',
    'colour_weights',
]


def labels_to_onehot(labels, classes):

    """ Convert a list of labels (integers) into one-hot format.

    Parameters
    ----------
    labels : list
        A list of integer labels, counting from 0.
    classes : int
        Number of class integers.

    Returns
    -------
    list
        List of one-hot lists.

    """

    for c, i in enumerate(labels):
        onehot = [0] * classes
        onehot[i] = 1
        labels[c] = onehot

    return labels


def colour_weights(path):

    """ Colour a weights image to show positive (blue) and negative (red) areas.

    Parameters
    ----------
    path : str
        Path of weights image to load.

    Returns
    -------
    None

    """

    image = array(imread(path), dtype=float32) / 255
    dimx, dimy, dimz = image.shape

    image_col = zeros([dimx, dimy, dimz])
    image_col[:, :, 3] = 1
    for i in range(dimx):
        for j in range(dimy):
            if image[i, j, 1] < 0.5:
                val = (0.5 - image[i, j, 1]) * 2
                image_col[i, j, 0] = val
            elif image[i, j, 1] > 0.5:
                val = (image[i, j, 1] - 0.5) * 2
                image_col[i, j, 2] = val

    plt.imshow(image_col)
    plt.show()


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    colour_weights(path='/home/al/downloads/ss.png')
