
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


__author__    = ['Andrew Liew <liew@arch.ethz.ch>']
__copyright__ = 'Copyright 2018, BLOCK Research Group - ETH Zurich'
__license__   = 'MIT License'
__email__     = 'liew@arch.ethz.ch'


__all__ = [
    'classes_to_onehot',
]


def classes_to_onehot(classes, length):

    """ Convert a list of classes (integers) into one-hot format.

    Parameters
    ----------
    classes : list
        A list of integer classes, counting from 0.
    length : int
        Number of class integers.

    Returns
    -------
    list
        List of one-hot lists.

    """

    for c, i in enumerate(classes):
        onehot = [0] * length
        onehot[i] = 1
        classes[c] = onehot

    return classes


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":

    pass
