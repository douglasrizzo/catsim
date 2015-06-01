import numpy as np
from scipy.spatial.distance import cdist


def manhattan(x, y=None):
    return pnorm(x, y, 1)


def euclidean(x, y=None):
    return pnorm(x, y, 2)


def chebyshev(x, y=None):
    return pnorm(x, y, float('inf'))


def minkowski(x, y=None, p=2):
    return pnorm(x, y, p)


def pnorm(x, y=None, p=2):
    """
    Calculates the p-norm for two sets of elements. The p-norm equals the
    Minkowski distance function and is given by:

    .. math:: ||x||_p = \\left( \\sum_{i=1}^n |x_i|^p  \\right)^{\\frac{1}{p}}

    For two vectors, the p-norm is given by:

    .. math:: ||x, y||_p = \\left( \\sum_{i=1}^n |x_i - y_i|^p  \\right)^{\\frac{1}{p}}

    In special cases where :math:`p=1`, :math:`p=2` or :math:`p=\\infty`, the
    p-norm is equal to the Manhattan, Euclidean and Chebyshev distances,
    respectively.
    """

    x = np.asarray(x)

    # arrays must have 2 dimensions, even if they are 1D-arrays
    if x.ndim == 1:
        np.reshape(x, [1, x.shape[0]])

    if y is None:
        y = x
    elif y.ndim == 1:
        np.reshape(y, [1, y.shape[0]])

    nfeatures = x.shape[1]
    y_features = y.shape[1]
    xpoints = x.shape[0]
    ypoints = y.shape[0]

    if nfeatures != y_features:
        raise ValueError(
            'x and y must have the same number of features. x =', nfeatures,
            'y =', y_features)

    D = np.zeros([xpoints, ypoints])

    for i in range(xpoints):
        for ii in range(ypoints):

            # special case when p = infinity
            if (p == float('inf')):
                D[i, ii] = max(np.absolute(x[i, iii] - y[ii, iii])
                               for iii in np.arange(nfeatures))

            else:
                D[i, ii] = np.power(
                    np.sum(np.power(np.absolute(x[i, iii] - y[ii, iii]), p)
                           for iii in np.arange(nfeatures)), 1 / p)

    return D


def mahalanobis(x, y=None):
    x = np.asarray(x)

    # arrays must have 2 dimensions, even if they are 1D-arrays
    if x.ndim == 1:
        np.reshape(x, [1, x.shape[0]])

    if y is None:
        y = x
    elif y.ndim == 1:
        np.reshape(y, [1, y.shape[0]])

    return cdist(x, y, metric='mahalanobis')
