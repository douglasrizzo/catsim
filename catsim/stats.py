"""A bunch of statistical functions that I implemented just to show that I
knew how they worked.
"""

import numpy as np


def coefvariation(x, axis=0):
    '''..math:: \\frac{\\sigma}{\\mu}'''
    if not isinstance(x, np.matrix):
        x = np.asarray(x)

    mean = np.mean(x, axis=axis)
    stddev = np.std(x, axis=axis)

    # print('Means:', mean)
    # print('Std. Devs:', stddev)

    result = stddev / \
        mean if axis == 0 else np.transpose(stddev) / np.transpose(mean)

    return result


def coefCorrelation(x):
    cov = covariance(x, False)
    stddev = np.std(x, axis=0)
    n_obs, n_features = x.shape

    corr = np.zeros([n_features, n_features])

    for i in range(n_features):
        for ii in range(i, n_features):
            corr[i, ii] = corr[ii, i] = cov[i, ii] / (stddev[i] * stddev[ii])

    return corr


def covariance(x: np.ndarray, minus_one: bool=True):
    """Calculates the covariance matrix of another matrix

    :param x: a data matrix
    :param minus_one: subtract one from the total number of observations
    >>> from sklearn.datasets import load_iris
    >>> x = load_iris()['data']
    >>> print(np.array_equal(covariance(x), np.cov(x.T)))
    True
    """
    x_means = np.mean(x, axis=0)
    n_obs, n_features = x.shape

    covars = np.zeros([n_features, n_features])

    for i in range(n_features):
        for ii in range(i, n_features):
            sum = 0
            for iii in range(n_obs):
                sum += (x[iii, i] - x_means[i]) * (x[iii, ii] - x_means[ii])
            covars[i, ii] = covars[ii, i] = sum / \
                ((n_obs - 1) if minus_one else n_obs)

    return covars


def bincount(x):
    """Count the number of occurrences from each integer in a list or 1-D :py:type:`np.ndarray`.
    If there are gaps between the numbers, then the numbers in those gaps are given a 0 value of occurrences.
    >>> bincount(np.array([-4, 0, 1, 1, 3, 2, 1, 5]))
    array([1, 0, 0, 0, 1, 3, 1, 1, 0, 1], dtype=int32)
    """
    x_max = np.max(x)
    x_min = np.min(x)
    size = abs(x_max) + abs(x_min) + 1

    count = np.zeros(size, dtype=np.int32)

    for i in x:
        count[i + abs(x_min)] += 1

    return count


def scatter_matrix(data: np.ndarray) -> np.ndarray:
    """Calculates the scatter matrix of a data matrix. The scatter matrix is an unnormalized
    version of the covariance matrix, in which the means of the observation values are subtracted.

    The calculations done by this function follow the following equation:

    .. math:: S=\\sum_{{j=1}}^{n}({\\mathbf{x}}_{j}-\\overline {{\\mathbf{x}}})({\\mathbf{x}}_{j}-\\overline {{\\mathbf{x}}})^{T}=\\sum _{{j=1}}^{n}({\\mathbf{x}}_{j}-\\overline {{\\mathbf{x}}})\\otimes({\\mathbf{x}}_{j}-\\overline{{\\mathbf{x}}})=\\left(\\sum _{{j=1}}^{n}{\\mathbf {x}}_{j}{\\mathbf {x}}_{j}^{T}\\right)-n\\overline {{\\mathbf {x}}}\\overline {{\\mathbf {x}}}^{T}

    :param data: the data matrix
    :returns: the scatter matrix of the given data matrix
    """
    mean_vector = np.mean(data, axis=0)
    scatter = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[0]):
        scatter += (
            data[i, :].reshape(data.shape[1], 1) - mean_vector
        ).dot((data[i, :].reshape(data.shape[1], 1) - mean_vector).T)

    return scatter


if __name__ == '__main__':
    import doctest
    doctest.testmod()
