"""Miscellaneous statistical functions"""

import numpy


def coef_variation(x: numpy.ndarray, axis: int = 0) -> numpy.ndarray:
    """Calculates the coefficientof variation of the rows or columns of a matrix.
    The coefficient of variation is given by the standard deviation divided by the mean of a variable:

    .. math:: \\frac{\\sigma}{\\mu}

    :param x: the data matrix
    :param axis: `0` to calculate for columns, `1` for rows
    :returns: a vector containing the coefficient of variations along the chosen axis
    """
    if not isinstance(x, numpy.matrix):
        x = numpy.asarray(x)

    mean = numpy.mean(x, axis=axis)
    stddev = numpy.std(x, axis=axis)

    # print('Means:', mean)
    # print('Std. Devs:', stddev)

    result = stddev / mean if axis == 0 else numpy.transpose(stddev) / numpy.transpose(mean)

    return result


def coef_correlation(x: numpy.ndarray):
    cov = covariance(x, False)
    stddev = numpy.std(x, axis=0)
    n_obs, n_features = x.shape

    corr = numpy.zeros([n_features, n_features])

    for i in range(n_features):
        for ii in range(i, n_features):
            corr[i, ii] = corr[ii, i] = cov[i, ii] / (stddev[i] * stddev[ii])

    return corr


def covariance(x: numpy.ndarray, minus_one: bool = True):
    """Calculates the covariance matrix of another matrix

    :param x: a data matrix
    :param minus_one: subtract one from the total number of observations

    >>> from sklearn.datasets import load_iris
    >>> x = load_iris()['data']
    >>> print(numpy.array_equal(covariance(x), numpy.cov(x.T)))
    True
    """
    x_means = numpy.mean(x, axis=0)
    n_obs, n_features = x.shape

    covars = numpy.zeros([n_features, n_features])

    for i in range(n_features):
        for ii in range(i, n_features):
            _sum = 0
            for iii in range(n_obs):
                _sum += (x[iii, i] - x_means[i]) * (x[iii, ii] - x_means[ii])
            covars[i, ii] = covars[ii, i] = _sum / ((n_obs - 1) if minus_one else n_obs)

    return covars


def bincount(x):
    """Count the number of occurrences from each integer in a list or 1-D :py:class:`numpy.ndarray`.
    If there are gaps between the numbers, then the numbers in those gaps are given a 0 value of occurrences.

    >>> bincount(numpy.array([-4, 0, 1, 1, 3, 2, 1, 5]))
    array([1, 0, 0, 0, 1, 3, 1, 1, 0, 1], dtype=int32)
    """
    x_max = numpy.max(x)
    x_min = numpy.min(x)
    size = abs(x_max) + abs(x_min) + 1

    count = numpy.zeros(size, dtype=numpy.int32)

    for i in x:
        count[i + abs(x_min)] += 1

    return count


def scatter_matrix(data: numpy.ndarray) -> numpy.ndarray:
    """Calculates the scatter matrix of a data matrix. The scatter matrix is an unnormalized
    version of the covariance matrix, in which the means of the observation values are subtracted.

    The calculations done by this function follow the following equation:

    .. math:: S=\\sum_{{j=1}}^{n}({\\mathbf{x}}_{j}-\\overline
              {{\\mathbf{x}}})({\\mathbf{x}}_{j}-\\overline {{\\mathbf{x}}})^{T}=\\sum
              _{{j=1}}^{n}({\\mathbf{x}}_{j}-\\overline
              {{\\mathbf{x}}})\\otimes({\\mathbf{x}}_{j}-\\overline{{\\mathbf{x}}})=\\left(\\sum
              _{{j=1}}^{n}{\\mathbf {x}}_{j}{\\mathbf {x}}_{j}^{T}\\right)-n\\overline
              {{\\mathbf {x}}}\\overline {{\\mathbf {x}}}^{T}

    :param data: the data matrix
    :returns: the scatter matrix of the given data matrix
    """
    mean_vector = numpy.mean(data, axis=0)
    scatter = numpy.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[0]):
        scatter += (data[i, :].reshape(data.shape[1], 1) - mean_vector).dot(
            (data[i, :].reshape(data.shape[1], 1) - mean_vector).T
        )

    return scatter


if __name__ == '__main__':
    import doctest

    doctest.testmod()
