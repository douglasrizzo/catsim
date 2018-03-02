"""Functions used specifically during the application/simulation of computerized adaptive tests."""

import operator

import numpy
import random

from catsim import irt


def dodd(theta: float, items: numpy.ndarray, correct: bool) -> float:
    """Method proposed by [Dod90]_ for the reestimation of :math:`\\hat{\\theta}`
    when the response vector is composed entirely of 1s or 0s.

    .. math::

        \\hat{\\theta}_{t+1} = \\left\\lbrace \\begin{array}{ll}
        \\hat{\\theta}_t+\\frac{b_{max}-\\hat{\\theta_t}}{2} & \\text{if } X_t = 1 \\\\
        \\hat{\\theta}_t-\\frac{\\hat{\\theta}_t-b_{min}}{2} & \\text{if }  X_t = 0
        \\end{array} \\right\\rbrace

    :param theta: the initial proficiency level
    :param items: a numpy array containing the parameters of the items in the
                  database. This is necessary to capture the maximum and minimum
                  difficulty levels necessary for the method.
    :param correct: a boolean value informing whether or not the examinee
                    correctly answered the current item.
    :returns: a new estimation for :math:`\\theta`
    """
    b = items[:, 1]

    return theta + ((max(b) - theta) / 2) if correct else theta - ((theta - min(b)) / 2)


def bias(actual: list, predicted: list):
    """Calculates the test bias, an evaluation criterion for computerized adaptive test methodolgies [Chang2001]_.
    The value is calculated by:

    .. math:: Bias = \\frac{\\sum_{i=1}^{N} (\\hat{\\theta}_i - \\theta_{i})}{N}

    where :math:`\\hat{\\theta}_i` is examinee :math:`i` estimated proficiency and
    :math:`\\hat{\\theta}_i` is examinee :math:`i` actual proficiency.

    :param actual: a list or 1-D numpy array containing the true proficiency values
    :param predicted: a list or 1-D numpy array containing the estimated proficiency values
    :returns: the bias between the predicted values and actual values.
    """
    if len(actual) != len(predicted):
        raise ValueError('actual and predicted vectors need to be the same size')
    return numpy.mean(list(map(operator.sub, predicted, actual)))


def mse(actual: list, predicted: list):
    """Mean squared error, a value used when measuring the precision
    with which a computerized adaptive test estimates examinees proficiencies [Chang2001]_.
    The value is calculated by:

    .. math:: MSE = \\frac{\\sum_{i=1}^{N} (\\hat{\\theta}_i - \\theta_{i})^2}{N}

    where :math:`\\hat{\\theta}_i` is examinee :math:`i` estimated proficiency and
    :math:`\\hat{\\theta}_i` is examinee :math:`i` actual proficiency.

    :param actual: a list or 1-D numpy array containing the true proficiency values
    :param predicted: a list or 1-D numpy array containing the estimated proficiency values
    :returns: the mean squared error between the predicted values and actual values.
    """
    if len(actual) != len(predicted):
        raise ValueError('actual and predicted vectors need to be the same size')
    return numpy.mean([x * x for x in list(map(operator.sub, predicted, actual))])


def rmse(actual: list, predicted: list):
    """Root mean squared error, a common value used when measuring the precision
    with which a computerized adaptive test estimates examinees proficiencies [Bar10]_.
    The value is calculated by:

    .. math:: RMSE = \\sqrt{\\frac{\\sum_{i=1}^{N} (\\hat{\\theta}_i - \\theta_{i})^2}{N}}

    where :math:`\\hat{\\theta}_i` is examinee :math:`i` estimated proficiency and
    :math:`\\hat{\\theta}_i` is examinee :math:`i` actual proficiency.

    :param actual: a list or 1-D numpy array containing the true proficiency values
    :param predicted: a list or 1-D numpy array containing the estimated proficiency values
    :returns: the root mean squared error between the predicted values and actual values.
    """
    if len(actual) != len(predicted):
        raise ValueError('actual and predicted vectors need to be the same size')
    return numpy.sqrt(mse(actual, predicted))


def overlap_rate(items: numpy.ndarray, test_size: int) -> float:
    """Test overlap rate, an average measure of how much of the test two examinees take is equal [Bar10]_. It is given by:

    .. math:: T=\\frac{N}{Q}S_{r}^2 + \\frac{Q}{N}

    If, for example :math:`T = 0.5`, it means that the tests of two random examinees have 50% of equal items.

    :param items: a matrix containing, in the 4th column, the number of
                  times each item was used in the tests.
    :param test_size: an integer informing the number of items in a test.
    :returns: test overlap rate.
    """

    bank_size = items.shape[0]
    var_r = numpy.var(items[:, 4])

    t = (bank_size / test_size) * var_r + (test_size / bank_size)

    return t


def generate_item_bank(n: int, itemtype: str = '4PL', corr: float = 0):
    """Generate a synthetic item bank whose parameters approximately follow
    real-world parameters, as proposed by [Bar10]_.

    Item parameters are extracted from the following probability distributions:

    * discrimination: :math:`N(1.2, 0.25)`

    * difficulty: :math:`N(0,  1)`

    * pseudo-guessing: :math:`N(0.25, 0.02)`

    * upper asymptote: :math:`U(0.94, 1)`

    :param n: how many items are to be generated
    :param itemtype: either ``1PL``, ``2PL``, ``3PL`` or ``4PL`` for the one-, two-,
                     three- or four-parameter logistic model
    :param corr: the correlation between item discrimination and difficulty. If
                 ``itemtype == '1PL'``, it is ignored.
    :return: an ``n x 4`` numerical matrix containing item parameters
    :rtype: numpy.ndarray

    >>> generate_item_bank(5, '1PL')
    >>> generate_item_bank(5, '2PL')
    >>> generate_item_bank(5, '3PL')
    >>> generate_item_bank(5, '4PL')
    >>> generate_item_bank(5, '4PL', corr=0)
    """

    valid_itemtypes = ['1PL', '2PL', '3PL', '4PL']

    if itemtype not in valid_itemtypes:
        raise ValueError('Item type not in ' + str(valid_itemtypes))

    means = [0, 1.2]
    stds = [1, 0.25]
    covs = [[stds[0]**2, stds[0] * stds[1] * corr], [stds[0] * stds[1] * corr, stds[1]**2]]

    b, a = numpy.random.multivariate_normal(means, covs, n).T

    # if by chance there is some discrimination value below zero
    # this makes the problem go away
    if any(disc < 0 for disc in a):
        min_disc = min(a)
        a = [disc + abs(min_disc) for disc in a]

    if itemtype not in ['2PL', '3PL', '4PL']:
        a = numpy.ones(n)

    if itemtype in ['3PL', '4PL']:
        c = numpy.random.normal(.25, .02, n)
    else:
        c = numpy.zeros(n)

    if itemtype == '4PL':
        d = numpy.random.uniform(.94, 1, n)
    else:
        d = numpy.ones(n)

    return irt.normalize_item_bank(numpy.array([a, b, c, d]).T)


def random_response_vector(size: int):
    return [bool(random.getrandbits(1)) for _ in range(size)]


if __name__ == '__main__':
    import doctest

    doctest.testmod()
