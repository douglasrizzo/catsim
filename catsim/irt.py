"""Module containing functions pertaining to Item Response Theory
three-parameter logistic model."""

import math
import numpy as np


def tpm(theta, a, b, c):
    """Item Response Theory three-parameter logistic function:

    .. math:: P(X_i = 1| \\theta) = c_i + \\frac{1-c_i}{1+ e^{Da_i(\\theta-b_i)}}

    :param theta: the individual's proficiency value. This parameter value has
                  no boundary, but if a distribution of the form :math:`N(0, 1)` was
                  used to estimate the parameters, then :math:`-4 \\leq \\theta \\leq
                  4`.

    :param a: the discrimination parameter of the item, usually a positive
              value in which :math:`0.8 \\leq a \\leq 2.5`.

    :param b: the item difficulty parameter. This parameter value has no
              boundary, but if a distribution of the form :math:`N(0, 1)` was used to
              estimate the parameters, then :math:`-4 \\leq b \\leq 4`.

    :param c: the item pseudo-guessing parameter. Being a probability,
        :math:`0\\leq c \\leq 1`, but items considered good usually have
        :math:`c \\leq 0.2`.
    """
    try:
        return c + ((1 - c) / (1 + math.exp(-a * (theta - b))))
    except OverflowError:
        print('----ERROR HAPPENED WITH THESE VALUES: ' +
              format([theta, a, b, c]))
        raise


def logLik(est_theta, response_vector, administered_items):
    """Calculates the log-likelihood of an estimated proficiency, given a
    response vector and the parameters of the answered items.

    .. math:: L(X_{Ij} | \\theta_j, a_I, b_I, c_I) = \\prod_{i=1} ^ I P_{ij}(\\theta)^{X_{ij}} Q_{ij}(\\theta)^{1-X_{ij}}

    For computational reasons, it is common to use the log-likelihood in
    maximization/minimization problems, transforming the product of
    probabilities in a sum of probabilities:

    .. math:: \\log L(X_{Ij} | \\theta_j, , a_I, b_I, c_I) = \\sum_{i=1} ^ I \\left\\lbrace x_{ij} \\log P_{ij}(\\theta)+ (1 - x_{ij}) \\log Q_{ij}(\\theta) \\right\\rbrace

    :param est_theta: estimated proficiency value
    :param response_vector: a binary list containing the response vector
    :param administered_items: a numpy array containing the parameters of the answered items
    """
    # inspired in the example found in
    # http://stats.stackexchange.com/questions/66199/maximum-likelihood-curve-
    # model-fitting-in-python
    # try:
    if len(response_vector) != administered_items.shape[0]:
        raise ValueError(
            'Response vector and administered items must have the same number of items')
    LL = 0

    for i in range(len(response_vector)):
        prob = tpm(est_theta, administered_items[i][
            0], administered_items[i][1], administered_items[i][2])

        LL += (response_vector[i] * math.log(prob)) + \
              ((1 - response_vector[i]) * math.log(1 - prob))
    return LL
    # except OverflowError:
    #     print('Deu pau com esses valores: \n' + str(est_theta) + '\n' +
    #           str([prob, math.log10(prob)]) + '\n' + str(response_vector))
    #     raise


def negativelogLik(est_theta, *args):
    """Function used by :py:mod:`scipy.optimize` functions to find the estimated
    proficiency that maximizes the likelihood of a given response vector

    :param est_theta: estimated proficiency value
    :type est_theta: float
    :param args: a list containing the response vector and the array of
                 administered items, just like :py:func:`logLik`
    :type args: list
    :return: the estimated proficiency that maximizes the likelihood function
    """
    return -logLik(est_theta, args[0], args[1])


def inf(theta, a, b, c):
    """Item Response Theory three-parameter information function

    .. math:: I(\\theta) = a^2\\frac{(P(\\theta)-c)^2}{(1-c)^2}.\\frac{(1-P(\\theta))}{P(\\theta)}

    :param theta: the individual's proficiency value. This parameter value has
                  no boundary, but if a distribution of the form
                  :math:`N(0, 1)` was used to estimate the parameters, then
                  :math:`-4 \\leq \\theta \\leq 4`.

    :param a: the discrimination parameter of the item, usually a positive
              value in which :math:`0.8 \\leq a \\leq 2.5`.

    :param b: the item difficulty parameter. This parameter value has no
              boundary, but if a distribution of the form :math:`N(0, 1)` was
              used to estimate the parameters, then :math:`-4 \\leq b \\leq 4`.

    :param c: the item pseudo-guessing parameter. Being a probability,
        :math:`0\\leq c \\leq 1`, but items considered good usually have
        :math:`c \\leq 0.2`.

    :returns: the information value of the item at the designated `theta` point
    :rtype: float
    """
    ml3 = tpm(theta, a, b, c)
    return math.pow(a, 2) * (math.pow(ml3 - c, 2) /
                             math.pow(1 - c, 2)) * (1 - ml3) / ml3


def normalize_item_bank(items):
    """Normalize an item matrix so that it conforms to the standard used by catsim.
    The item matrix must have dimension nx3, in which column 1 represents item discrimination,
    column 2 represents item difficulty and column 3 represents the pseudo-guessing parameter.

    If the matrix has one column, it is assumed to be the difficulty column and the other
    two columns are added such that items simulate the 1-parameter logistic model.

    If the matrix has two columns, they are assumed to be the discrimination and difficulty
    columns, respectively. the pseudo-guessing column is added such that items simulate the 2-parameter logistic model.

    :param items: the item matrix
    :type items: numpy.ndarray

    :returns: an nx3 item matrix conforming to 1, 2 and 3 parameter logistic models
    :rtype: numpy.ndarray
    """
    if items.shape[1] == 1:
        items = np.append(np.ones((items.shape[0])), items, axis=1)
    if items.shape[1] == 2:
        items = np.append(items, np.zeros((items.shape[0])), axis=1)

    return items


def validate_item_bank(items, raise_err=False):
    """Validates the shape and parameters in the item matrix so that it conforms to the standard
    used by catsim. The item matrix must have dimension nx3, in which column 1 represents item
    discrimination, column 2 represents item difficulty and column 3 represents the
    pseudo-guessing parameter.

    The item matrix must have at least one line, exactly three columns and
    :math:`\\forall i \\in I , a_i > 0 \\wedge 0 < c_i < 1`

    :param items: the item matrix
    :type items: numpy.ndarray
    :param raise_err: whether to raise an error in case the validation fails or
                      just print the error message to standard output.
    :type raise_err: bool
    """
    if type(items) is not np.ndarray:
        raise ValueError('Item matrix is not of type {0}'.format(type(np.zeros((1)))))

    err = ''

    if items.shape[1] > 3:
        err += '\nItem matriz has too many columns'
    elif items.shape[1] < 3:
        if items.shape[1] == 1:
            err += '\nItem matrix has no discrimination or pseudo-guessing parameter columns'
        elif items.shape[1] == 2:
            err += '\nItem matrix has no pseudo-guessing parameter column'
    else:
        if any([disc for disc in items[:, 2]] < 0):
            err += '\nThere are items with discrimination < 0'
        if any([guess for guess in items[:, 2]] < 0):
            err += '\nThere are items with pseudo-guessing < 0'
        if any([guess for guess in items[:, 2]] > 1):
            err += '\nThere are items with pseudo-guessing > 1'

    if raise_err:
        raise ValueError(err)

    print(err)
