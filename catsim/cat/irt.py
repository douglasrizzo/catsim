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

    :param est_theta: estimated profficiency value
    :param response_vector: a binary list containing the response vector
    :param administered_items: a numpy array containing the parameters of the answered items
    """
    # inspired in the example found in
    # http://stats.stackexchange.com/questions/66199/maximum-likelihood-curve-
    # model-fitting-in-python
    # try:
    if len(response_vector) != administered_items.shape[0]:
        raise ValueError(
            'Response vector and administered items must have the number of items')
    LL = 0

    for i in range(len(response_vector)):
        prob = tpm(est_theta, administered_items[i][
            0], administered_items[i][1], administered_items[i][2])

        LL += (response_vector[i] * math.log10(prob)) + \
              ((1 - response_vector[i]) * math.log10(1 - prob))
    return LL
    # except OverflowError:
    #     print('Deu pau com esses valores: \n' + str(est_theta) + '\n' +
    #           str([prob, math.log10(prob)]) + '\n' + str(response_vector))
    #     raise


def negativelogLik(est_theta, *args):
    """Function used by :py:mod:`scipy.optimize` functions to find the estimated
    proficiency that maximizes the likelihood of a given response vector

    :param est_theta: estimated profficiency value
    :type est_theta: float
    :param args: a list containing the response vector and the array of
                 administered items, just like :py:func:`logLik`
    :type args: list
    :return: the estimated proficiency that maximizes the likelihood function
    """
    return -logLik(est_theta, args[0], args[1])


def bruteMLE(response_vector, administered_items, precision=12):
    """Finds and returns the theta value which maximizes the likelihood
    function, given a response vector and a matrix with the administered
    items parameters

    :param response_vector: a binary list containing the response vector
    :param administered_items: a numpy array containing the parameters of the
                               answered items
    :param precision: number of decimal points of precision
    """
    lbound = -8
    ubound = 8

    for i in range(precision):
        intervals = np.linspace(lbound, ubound, 20)
        previous_ll = -float('inf')
        # print('Bounds: ' + str(lbound) + ' ' + str(ubound))
        # print('Interval size: ' + str(intervals[1] - intervals[0]))

        for ii in intervals:
            ll = logLik(ii, response_vector, administered_items)
            if ll > previous_ll:
                previous_ll = ll
                best_theta = ii
                # print(best_theta)
            else:
                lbound = best_theta - (intervals[1] - intervals[0])
                ubound = best_theta + (intervals[1] - intervals[0])

    return best_theta


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
    """
    ml3 = tpm(theta, a, b, c)
    return math.pow(a, 2) * (math.pow(ml3 - c, 2) /
                             math.pow(1 - c, 2)) * (1 - ml3) / ml3
