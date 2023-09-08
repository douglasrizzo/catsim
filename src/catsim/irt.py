import math

from math import pi  # noqa # pylint: disable=unused-import
from typing import List

import numexpr
import numpy


def icc(theta: float, a: float, b: float, c: float = 0, d: float = 1) -> float:
    """Item Response Theory four-parameter logistic function [Ayala2009]_, [Magis13]_:

    .. math:: P(X_i = 1| \\theta) = c_i + \\frac{d_i-c_i}{1+ e^{-a_i(\\theta-b_i)}}

    :param theta: the individual's ability value. This parameter value has
                  no boundary, but if a distribution of the form :math:`N(0, 1)` was
                  used to estimate the parameters, then :math:`-4 \\leq \\theta \\leq
                  4`.

    :param a: the discrimination parameter of the item, usually a positive
              value in which :math:`0.8 \\leq a \\leq 2.5`.

    :param b: the item difficulty parameter. This parameter value has no
              boundaries, but it is necessary that it be in the same value space
              as `theta` (usually :math:`-4 \\leq b \\leq 4`).

    :param c: the item pseudo-guessing parameter. Being a probability,
              :math:`0\\leq c \\leq 1`, but items considered good usually have
              :math:`c \\leq 0.2`.

    :param d: the item upper asymptote. Being a probability,
              :math:`0\\leq d \\leq 1`, but items considered good usually have
              :math:`d \\approx 1`.
    """
    return c + ((d - c) / (1 + math.e ** (-a * (theta - b))))


def _split_params(items: numpy.ndarray):
    """Split the item matrix parameters into columns.

    :param items: an item matrix with four columns representing four parameters.
    :returns: a 4-tuple with each column in a different slot."""
    return items[:, 0], items[:, 1], items[:, 2], items[:, 3]


def detect_model(items: numpy.ndarray) -> int:
    """Detects which logistic model an item matrix fits into.

    :param items: an item matrix
    :return: an int between 1 and 4 denoting the logistic model of the given item matrix
    """
    a, b, c, d = _split_params(items)

    if any(d != 1):
        return 4
    if any(c != 0):
        return 3
    if len(set(a)) > 1:
        return 2
    return 1


def icc_hpc(theta: float, items: numpy.ndarray) -> numpy.ndarray:
    """Implementation of :py:func:`icc` using :py:mod:`numpy` and :py:mod:`numexpr` in which the characteristic
    function for all items in a `numpy.ndarray` are computed at once.

    :param theta: the individual's ability value.
    :param items: array containing the four item parameters.
    :returns: an array of all item characteristic functions, given the current ``theta``"""
    a, b, c, d = _split_params(items)

    return numexpr.evaluate("c + ((d - c) / (1 + exp((-a * (theta - b)))))")


def inf_hpc(theta: float, items: numpy.ndarray):
    """Implementation of :py:func:`inf` using :py:mod:`numpy` and :py:mod:`numexpr` in which the information
    values for all items in a `numpy.ndarray` are computed at once.

    :param theta: the individual's ability value.
    :param items: array containing the four item parameters.
    :returns: an array of all item information values, given the current ``theta``"""
    a, b, c, d = _split_params(items)
    p = icc_hpc(theta, items)

    return numexpr.evaluate("(a ** 2 * (p - c) ** 2 * (d - p) ** 2) / ((d - c) ** 2 * p * (1 - p))")


def inf(theta: float, a: float, b: float, c: float = 0, d: float = 1) -> float:
    """Calculates the information value of an item using the Item Response Theory
    four-parameter logistic model function [Ayala2009]_, [Magis13]_:

    .. math:: I_i(\\theta) = \\frac{a^2[(P(\\theta)-c)]^2[d - P(\\theta)]^2}{(d-c)^2[1-P(\\theta)]P(\\theta)}

    :param theta: the individual's ability value. This parameter value has
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

    :param d: the item upper asymptote. Being a probability,
              :math:`0\\leq d \\leq 1`, but items considered good usually have
              :math:`d \\approx 1`.

    :returns: the information value of the item at the designated `theta` point."""
    p = icc(theta, a, b, c, d)

    return (a**2 * (p - c) ** 2 * (d - p) ** 2) / ((d - c) ** 2 * p * (1 - p))


def test_info(theta: float, items: numpy.ndarray) -> float:
    """Computes the test information of a test at a specific :math:`\\theta` value [Ayala2009]_:

    .. math:: I(\\theta) = \\sum_{j \\in J} I_j(\\theta)

    where :math:`J` is the set of items in the test and :math:`I_j(\\theta)` is the
    item information of :math:`j` at aspecific :math:`\\theta` value.

    :param theta: a ability value.
    :param items: a matrix containing item parameters.
    :returns: the test information at `theta` for a test represented by `items`.
    """
    return float(numpy.sum(inf_hpc(theta, items)))


def var(theta: float, items: numpy.ndarray) -> float:
    """Computes the variance (:math:`Var`) of the ability estimate of a test at a
    specific :math:`\\theta` value [Ayala2009]_:

    .. math:: Var = \\frac{1}{I(\\theta)}

    where :math:`I(\\theta)` is the test information (see :py:func:`test_info`).

    :param theta: a ability value.
    :param items: a matrix containing item parameters.
    :returns: the variance of ability estimation at `theta` for a test represented by `items`.
    """
    try:
        return 1 / test_info(theta, items)
    except ZeroDivisionError:
        return float("-inf")


def see(theta: float, items: numpy.ndarray) -> float:
    """Computes the standard error of estimation (:math:`SEE`) of a test at a
    specific :math:`\\theta` value [Ayala2009]_:

    .. math:: SEE = \\sqrt{\\frac{1}{I(\\theta)}}

    where :math:`I(\\theta)` is the test information (see :py:func:`test_info`).

    :param theta: a ability value.
    :param items: a matrix containing item parameters.
    :returns: the standard error of estimation at `theta` for a test represented by `items`.
    """
    try:
        return math.sqrt(var(theta, items))
    except ValueError:
        return float("inf")


def reliability(theta: float, items: numpy.ndarray):
    """Computes test reliability [Thissen00]_, given by:

    .. math:: Rel = 1 - \\frac{1}{I(\\theta)}

    Test reliability is a measure of internal consistency for the test, similar
    to Cronbach's :math:`\\alpha` in Classical Test Theory. Its value is always
    lower than 1, with values close to 1 indicating good reliability. If
    :math:`I(\\theta) < 1`, :math:`Rel < 0` and in these cases it does not make
    sense, but usually the application of additional items solves this problem.

    :param theta: a ability value.
    :param items: a matrix containing item parameters.
    :returns: the test reliability at `theta` for a test represented by `items`.
    """
    return 1 - var(theta, items)


def max_info(a: float = 1, b: float = 0, c: float = 0, d: float = 1) -> float:
    """Returns the :math:`\\theta` value to which the item with the given parameters
    gives maximum information. For the 1-parameter and 2-parameter
    logistic models, this :math:`\\theta` corresponds to where :math:`b = 0.5`.
    In the 3-parameter and 4-parameter logistic models, however, this value is given
    by ([Magis13]_)

    .. math:: argmax_{\\theta}I(\\theta) = b + \\frac{1}{a} log \\left(\\frac{x^* - c}{d - x^*}\\right)

    where

    .. math:: x^* = 2 \\sqrt{\\frac{-u}{3}} cos\\left\{\\frac{1}{3}acos\\left(-\\frac{v}{2}\\sqrt{\\frac{27}{-u^3}}\\right)+\\frac{4 \\pi}{3}\\right\} + 0.5

    .. math:: u = -\\frac{3}{4} + \\frac{c + d - 2cd}{2}

    .. math:: v = -\\frac{c + d - 1}{4}

    A few results can be seen in the plots below:

    .. plot::

        from catsim.cat import generate_item_bank
        from catsim import plot
        items = generate_item_bank(2)
        for item in items:
            plot.item_curve(item[0], item[1], item[2], item[3], ptype='iic', max_info=True)

    :param a: item discrimination parameter
    :param b: item difficulty parameter
    :param c: item pseudo-guessing parameter
    :param d: item upper asymptote
    :returns: the :math:`\\theta` value to which the item with the given parameters
              gives maximum information
    """
    # for explanations on finding the following values, see referenced work in function description
    if d == 1:
        if c == 0:
            return b
        return b + (1 / a) * math.log((1 + math.sqrt(1 + 8 * c)) / 2)
    else:
        u = -(3 / 4) + ((c + d - 2 * c * d) / 2)
        v = (c + d - 1) / 4
        x_star = (
            2
            * math.sqrt(-u / 3)
            * math.cos(
                (1 / 3) * math.acos(-(v / 2) * math.sqrt(27 / (-math.pow(u, 3))))
                + (4 * math.pi / 3)
            )
            + 0.5
        )

        return b + (1 / a) * math.log((x_star - c) / (d - x_star))


def max_info_hpc(items: numpy.ndarray):
    """Implementation of :py:func:`max_info` using :py:mod:`numpy` and :py:mod:`numexpr` in which the theta
    value that maximizes the information function for all items is returned in a `numpy.ndarray`.

    :param items: array containing the four item parameters.
    :returns: an array of all theta values that maximize the information function of each item."""
    a, b, c, d = _split_params(items)

    if all(d == 1):
        if all(c == 0):
            return b
        return numexpr.evaluate("b + (1 / a) * log((1 + sqrt(1 + 8 * c)) / 2)")
    else:
        u = numexpr.evaluate("-(3 / 4) + ((c + d - 2 * c * d) / 2)")
        v = numexpr.evaluate("(c + d - 1) / 4")
        x_star = numexpr.evaluate(
            "2 * sqrt(-u / 3) * cos((1 / 3) * arccos(-(v / 2) * sqrt(27 / -(u ** 3))) + (4 * pi / 3)) + 0.5"
        )

        return numexpr.evaluate("b + (1 / a) * log((x_star - c) / (d - x_star))")


def log_likelihood(
    est_theta: float, response_vector: List[bool], administered_items: numpy.ndarray
) -> float:
    """Calculates the log-likelihood of an estimated ability, given a
    response vector and the parameters of the answered items [Ayala2009]_.

    The likelihood function of a given :math:`\\theta` value given the answers to :math:`I` items is given by:

    .. math:: L(X_{Ij} | \\theta_j, a_I, b_I, c_I, d_I) = \\prod_{i=1} ^ I P_{ij}(\\theta)^{X_{ij}} Q_{ij}(\\theta)^{1-X_{ij}}

    Finding the maximum of :math:`L(X_{Ij})` includes using the product rule of derivations.
    Since :math:`L(X_{Ij})` has :math:`j` parts, it can be quite complicated to do so. Also,
    for computational reasons, the product of probabilities can quickly tend to 0, so it is
    common to use the log-likelihood in maximization/minimization problems, transforming the
    product of probabilities in a sum of probabilities:

     .. math:: \\log L(X_{Ij} | \\theta_j, a_I, b_I, c_I, d_I) = \\sum_{i=1} ^ I
               \\left\\lbrace x_{ij} \\log P_{ij}(\\theta)+ (1 - x_{ij}) \\log
               Q_{ij}(\\theta) \\right\\rbrace

    :param est_theta: estimated ability value.
    :param response_vector: a Boolean list containing the response vector.
    :param administered_items: a numpy array containing the parameters of the answered items.
    :returns: log-likelihood of a given ability value, given the responses to the administered items.
    """
    if len(response_vector) != administered_items.shape[0]:
        raise ValueError(
            "Response vector and administered items must have the same number of items"
        )
    if len(set(response_vector) - {True, False}) > 0:
        raise ValueError("Response vector must contain only Boolean elements")

    ps = icc_hpc(est_theta, administered_items)
    ll = numexpr.evaluate("sum(where(response_vector, log(ps), log(1 - ps)))")

    return ll


def negative_log_likelihood(est_theta: float, *args) -> float:
    """Function used by :py:mod:`scipy.optimize` optimization functions that tend to minimize
    values, instead of maximizing them. Calculates the negative log-likelihood of a ability
    value, given a response vector and the parameters of the administered items. The value of
    :py:func:`negative_log_likelihood` is simply the value of :math:`-` :py:func:`log_likelihood`.

    :param est_theta: estimated ability value

    args:

    :param response_vector list: a Boolean list containing the response vector
    :param administered_items numpy.ndarray: a numpy array containing the parameters of the answered items
    :returns: negative log-likelihood of a given ability value, given the responses to the administered items
    """
    return -log_likelihood(est_theta, args[0], args[1])


def normalize_item_bank(items: numpy.ndarray) -> numpy.ndarray:
    """Normalize an item matrix so that it conforms to the standard used by catsim.
    The item matrix must have dimension :math:`n \\times 4`, in which column 1 represents item
    discrimination, column 2 represents item difficulty, column 3 represents the pseudo-guessing
    parameter and column 4 represents the item upper asymptote.

    If the matrix has one column, it is assumed to be the difficulty column and the other
    three columns are added such that items simulate the 1-parameter logistic model.

    If the matrix has two columns, they are assumed to be the discrimination and difficulty
    columns, respectively. The pseudo-guessing and upper asymptote columns are added such that
    items simulate the 2-parameter logistic model.

    If the matrix has three columns, they are assumed to be the discrimination, difficulty
    and pseudo-guessing columns, respectively. The upper asymptote column is added such that
    items simulate the 3-parameter logistic model.

    :param items: the item matrix.
    :returns: an :math:`n \\times 4` item matrix conforming to the 4 parameter logistic model.
    """
    if len(items.shape) == 1:
        items = numpy.expand_dims(items, axis=0)
    if items.shape[1] == 1:
        items = numpy.append(numpy.ones((items.shape[0], 1)), items, axis=1)
    if items.shape[1] == 2:
        items = numpy.append(items, numpy.zeros((items.shape[0], 1)), axis=1)
    if items.shape[1] == 3:
        items = numpy.append(items, numpy.ones((items.shape[0], 1)), axis=1)

    return items


def validate_item_bank(items: numpy.ndarray, raise_err: bool = False):
    """Validates the shape and parameters in the item matrix so that it conforms to the standard
    used by catsim. The item matrix must have dimension nx4, in which column 1 represents item
    discrimination, column 2 represents item difficulty, column 3 represents the
    pseudo-guessing parameter and column 4 represents the item upper asymptote.

    The item matrix must have at least one line, representing an item, and exactly four columns, representing the discrimination, difficulty, pseudo-guessing and upper asymptote parameters (:math:`a`, :math:`b`, :math:`c` and :math:`d`), respectively. The item matrix is considered valid if, for all items in the matrix,
    :math:`a > 0 \\wedge 0 < c < 1 \\wedge 0 < d < 1`.

    :param items: the item matrix.
    :param raise_err: whether to raise an error in case the validation fails or
                      just print the error message to standard output.
    """
    if not isinstance(items, numpy.ndarray):
        raise ValueError("Item matrix is not of type numpy.ndarray")

    err = ""

    if len(items.shape) == 1:
        err += "Item matrix has only one dimension."
    elif items.shape[1] > 4:
        print(
            "\nItem matrix has more than 4 columns. catsim tends to add \
            columns to the matrix during the simulation, so it's not a good idea to keep them."
        )
    elif items.shape[1] < 4:
        if items.shape[1] == 1:
            err += "\nItem matrix has no discrimination, pseudo-guessing or upper asymptote parameter columns"
        elif items.shape[1] == 2:
            err += "\nItem matrix has no pseudo-guessing or upper asymptote parameter columns"
        elif items.shape[1] == 3:
            err += "\nItem matrix has no upper asymptote parameter column"
    else:
        if any(items[:, 0] < 0):
            err += "\nThere are items with discrimination < 0"
        if any(items[:, 2] < 0):
            err += "\nThere are items with pseudo-guessing < 0"
        if any(items[:, 2] > 1):
            err += "\nThere are items with pseudo-guessing > 1"
        if any(items[:, 3] > 1):
            err += "\nThere are items with upper asymptote > 1"
        if any(items[:, 3] < 0):
            err += "\nThere are items with upper asymptote < 0"

    if len(err) > 0 and raise_err:
        raise ValueError(err)

    print(err)
