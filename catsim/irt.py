import math
import numpy


def tpm(theta: float, a: float, b: float, c: float=0) -> float:
    """Item Response Theory three-parameter logistic function [Ayala2009]_:

    .. math:: P(X_i = 1| \\theta) = c_i + \\frac{1-c_i}{1+ e^{a_i(\\theta-b_i)}}

    :param theta: the individual's proficiency value. This parameter value has
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
    """
    return c + ((1 - c) / (1 + _tpm_exponent(theta, a, b)))


def _w(theta, a, b, c):
    p_star = math.pow(1 + _tpm_exponent(theta, a, b), -1)
    q_star = 1 - p_star
    p = tpm(theta, a, b, c)
    q = 1 - p

    return p_star * q_star / p * q


def _tpm_exponent(theta, a, b):
    return math.exp(-a * (theta - b))


def _theta_map(
    theta: float,
    response_vector: list,
    administered_items: numpy.ndarray,
    mu: float=0,
    sigma: float=1
) -> float:
    """ Maximum a posteriori estimation function for :math:`\\hat\\theta` [Andrade2000]_, given by:

    .. math:: \\sum_{i-1}^{I} a_i(1-c_i)(u_{ji}-P_{JI})W_{ji} - \\frac{(\\theta_j - \\mu)}{\\sigma^2}

    where .. math:: W = \\frac{P^*Q^*}{P^Q^}

    where :math:`P` is given by :py:func:`catsim.irt.tpm`, :math:`P^* = (1 + e^{a_i(\\theta-b_i)})^{-1}`,
    :math:`Q^* = 1 - P^*` and :math:`Q = 1 - P`
    """
    return sum(
        [
            administered_items[i, 0] * (1 - administered_items[i, 2]) * (
                response_vector[i] - tpm(
                    theta, administered_items[i, 0], administered_items[
                        i, 1
                    ], administered_items[i, 2]
                ) * _w(
                    theta, administered_items[
                        i, 0
                    ], administered_items[i, 1], administered_items[i, 2]
                )
            ) for i in range(len(response_vector))
        ]
    ) - (theta - mu) / (sigma**2)


def var(theta: float, items: numpy.ndarray) -> float:
    """Computes the variance (:math:`Var`) of the proficiency estimate of a test at a
    specific :math:`\\theta` value [Ayala2009]_:

    .. math:: Var = \\frac{1}{I(\\theta)}

    where :math:`I(\\theta)` is the test information (see :py:func:`test_info`).

    :param theta: a proficiency value.
    :param items: a matrix containing item parameters.
    :returns: the variance of proficency estimation at `theta` for a test represented by `items`.
    """
    try:
        return 1 / test_info(theta, items)
    except ZeroDivisionError:
        return float('-inf')


def see(theta: float, items: numpy.ndarray) -> float:
    """Computes the standard error of estimation (:math:`SEE`) of a test at a
    specific :math:`\\theta` value [Ayala2009]_:

    .. math:: SEE = \\\\sqrt{frac{1}{I(\\theta)}}

    where :math:`I(\\theta)` is the test information (see :py:func:`test_info`).

    :param theta: a proficiency value.
    :param items: a matrix containing item parameters.
    :returns: the standard error of estimation at `theta` for a test represented by `items`.
    """
    try:
        return math.sqrt(var(theta, items))
    except ZeroDivisionError:
        return float('-inf')


def test_info(theta: float, items: numpy.ndarray):
    """Computes the test information of a test at a specific :math:`\\theta` value [Ayala2009]_:

    .. math:: I(\\theta) = \\sum_{j \\in J} I_j(\\theta)

    where :math:`J` is the set of items in the test and :math:`I_j(\\theta)` is the
    item information of :math:`j` at aspecific :math:`\\theta` value.

    :param theta: a proficiency value.
    :param items: a matrix containing item parameters.
    :returns: the test information at `theta` for a test represented by `items`.
    """
    return sum([inf(theta, item[0], item[1], item[2]) for item in items])


def reliability(theta: float, items: numpy.ndarray):
    """ Computes test reliability [Thissen00]_, given by:

    .. math:: Rel = 1 - \\frac{1}{I(\\theta)}

    Test reliability is a measure of internal consistency for the test, similar
    to Cronbach's :math:`\\alpha` in Classical Test Theory. Its value is always
    lower than 1, with values close to 1 indicating good reliability. If
    :math:`I(\\theta) < 1`, :math:`Rel < 0` and in these cases it does not make
    sense, but usually the application of additional items solves this problem.

    :param theta: a proficiency value.
    :param items: a matrix containing item parameters.
    :returns: the test reliability at `theta` for a test represented by `items`.
    """
    return 1 - (1 / test_info(theta, items))


def inf(theta: float, a: float, b: float, c: float=0) -> float:
    """Calculates the information value of an item using the Item Response Theory
    three-parameter logistic model function [Ayala2009]_:

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

    :returns: the information value of the item at the designated `theta` point."""
    ml3 = tpm(theta, a, b, c)

    return math.pow(a, 2) * (math.pow(ml3 - c, 2) / math.pow(1 - c, 2)) * (1 - ml3) / ml3


def max_info(a=1, b=0, c=0):
    """Returns the :math:`\\theta` value to which the item with the given parameters
    gives maximum information. For the 1-parameter and 2-parameter logistic models,
    this :math:`\\theta` corresponds to where :math:`b = 0.5`. In the 3-parameter
    logistic model, however, this value is approximated by the following function:

    .. math:: argmax_{\\theta}I(\\theta) \\approx \\frac{a*b-\\log(-2(c-0.616))}{a}

    A few results can be seen in the plots below:

    .. plot::

        from catsim.cat import generate_item_bank
        from catsim import plot
        items = generate_item_bank(2)
        for item in items:
            plot.item_curve(item[0], item[1], item[2], ptype='iic', max_info=True)

    :param a: item discrimination parameter
    :param b: item difficulty parameter
    :param c: item pseudo-guessing parameter
    :param title: plot title
    :param ptype: 'icc' for the item characteristic curve, 'iic' for the item
                  information curve or 'both' for both curves in the same plot
    :param filepath: saves the plot in the given path
    :param show: whether the generated plot is to be shown

    """
    return (a * b - math.log(-2 * (c - 0.616))) / a


def logLik(est_theta: float, response_vector: list, administered_items: numpy.ndarray) -> float:
    """Calculates the log-likelihood of an estimated proficiency, given a
    response vector and the parameters of the answered items [Ayala2009]_.

    The likelihood function of a given :math:`\\theta` value given the answers to :math:`I` items is given by:

    .. math:: L(X_{Ij} | \\theta_j, a_I, b_I, c_I) = \\prod_{i=1} ^ I P_{ij}(\\theta)^{X_{ij}} Q_{ij}(\\theta)^{1-X_{ij}}

    For computational reasons, it is common to use the log-likelihood in
    maximization/minimization problems, transforming the product of
    probabilities in a sum of probabilities:

     .. math:: \\log L(X_{Ij} | \\theta_j, , a_I, b_I, c_I) = \\sum_{i=1} ^ I
               \\left\\lbrace x_{ij} \\log P_{ij}(\\theta)+ (1 - x_{ij}) \\log
               Q_{ij}(\\theta) \\right\\rbrace

    :param est_theta: estimated proficiency value.
    :param response_vector: a binary list containing the response vector.
    :param administered_items: a numpy array containing the parameters of the answered items.
    :returns: log-likelihood of a given proficiency value, given the responses to the administered items.
    """
    # inspired in the example found in
    # http://stats.stackexchange.com/questions/66199/maximum-likelihood-curve-
    # model-fitting-in-python
    # try:
    if len(response_vector) != administered_items.shape[0]:
        raise ValueError(
            'Response vector and administered items must have the same number of items'
        )
    LL = 0

    for i in range(len(response_vector)):
        prob = tpm(
            est_theta, administered_items[i][0], administered_items[i][1], administered_items[i][2]
        )

        LL += (response_vector[i] * math.log(prob)) + \
              ((1 - response_vector[i]) * math.log(1 - prob))
    return LL


def negativelogLik(est_theta: float, *args) -> float:
    """Function used by :py:mod:`scipy.optimize` optimization functions that tend to minimize
    values, instead of maximizing them. Calculates the negative log-likelihood of a proficiency
    value, given a response vector and the parameters of the administered items. The value of
    :py:func:`negativelogLik` is simply the value of :math:`-` :py:func:`logLik` or, mathematically:

    .. math:: - \\log L(X_{Ij} | \\theta_j, , a_I, b_I, c_I)

    :param est_theta: estimated proficiency value

    args:

    :param response_vector list: a binary list containing the response vector
    :param administered_items numpy.ndarray: a numpy array containing the parameters of the answered items
    :returns: negative log-likelihood of a given proficiency value, given the responses to the administered items
    """
    return -logLik(est_theta, args[0], args[1])


def normalize_item_bank(items: numpy.ndarray) -> numpy.ndarray:
    """Normalize an item matrix so that it conforms to the standard used by catsim.
    The item matrix must have dimension nx3, in which column 1 represents item discrimination,
    column 2 represents item difficulty and column 3 represents the pseudo-guessing parameter.

    If the matrix has one column, it is assumed to be the difficulty column and the other
    two columns are added such that items simulate the 1-parameter logistic model.

    If the matrix has two columns, they are assumed to be the discrimination and difficulty
    columns, respectively. the pseudo-guessing column is added such that items simulate the 2-parameter logistic model.

    :param items: the item matrix.
    :returns: an nx3 item matrix conforming to 1, 2 and 3 parameter logistic models.
    """
    if len(items.shape) == 1:
        items = numpy.expand_dims(items, axis=0)
    if items.shape[1] == 1:
        items = numpy.append(numpy.ones((items.shape[0])), items, axis=1)
    if items.shape[1] == 2:
        items = numpy.append(items, numpy.zeros((items.shape[0])), axis=1)

    return items


def validate_item_bank(items: numpy.ndarray, raise_err: bool=False):
    """Validates the shape and parameters in the item matrix so that it conforms to the standard
    used by catsim. The item matrix must have dimension nx3, in which column 1 represents item
    discrimination, column 2 represents item difficulty and column 3 represents the
    pseudo-guessing parameter.

    The item matrix must have at least one line, exactly three columns and
    :math:`\\forall i \\in I , a_i > 0 \\wedge 0 < c_i < 1`

    :param items: the item matrix.
    :param raise_err: whether to raise an error in case the validation fails or
                      just print the error message to standard output.
    """
    if type(items) is not numpy.ndarray:
        raise ValueError('Item matrix is not of type {0}'.format(numpy.ndarray))

    err = ''

    if len(items.shape) == 1:
        err += 'Item matrix has only one dimension.'
    elif items.shape[1] > 3:
        print(
            '\nItem matrix has more than 3 columns. catsim tends to add additional\
            columns to the matriz during the simulation, so it\'s not a good idea to keep them.'
        )
    elif items.shape[1] < 3:
        if items.shape[1] == 1:
            err += '\nItem matrix has no discrimination or pseudo-guessing parameter columns'
        elif items.shape[1] == 2:
            err += '\nItem matrix has no pseudo-guessing parameter column'
    else:
        if any(items[:, 2] < 0):
            err += '\nThere are items with discrimination < 0'
        if any(items[:, 2] < 0):
            err += '\nThere are items with pseudo-guessing < 0'
        if any(items[:, 2] > 1):
            err += '\nThere are items with pseudo-guessing > 1'

    if len(err) > 0 and raise_err:
        raise ValueError(err)

    print(err)
