from abc import ABCMeta, abstractmethod

from catsim import irt
import numpy
from scipy.optimize import fmin
from scipy.optimize import differential_evolution


class Estimator:
    """Base class for proficiency estimators"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(Estimator, self).__init__()

    @abstractmethod
    def estimate(self, response_vector, administered_items, current_theta):
        """Uses a hill-climbing algorithm to find and returns the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :type response_vector: list
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :type administered_items: numpy.ndarray
        :param current_theta: the current estimation of the examinee's :math:`\\theta` value
        :type current_theta: float
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        :rtype: float
        """
        pass


class HillClimbingEstimator(Estimator):
    """Estimator that uses a hill-climbing algorithm to maximize the likelihood function

    :param precision: number of decimal points of precision
    :type precision: int
    :param verbose: verbosity level of the maximization method
    :type verbose: boolean
    """

    def __init__(self, precision=6, verbose=False):
        super(HillClimbingEstimator, self).__init__()
        self.__precision = precision
        self.__verbose = verbose

    def estimate(self, response_vector, administered_items, current_theta=None):
        """Uses a hill-climbing algorithm to find and returns the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :type response_vector: list
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :type administered_items: numpy.ndarray
        :param current_theta: the current estimation of the examinee's :math:`\\theta` value; not used by this selector.
        :type current_theta: float
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        :rtype: float
        """

        if set(response_vector) == 1:
            return float('inf')
        elif set(response_vector) == 0:
            return float('-inf')

        lbound = min(administered_items[:, 1])
        ubound = max(administered_items[:, 1])
        best_theta = -float('inf')
        max_ll = -float('inf')

        iters = 0

        for i in range(10):
            intervals = numpy.linspace(lbound, ubound, 10)
            if self.__verbose:
                print('Bounds: ' + str(lbound) + ' ' + str(ubound))
                print('Interval size: ' + str(intervals[1] - intervals[0]))

            for ii in intervals:
                iters += 1
                ll = irt.logLik(ii, response_vector, administered_items)
                if ll > max_ll:
                    max_ll = ll

                    if self.__verbose:
                        print('Iteration: {0}, Theta: {1}, LL: {2}'.format(iters, ii, ll))

                    if abs(best_theta - ii) < float('1e-' + str(self.__precision)):
                        return ii

                    best_theta = ii

                else:
                    lbound = best_theta - (intervals[1] - intervals[0])
                    ubound = ii
                    break

        return best_theta


class BinarySearchEstimator(Estimator):
    """Estimator that uses a binary search approach in the log-likelihood function domain to maximize it

    :param precision: number of decimal points of precision
    :type precision: int
    :param verbose: verbosity level of the maximization method
    :type verbose: boolean
    """

    def __init__(self):
        super(BinarySearchEstimator, self).__init__()

    def estimate(response_vector, administered_items, **kwargs):
        """Uses a binary search algorithm to find and returns the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :type response_vector: list
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :type administered_items: numpy.ndarray
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        :rtype: float
        """
        precision = kwargs['precision'] if 'precision' in kwargs else 6
        verbose = kwargs['verbose'] if 'verbose' in kwargs else False

        if set(response_vector) == 1:
            return float('inf')
        elif set(response_vector) == 0:
            return float('-inf')

        lbound = min(administered_items[:, 1])
        ubound = max(administered_items[:, 1])
        best_theta = -float('inf')
        iters = 0

        while True:
            iters += 1
            if verbose:
                print('Bounds: ' + str(lbound) + ' ' + str(ubound))
                print('Iteration: {0}, Theta: {1}, LL: {2}'.format(iters, best_theta, irt.logLik(best_theta, response_vector, administered_items)))

            if irt.logLik(ubound, response_vector, administered_items) > irt.logLik(lbound, response_vector, administered_items):

                if abs(best_theta - ubound) < float('1e-' + str(precision)):
                    return ubound

                best_theta = ubound
                lbound += (ubound - lbound) / 2
            else:

                if abs(best_theta - lbound) < float('1e-' + str(precision)):
                    return lbound

                best_theta = lbound
                ubound -= (ubound - lbound) / 2


class FMinEstimator(Estimator):
    """Estimator that uses :py:func:`scipy.optimize.fmin` to minimize the negative log-likelihood function"""

    def __init__(self):
        super(FMinEstimator, self).__init__()

    def estimate(response_vector, administered_items, current_theta):
        """Uses :py:func:`scipy.optimize.fmin` to find and return the theta value
        that minimizes the negative log-likelihood function, given a response vector, a
        matrix with the administered items parameters and the current :math:`\\theta`.

        :param response_vector: a binary list containing the response vector
        :type response_vector: list
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :type administered_items: numpy.ndarray
        :param current_theta: the current estimation of the examinee's :math:`\\theta` value
        :type current_theta: float
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        :rtype: float
        """
        return fmin(irt.negativelogLik, current_theta, (response_vector, administered_items))


class DifferentialEvolutionEstimator(Estimator):
    """Estimator that uses :py:func:`scipy.optimize.differential_evolution` to minimize the negative log-likelihood function

    :param lower_bound: a lower bound for the differential evolution algorithm search space.
                           In theory, it is best if it represents the minimum possible
                           :math:`\\theta` value; in practice, one could also use the smallest
                           difficulty parameter of an item in the item bank, in case no better
                           lower bound for :math:`\\theta` exists.
    :type lower_bound: float
    :param upper_bound: an upper bound for the differential evolution algorithm search space.
                           In theory, it is best if it represents the maximum possible
                           :math:`\\theta` value; in practice, one could also use the largest
                           difficulty parameter of an item in the item bank, in case no better
                           upper bound for :math:`\\theta` exists.
    :type upper_bound: float
    """

    def __init__(self, lower_bound, upper_bound):
        super(DifferentialEvolutionEstimator, self).__init__()
        self.__lower_bound = lower_bound
        self.__upper_bound = upper_bound

    def estimate(self, response_vector, administered_items, current_theta):
        """Uses :py:func:`scipy.optimize.fmin` to find and return the theta value
        that minimizes the negative log-likelihood function, given a response vector, a
        matrix with the administered items parameters and the current :math:`\\theta`.

        :param response_vector: a binary list containing the response vector
        :type response_vector: list
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :type administered_items: numpy.ndarray
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        :rtype: float
        """

        return differential_evolution(
            irt.negativelogLik,
            bounds=[
                [self.__lower_bound * 2, self.__upper_bound * 2]
            ],
            args=(response_vector, administered_items)
        ).x[0]
