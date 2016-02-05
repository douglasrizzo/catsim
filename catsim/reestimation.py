from abc import ABCMeta, abstractmethod

from catsim import irt
import numpy
from scipy.optimize import fmin
from scipy.optimize import differential_evolution


class Estimator:
    """Base class for proficiency estimators"""

    __metaclass__ = ABCMeta

    def __init__(self, arg):
        super(Estimator, self).__init__()
        self.arg = arg

    @abstractmethod
    def estimate():
        pass


class HillClimbingEstimator(Estimator):
    """Estimator that uses a hill-climbing algorithm to maximize the likelihood function"""

    def __init__(self):
        super(HillClimbingEstimator, self).__init__()

    def estimate(response_vector, administered_items, precision=6, verbose=False):
        """Uses a hill-climbing algorithm to find and returns the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a numpy array containing the parameters of the
                                   answered items
        :param precision: number of decimal points of precision
        :param verbose: verbosity level of the maximization method
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
            if verbose:
                print('Bounds: ' + str(lbound) + ' ' + str(ubound))
                print('Interval size: ' + str(intervals[1] - intervals[0]))

            for ii in intervals:
                iters += 1
                ll = irt.logLik(ii, response_vector, administered_items)
                if ll > max_ll:
                    max_ll = ll

                    if verbose:
                        print('Iteration: {0}, Theta: {1}, LL: {2}'.format(iters, ii, ll))

                    if abs(best_theta - ii) < float('1e-' + str(precision)):
                        return ii

                    best_theta = ii

                else:
                    lbound = best_theta - (intervals[1] - intervals[0])
                    ubound = ii
                    break

        return best_theta


class FMinEstimator(Estimator):
    """Estimator that uses :py:func:`scipy.optimize.fmin` to minimize the negative log-likelihood function"""

    def __init__(self):
        super(FMinEstimator, self).__init__()

    def estimate(current_theta, response_vector, administered_items):
        return fmin(irt.negativelogLik, current_theta, (response_vector, administered_items))


class DifferentialEvolutionEstimator(Estimator):
    """Estimator that uses :py:func:`scipy.optimize.differential_evolution` to minimize the negative log-likelihood function"""

    def __init__(self):
        super(DifferentialEvolutionEstimator, self).__init__()

    def estimate(current_theta, response_vector, administered_items, min_difficulty, max_difficulty):
        return differential_evolution(
            irt.negativelogLik, bounds=[
                [min_difficulty * 2, max_difficulty * 2]],
            args=(response_vector, administered_items)).x[0]


class BinarySearchEstimator(Estimator):
    """Estimator that uses a binary search approach in the log-likelihood function domain to maximize it"""

    def __init__(self):
        super(BinarySearchEstimator, self).__init__()

    def estimate(response_vector, items, administered_items, precision=35, verbose=False):
        """Uses a binary search algorithm to find and returns the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a numpy array containing the parameters of the
                                   answered items
        :param precision: number of decimal points of precision
        :param verbose: verbosity level of the maximization method
        """

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
                print('Iteration: {0}, Theta: {1}, LL: {2}'.format(iters, best_theta,
                                                                   irt.logLik(best_theta, response_vector, administered_items)))

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
