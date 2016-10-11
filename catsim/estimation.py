import numpy
from scipy.optimize import differential_evolution

from catsim import irt, cat
from catsim.simulation import Estimator


class HillClimbingEstimator(Estimator):
    """Estimator that uses a hill-climbing algorithm to maximize the likelihood function

    :param precision: number of decimal points of precision
    :param verbose: verbosity level of the maximization method
    """

    def __str__(self):
        return 'Hill Climbing Estimator'

    def __init__(self, precision: int = 6, dodd: bool = False, verbose: bool = False):
        super().__init__()
        self._precision = precision
        self._verbose = verbose
        self._evaluations = 0
        self._calls = 0
        self._dodd = dodd

    @property
    def calls(self) -> float:
        """How many times the estimator has been called to maximize/minimize the log-likelihood function

        :returns: number of times the estimator has been called to maximize/minimize the log-likelihood function"""
        return self._calls

    @property
    def evaluations(self) -> float:
        """Total number of times the estimator has evaluated the log-likelihood function during its existence

        :returns: number of function evaluations"""
        return self._evaluations

    @property
    def avg_evaluations(self) -> float:
        """Average number of function evaluations for all tests the estimator has been used

        :returns: average number of function evaluations"""
        return self._evaluations / self._calls

    @property
    def dodd(self) -> bool:
        """Whether Dodd's method will be called by estimator in case the response vector
        is composed solely of right or wrong answers.

        :returns: boolean value indicating if Dodd's method will be used or not."""
        return self._dodd

    def estimate(self, index: int) -> float:
        """Returns the theta value that minimizes the negative log-likelihood function, given the current state of the
         test for the given examinee.

        :param index: index of the current examinee in the simulator
        :returns: the current :math:`\\hat\\theta`
        """
        self._calls += 1

        if len(set(self.simulator.response_vectors[index])) == 1 and self._dodd:
            return cat.dodd(self.simulator.latest_estimations[index], self.simulator.items,
                            self.simulator.response_vectors[index][-1])

        if set(self.simulator.response_vectors[index]) == 1:
            return float('inf')
        elif set(self.simulator.response_vectors[index]) == 0:
            return float('-inf')

        if len(self.simulator.administered_items[index]) > 0:
            lbound = min(self.simulator.items[self.simulator.administered_items[index]][:, 1])
            ubound = max(self.simulator.items[self.simulator.administered_items[index]][:, 1])
        else:
            lbound = min(self.simulator.items[:, 1])
            ubound = max(self.simulator.items[:, 1])

        best_theta = -float('inf')
        max_ll = -float('inf')

        self._evaluations = 0

        for _ in range(10):
            intervals = numpy.linspace(lbound, ubound, 10)
            if self._verbose:
                print(('Bounds: ' + str(lbound) + ' ' + str(ubound)))
                print(('Interval size: ' + str(intervals[1] - intervals[0])))

            for ii in intervals:
                self._evaluations += 1
                ll = irt.logLik(ii, self.simulator.response_vectors[index],
                                self.simulator.items[self.simulator.administered_items[index]])
                if ll > max_ll:
                    max_ll = ll

                    if self._verbose:
                        print(('Iteration: {0}, Theta: {1}, LL: {2}'.format(self._evaluations, ii, ll)))

                    if abs(best_theta - ii) < float('1e-' + str(self._precision)):
                        return ii

                    best_theta = ii

                else:
                    lbound = best_theta - (intervals[1] - intervals[0])
                    ubound = ii
                    break

        return best_theta


class DifferentialEvolutionEstimator(Estimator):
    """Estimator that uses :py:func:`scipy.optimize.differential_evolution` to minimize the negative log-likelihood function

    :param bounds: a tuple containing both lower and upper bounds for the differential
                   evolution algorithm search space. In theory, it is best if they
                   represent the minimum and maximum possible :math:`\\theta` values;
                   in practice, one could also use the smallest and largest difficulty
                   parameters in the item bank, in case no better bounds for
                   :math:`\\theta` exist.
    """

    def __str__(self):
        return 'Differential Evolution Estimator'

    def __init__(self, bounds: tuple):
        super(DifferentialEvolutionEstimator, self).__init__()
        self._lower_bound = min(bounds)
        self._upper_bound = max(bounds)
        self._evaluations = 0
        self._calls = 0

    @property
    def calls(self):
        """How many times the estimator has been called to maximize/minimize the log-likelihood function

        :returns: number of times the estimator has been called to maximize/minimize the log-likelihood function"""
        return self._calls

    @property
    def evaluations(self):
        """Total number of times the estimator has evaluated the log-likelihood function during its existence

        :returns: number of function evaluations"""
        return self._evaluations

    @property
    def avg_evaluations(self):
        """Average number of function evaluations for all tests the estimator has been used

        :returns: average number of function evaluations"""
        return self._evaluations / self._calls

    def estimate(self, index: int) -> float:
        """Uses :py:func:`scipy.optimize.differential_evolution` to return the theta value
        that minimizes the negative log-likelihood function, given the current state of the
        test for the given examinee.

        :param index: index of the current examinee in the simulator
        :returns: the current :math:`\\hat\\theta`
        """

        self._calls += 1

        res = differential_evolution(irt.negativelogLik, bounds=[[self._lower_bound * 2, self._upper_bound * 2]], args=(
        self.simulator.response_vectors[index], self.simulator.items[self.simulator.administered_items[index]]))

        self._evaluations = res.nfev

        return res.x[0]
