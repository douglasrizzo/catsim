import numpy
from scipy.optimize import differential_evolution

from catsim import cat, irt
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

    def estimate(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        response_vector: list = None,
        est_theta: float = None,
        **kwargs
    ) -> float:
        """Returns the theta value that minimizes the negative log-likelihood function, given the current state of the
         test for the given examinee.

        :param index: index of the current examinee in the simulator
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param response_vector: a boolean list containing the examinee's answers to the administered items
        :param est_theta: a float containing the current estimated proficiency
        :returns: the current :math:`\\hat\\theta`
        """
        items, administered_items, response_vector, est_theta = \
            self._prepare_args(
                return_items=True,
                return_response_vector=True,
                return_est_theta=True,
                index=index,
                items=items,
                administered_items=administered_items,
                response_vector=response_vector,
                est_theta=est_theta,
                **kwargs
            )

        assert items is not None
        assert administered_items is not None
        assert response_vector is not None
        assert est_theta is not None

        self._calls += 1
        self._evaluations = 0

        if len(set(response_vector)) == 1 and self._dodd:
            return cat.dodd(est_theta, items, response_vector[-1])

        # TODO may need to check if response_vector is empty
        if all(response_vector):
            return float('inf')
        elif not any(response_vector):
            return float('-inf')

        if len(administered_items) > 0:
            lower_bound = min(items[administered_items][:, 1])
            upper_bound = max(items[administered_items][:, 1])
        else:
            lower_bound = min(items[:, 1])
            upper_bound = max(items[:, 1])

        best_theta = float('-inf')
        max_ll = float('-inf')

        # the estimator starts with a rough search, which gets finer with each pass
        for granularity in range(10):

            # generate a list of candidate theta values
            candidates = numpy.linspace(lower_bound, upper_bound, 10)
            interval_size = candidates[1] - candidates[0]

            if self._verbose:
                print(
                    'Pass: {0}\n\tBounds: {1} {2}\n\tInterval size: {3}'.format(
                        granularity + 1, lower_bound, upper_bound, interval_size
                    )
                )

            # we'll use the concave nature of the log-likelihood function
            # to program a primitive early stopping method in our search
            previous_ll = float('-inf')

            # iterate through each candidate
            for candidate_theta in candidates:
                self._evaluations += 1

                current_ll = irt.log_likelihood(
                    candidate_theta, response_vector, items[administered_items]
                )

                # we search the function from left to right, so when the
                # log-likelihood of the current theta is smaller than the one
                # from the previous theta we tested, it means it's all downhill
                # from then on, so we stop our search
                if current_ll < previous_ll:
                    break
                previous_ll = current_ll

                # check if the LL of the current candidate theta is larger than the best one checked as of yet
                if current_ll > max_ll:
                    if self._verbose:
                        print('\t\tTheta: {0}, LL: {1}'.format(candidate_theta, current_ll))

                    if abs(best_theta - candidate_theta) < float('1e-' + str(self._precision)):
                        return self._getout(candidate_theta)

                    max_ll = current_ll
                    best_theta = candidate_theta

            # the bounds of the new candidates are adjusted around the current best theta value
            lower_bound = best_theta - interval_size
            upper_bound = best_theta + interval_size

        return self._getout(best_theta)

    def _getout(self, theta: float) -> float:
        if self._verbose:
            print('{0} evaluations'.format(self._evaluations))

        return theta


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

    def estimate(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        response_vector: list = None,
        **kwargs
    ) -> float:
        """Uses :py:func:`scipy.optimize.differential_evolution` to return the theta value
        that minimizes the negative log-likelihood function, given the current state of the
        test for the given examinee.

        :param index: index of the current examinee in the simulator
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param response_vector: a boolean list containing the examinee's answers to the administered items
        :returns: the current :math:`\\hat\\theta`
        """
        items, administered_items, response_vector = \
            self._prepare_args(
                return_items=True,
                return_response_vector=True,
                index=index,
                items=items,
                administered_items=administered_items,
                response_vector=response_vector,
                **kwargs
            )

        assert response_vector is not None
        assert items is not None
        assert administered_items is not None

        self._calls += 1

        res = differential_evolution(
            irt.negative_log_likelihood,
            bounds=[[self._lower_bound * 2, self._upper_bound * 2]],
            args=(response_vector, items[administered_items])
        )

        self._evaluations = res.nfev

        return res.x[0]
