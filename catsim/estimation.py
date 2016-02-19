from abc import ABCMeta, abstractmethod

import numpy
from catsim import irt
from scipy.optimize import fmin
from scipy.optimize import differential_evolution


class Estimator:
    """Base class for proficiency estimators"""

    __metaclass__ = ABCMeta

    def __init__(self):
        super(Estimator, self).__init__()

    @abstractmethod
    def estimate(
        self, response_vector: list, administered_items: numpy.ndarray, current_theta: float
    ) -> float:
        """Uses a hill-climbing algorithm to find and return the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :param current_theta: the current estimation of the examinee's :math:`\\theta` value
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        """
        pass


class HillClimbingEstimator(Estimator):
    """Estimator that uses a hill-climbing algorithm to maximize the likelihood function

    :param precision: number of decimal points of precision
    :param verbose: verbosity level of the maximization method
    """

    def __init__(self, precision: int=6, verbose: bool=False):
        super(HillClimbingEstimator, self).__init__()
        self._precision = precision
        self._verbose = verbose
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
        response_vector: list,
        administered_items: numpy.ndarray,
        current_theta: float=None
    ) -> float:
        """Uses a hill-climbing algorithm to find and returns the theta value
        that maximizes the likelihood function, given a response vector and a
        matrix with the administered items parameters.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :param current_theta: not used by this selector
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        :rtype: float
        """

        self._calls += 1

        if set(response_vector) == 1:
            return float('inf')
        elif set(response_vector) == 0:
            return float('-inf')

        lbound = min(administered_items[:, 1])
        ubound = max(administered_items[:, 1])
        best_theta = -float('inf')
        max_ll = -float('inf')

        self._evaluations = 0

        for _ in range(10):
            intervals = numpy.linspace(lbound, ubound, 10)
            if self._verbose:
                print('Bounds: ' + str(lbound) + ' ' + str(ubound))
                print('Interval size: ' + str(intervals[1] - intervals[0]))

            for ii in intervals:
                self._evaluations += 1
                ll = irt.logLik(ii, response_vector, administered_items)
                if ll > max_ll:
                    max_ll = ll

                    if self._verbose:
                        print(
                            'Iteration: {0}, Theta: {1}, LL: {2}'.format(
                                self._evaluations, ii, ll
                            )
                        )

                    if abs(best_theta - ii) < float('1e-' + str(self._precision)):
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
        self, response_vector: list, administered_items: numpy.ndarray, current_theta: float
    ) -> float:
        """Uses :py:func:`scipy.optimize.fmin` to find and return the theta value
        that minimizes the negative log-likelihood function, given a response vector, a
        matrix with the administered items parameters and the current :math:`\\theta`.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :param current_theta: the current estimation of the examinee's :math:`\\theta` value
        :returns: a new estimation of the examinee's proficiency, given his answers up until now
        """
        self._calls += 1
        res = fmin(
            irt.negativelogLik,
            current_theta,
            (response_vector, administered_items),
            disp=False,
            full_output=True
        )

        self._evaluations = res[3]
        return res[0][0]


class DifferentialEvolutionEstimator(Estimator):
    """Estimator that uses :py:func:`scipy.optimize.differential_evolution` to minimize the negative log-likelihood function

    :param bounds: a tuple containing both lower and upper bounds for the differential
                   evolution algorithm search space. In theory, it is best if they
                   represent the minimum and maximum possible :math:`\\theta` values;
                   in practice, one could also use the smallest and largest difficulty
                   parameters in the item bank, in case no better bounds for
                   :math:`\\theta` exist.
    """

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
        response_vector: list,
        administered_items: numpy.ndarray,
        current_theta: float=None
    ) -> float:
        """Uses :py:func:`scipy.optimize.fmin` to find and return the theta value
        that minimizes the negative log-likelihood function, given a response vector, a
        matrix with the administered items parameters and the current :math:`\\theta`.

        :param response_vector: a binary list containing the response vector
        :param administered_items: a matrix containing the parameters of the
                                   answered items
        :param current_theta: the current estimation of the examinee's :math:`\\theta` value
        :returns: not used by this selector
        """

        self._calls += 1

        res = differential_evolution(
            irt.negativelogLik,
            bounds=[
                [self._lower_bound * 2, self._upper_bound * 2]
            ],
            args=(response_vector, administered_items)
        )

        self._evaluations = res.nfev

        return res.x[0]

        class MAPEstimator(Estimator):
            """Estimator that uses the Bayesian *maximum a posteriori* concept to find the most probable value for :math:`\\hat\\theta`, given the the data
            """

            def __init__(self):
                super(MAPEstimator, self).__init__()
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
                self, response_vector: list, administered_items: numpy.ndarray, current_theta: float
            ) -> float:
                """Uses :py:func:`scipy.optimize.fmin` to find and return the theta value
                that minimizes the negative log-likelihood function, given a response vector, a
                matrix with the administered items parameters and the current :math:`\\theta`.

                :param response_vector: a binary list containing the response vector
                :param administered_items: a matrix containing the parameters of the
                                           answered items
                :param current_theta: the current estimation of the examinee's :math:`\\theta` value
                :returns: not used by this selector
                """

                # class MAPEstimator(Estimator):
                #     """Estimator that uses the Bayesian *maximum a posteriori* concept to find the most probable value for :math:`\\hat\\theta`, given the the data
                #     """
                #
                #     def __init__(self):
                #         super(MAPEstimator, self).__init__()
                #         self._evaluations = 0
                #         self._calls = 0
                #
                #     @property
                #     def calls(self):
                #         """How many times the estimator has been called to maximize/minimize the log-likelihood function
                #
                #         :returns: number of times the estimator has been called to maximize/minimize the log-likelihood function"""
                #         return self._calls
                #
                #     @property
                #     def evaluations(self):
                #         """Total number of times the estimator has evaluated the log-likelihood function during its existence
                #
                #         :returns: number of function evaluations"""
                #         return self._evaluations
                #
                #     @property
                #     def avg_evaluations(self):
                #         """Average number of function evaluations for all tests the estimator has been used
                #
                #         :returns: average number of function evaluations"""
                #         return self._evaluations / self._calls
                #
                #     def estimate(
                #         self, response_vector: list, administered_items: numpy.ndarray, current_theta: float
                #     ) -> float:
                #         """Uses :py:func:`scipy.optimize.fmin` to find and return the theta value
                #         that minimizes the negative log-likelihood function, given a response vector, a
                #         matrix with the administered items parameters and the current :math:`\\theta`.
                #
                #         :param response_vector: a binary list containing the response vector
                #         :param administered_items: a matrix containing the parameters of the
                #                                    answered items
                #         :param current_theta: the current estimation of the examinee's :math:`\\theta` value
                #         :returns: not used by this selector
                #         """
                #
                #         # def f_x(response_vector, administered_items):
                #         #     return numpy.multiply(
                #         #         [
                #         #             irt.tpm(
                #         #                 response_vector[i], administered_items[i, 0], administered_items[
                #         #                     i, 1
                #         #                 ], administered_items[i, 2]
                #         #             ) for i in range(len(response_vector))
                #         #         ]
                #         #     )
                #
                #         self._calls += 1
                #
                #         # theta (proficiency params) are sampled from a normal distribution
                #         theta = pymc.Normal("theta", mu=0, tau=1)
                #
                #         # question-parameters (IRT params) are sampled from normal distributions (though others were tried)
                #         # (note that the mean for the discrimination parameters isn't 0, since in general questions will be somewhat diagnostic)
                #         a = pymc.Normal("a", mu=1, tau=1, value=administered_items[:, 0])
                #         # a = Exponential("a", beta=0.01, value=[[0.0] * numthetas] * numquestions)
                #         b = pymc.Normal("b", mu=0, tau=1, value=[0.0] * administered_items[:, 1])
                #         c = pymc.Normal("c", mu=0.18, tau=0.025, value=[0.0] * administered_items[:, 2])
                #
                #         # a = administered_items[:, 0]
                #         # b = administered_items[:, 1]
                #         # c = administered_items[:, 2]
                #
                #         # take vectors theta/a/b/c, return a vector of probabilities of each person getting each question correct
                #         @pymc.deterministic
                #         def sigmoid(theta=theta, a=a, b=b, c=c):
                #             return c + ((1 - c) / (1 + pymc.exp(-a * (theta - b))))
                #
                #         # take the probabilities coming out of the sigmoid, and flip weighted coins
                #         correct = pymc.Bernoulli('correct', p=sigmoid, value=response_vector, observed=True)
                #
                #         # create a pymc simulation object, including all the above variables
                #         m = pymc.MCMC([a, b, theta, sigmoid, correct])
                #
                #         # run an interactive MCMC sampling session
                #         print(m.isample(iter=20000, burn=15000))
                #
                #         # prob_dist = pymc.Normal('thetas', mu=0, tau=1)
                #         #
                #         # obj = pymc.Deterministic(
                #         #     eval=irt.logLik,
                #         #     name='loglikelihood',
                #         #     parents={
                #         #         'est_theta': prob_dist,
                #         #         'response_vector': response_vector,
                #         #         'administered_items': administered_items
                #         #     },
                #         #     dtype=float,
                #         #     doc='IRT log-likelihood function'
                #         # )
                #         #
                #         # M = pymc.MAP(obj)
                #         # M.fit()
                #         # pymc.Matplot.plot(M)
                #         # print(M.prob_dist.value)
                #         #
                #         # return M.prob_dist.value
