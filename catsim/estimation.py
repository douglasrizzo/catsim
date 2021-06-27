from abc import ABC, abstractmethod
from typing import List

import numpy
from scipy.optimize import minimize_scalar

from catsim import cat, irt
from catsim.simulation import Estimator


class NumericalSearchEstimator(Estimator):
    methods = [
        "hillclimbing",
        "ternary",
        "dichotomous",
        "fibonacci",
        "golden",
        "brent",
        "bounded",
        "golden2",
    ]
    golden_ratio = (1 + 5**0.5) / 2

    def __str__(self):
        return "Numerical Search Estimator ({})".format(self.__search_method)

    def __init__(
        self,
        precision: int = 6,
        dodd: bool = True,
        verbose: bool = False,
        method="bounded",
    ):
        """Estimator that implements multiple search algorithms in unimodal functions to find the maximum of the log-likelihood function.

        There are implementations of ternary search, dichotomous search, Fibonacci search and golden-section search, according to [Vel20]_. Also check [Brent02]_.

        It is also possible to use the methods from :py:func:`scipy.optimize.minimize_scalar`.

        Lastly, there is an implementation of a simple hill climbing algorithm.

        :param precision: number of decimal points of precision, defaults to 6
        :type precision: int, optional
        :param dodd: whether to employ Dodd's estimation heuristic when the response vector only has one kind of response (all correct or all incorrect), defaults to True
        :type dodd: bool, optional
        :param verbose: verbosity level of the maximization method
        :type verbose: bool, optional
        :param method: the search method to employ, one of `'hillclimbing'`, `'ternary'`, `'dichotomous'`, `'fibonacci'`, `'golden'`, `'brent'`, `'bounded'` and `'golden2'`, defaults to bounded
        :type method: str, optional
        """
        super().__init__(verbose)

        if precision < 1:
            raise ValueError(
                "precision for numerical estimator must be an integer larger than 1, {} was passed".
                format(precision)
            )

        if method not in NumericalSearchEstimator.methods:
            raise ValueError(
                "Parameter 'method' must be one of {}".format(NumericalSearchEstimator.methods)
            )

        self._epsilon = float("1e-" + str(precision))
        self._dodd = dodd
        self.__search_method = method

    def estimate(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: List[int] = None,
        response_vector: List[bool] = None,
        est_theta: float = None,
        **kwargs
    ) -> float:
        """Returns the theta value that maximizes the log-likelihood function, given the current state of the
         test for the given examinee.

        :param index: index of the current examinee in the simulator
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param response_vector: a boolean list containing the examinee's answers to the administered items
        :param est_theta: a float containing the current estimated proficiency
        :returns: the current :math:`\\hat\\theta`
        """
        items, administered_items, response_vector, est_theta = self._prepare_args(
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

        summarized_answers = set(response_vector)

        # enter here if examinee has only answered correctly or incorrectly
        if len(summarized_answers) == 1:
            answer = summarized_answers.pop()

            # if the estimator was initialized with dodd = True,
            # use Dodd's estimation heuristic to return a theta value
            if self._dodd:
                candidate_theta = cat.dodd(est_theta, items, answer)

            # otherwise, return positive or negative infinity,
            # in accordance with the definition of the MLE
            elif answer:
                candidate_theta = float("inf")
            else:
                candidate_theta = float("-inf")

            return candidate_theta

        # select lower and upper bounds for an interval in which the estimator will
        # look for the most probable new theta

        # these bounds are computed as a the minimum and maximum item difficulties
        # in the bank...
        lower_bound = min(items[:, 1])
        upper_bound = max(items[:, 1])

        # ... plus an arbitrary error margin
        margin = (upper_bound - lower_bound) / 3
        upper_bound += margin
        lower_bound -= margin

        if self.__search_method == "hillclimbing":
            candidate_theta = self._solve_hill_climbing(
                upper_bound, lower_bound, response_vector, items, administered_items
            )
        elif self.__search_method in ["ternary", "dichotomous"]:
            candidate_theta = self._solve_ternary_dichotomous(
                upper_bound, lower_bound, response_vector, items, administered_items
            )
        elif self.__search_method == "fibonacci":
            candidate_theta = self._solve_fibonacci(
                upper_bound, lower_bound, response_vector, items, administered_items
            )
        elif self.__search_method == "golden2":
            candidate_theta = self._solve_golden_section(
                upper_bound, lower_bound, response_vector, items, administered_items
            )
        elif self.__search_method in ["brent", "bounded", "golden"]:
            res = minimize_scalar(
                irt.negative_log_likelihood,
                bracket=(lower_bound, upper_bound),
                bounds=(lower_bound, upper_bound),
                method=self.__search_method,
                args=(response_vector, items[administered_items]),
                tol=self._epsilon if self.__search_method != "bounded" else None,
            )
            self._evaluations = res.nfev
            candidate_theta = res.x

        if self._verbose:
            print("{0} evaluations".format(self._evaluations))

        return candidate_theta

    def _solve_hill_climbing(
        self, upper_bound, lower_bound, response_vector, items, administered_items
    ):
        best_theta = float("-inf")
        max_ll = float("-inf")

        # the estimator starts with a rough search inside the interval, which gets
        # finer with each pass
        for granularity in range(10):

            # generate a discrete list of candidate theta values inside the interval
            candidates = numpy.linspace(lower_bound, upper_bound, 9)
            interval_size = candidates[1] - candidates[0]

            if self._verbose:
                print(
                    "Pass: {0}\n\tBounds: {1} {2}\n\tInterval size: {3}".format(
                        granularity + 1, lower_bound, upper_bound, interval_size
                    )
                )

            # we'll use the concave nature of the log-likelihood function
            # to program a primitive early stopping method in our search.
            previous_ll = float("-inf")

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

                # check if the LL of the current candidate theta is larger than
                # the best one checked as of yet
                if current_ll > max_ll:
                    if self._verbose:
                        print("\t\tTheta: {0}, LL: {1}".format(candidate_theta, current_ll))

                    # if the difference between the best theta as of yet and the current theta
                    # is negligible, we stop our search and return the current theta, which is
                    # better than the as-of-yet best theta
                    if abs(best_theta - candidate_theta) < self._epsilon:
                        best_theta = candidate_theta
                        break

                    max_ll = current_ll
                    best_theta = candidate_theta

            # the bounds of the new candidates are adjusted around the current best theta value
            lower_bound = best_theta - (interval_size * 2)
            upper_bound = best_theta + (interval_size * 2)
        return best_theta

    def _solve_ternary_dichotomous(self, b, a, response_vector, items, administered_items):
        error = float("inf")
        while error >= self._epsilon:
            self._evaluations += 2

            if self.__search_method == "ternary":
                c = (b + 2 * a) / 3
                d = (2 * b + a) / 3
            elif self.__search_method == "dichotomous":
                m = (a + b) / 2
                c = m - (self._epsilon / 2)
                d = m + (self._epsilon / 2)

            left_side_ll = irt.log_likelihood(c, response_vector, items[administered_items])
            right_side_ll = irt.log_likelihood(d, response_vector, items[administered_items])

            if left_side_ll >= right_side_ll:
                b = d
            else:
                a = c

            assert a <= c <= d <= b

            candidate_theta = (b + a) / 2

            error = abs(b - a)
            if self.__search_method == "dichotomous":
                error /= 2

            if self._verbose:
                print(
                    "\t\tTheta: {0}, LL: {1}".format(
                        candidate_theta, max(left_side_ll, right_side_ll)
                    )
                )
        return candidate_theta

    def _solve_fibonacci(self, b, a, response_vector, items, administered_items):
        fib = [1, 1]
        n = 1

        # while (upper_bound - lower_bound) / fib[-1] > .001:
        while (b - a) / fib[-1] > self._epsilon:
            n += 1
            fib.append(fib[-1] + fib[-2])

        c = a + (fib[n - 2] / fib[n]) * (b - a)
        d = a + (fib[n - 1] / fib[n]) * (b - a)

        left_side_ll = irt.log_likelihood(c, response_vector, items[administered_items])
        right_side_ll = irt.log_likelihood(d, response_vector, items[administered_items])
        self._evaluations += 2

        while n != 2:
            self._evaluations += 1

            n -= 1

            if left_side_ll >= right_side_ll:
                b = d
                d = c
                c = a + (fib[n - 2] / fib[n]) * (b - a)

                right_side_ll = left_side_ll
                left_side_ll = irt.log_likelihood(c, response_vector, items[administered_items])
            else:
                a = c
                c = d
                d = a + (fib[n - 1] / fib[n]) * (b - a)

                left_side_ll = right_side_ll
                right_side_ll = irt.log_likelihood(d, response_vector, items[administered_items])

            # assert a <= c <= d <= b

            if self._verbose:
                print(
                    "\t\tTheta: {0}, LL: {1}".format((b + a) / 2, max(left_side_ll, right_side_ll))
                )
        return (b + a) / 2

    def _solve_golden_section(self, b, a, response_vector, items, administered_items):
        c = b + (a - b) / NumericalSearchEstimator.golden_ratio
        d = a + (b - a) / NumericalSearchEstimator.golden_ratio

        left_side_ll = irt.log_likelihood(c, response_vector, items[administered_items])
        right_side_ll = irt.log_likelihood(d, response_vector, items[administered_items])

        while abs(b - a) > self._epsilon:
            self._evaluations += 1

            if left_side_ll >= right_side_ll:
                b = d
                d = c
                c = b + (a - b) / NumericalSearchEstimator.golden_ratio

                right_side_ll = left_side_ll
                left_side_ll = irt.log_likelihood(c, response_vector, items[administered_items])
            else:
                a = c
                c = d
                d = a + (b - a) / NumericalSearchEstimator.golden_ratio

                left_side_ll = right_side_ll
                right_side_ll = irt.log_likelihood(d, response_vector, items[administered_items])

            assert a < c <= d < b

            if self._verbose:
                print(
                    "\t\tTheta: {0}, LL: {1}".format((b + a) / 2, max(left_side_ll, right_side_ll))
                )
        return (b + a) / 2

    @property
    def dodd(self) -> bool:
        """Whether Dodd's method will be called by estimator in case the response vector
        is composed solely of right or wrong answers.

        :returns: boolean value indicating if Dodd's method will be used or not."""
        return self._dodd

    @property
    def method(self) -> str:
        """Get the estimator search method selected during instantiation.

        :returns: search method"""
        return self.__search_method
