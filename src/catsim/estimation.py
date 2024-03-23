from typing import Any

import numpy
from scipy.optimize import minimize_scalar

from catsim import cat, irt
from catsim.simulation import Estimator


class NumericalSearchEstimator(Estimator):
  """Class that implements search algorithms in unimodal functions to find the maximum of the log-likelihood function.

  There are implementations of ternary search, dichotomous search, Fibonacci search and golden-section search, according
  to [Veliz20]_. Also check [Brent02]_. It is also possible to use the methods from
  :py:func:`scipy.optimize.minimize_scalar`.

  :param precision: number of decimal points of precision, defaults to 6
  :type precision: int, optional
  :param dodd: whether to employ Dodd's estimation heuristic [Dod90]_ when the response vector only has one kind of
  response (all correct or all incorrect, see :py:func:`catsim.cat.dodd`), defaults to True
  :type dodd: bool, optional
  :param verbose: verbosity level of the maximization method
  :type verbose: bool, optional
  :param method: the search method to employ, one of `'ternary'`, `'dichotomous'`, `'fibonacci'`, `'golden'`, `'brent'`,
  `'bounded'` and `'golden2'`, defaults to bounded
  :type method: str, optional
  """

  __methods = frozenset(["ternary", "dichotomous", "fibonacci", "golden", "brent", "bounded", "golden2"])

  golden_ratio = (1 + 5**0.5) / 2

  @staticmethod
  def methods() -> frozenset[str]:
    """Get a set of available estimation methods.

    :return: Set of available estimation methods.
    :rtype: set[str]
    """
    return NumericalSearchEstimator.__methods

  def __str__(self) -> str:
    """Return a string representation of the estimator."""
    return f"Numerical Search Estimator ({self.__search_method})"

  def __init__(
    self,
    tol: int = 1e-6,
    dodd: bool = True,
    verbose: bool = False,
    method: str = "bounded",
  ) -> None:
    """Initialize the estimator.

    :param tol: _description_, defaults to 1e-6
    :type tol: float, optional
    :param dodd: _description_, defaults to True
    :type dodd: bool, optional
    :param verbose: _description_, defaults to False
    :type verbose: bool, optional
    :param method: _description_, defaults to "bounded"
    :type method: str, optional
    :raises ValueError: _description_
    :raises ValueError: _description_
    """
    super().__init__(verbose)

    if method not in NumericalSearchEstimator.__methods:
      msg = f"Parameter 'method' must be one of {NumericalSearchEstimator.__methods}."
      raise ValueError(msg)

    self._tol = tol
    self._dodd = dodd
    self.__search_method = method

  def estimate(
    self,
    index: int | None = None,
    items: numpy.ndarray = None,
    administered_items: list[int] | None = None,
    response_vector: list[bool] | None = None,
    est_theta: float | None = None,
    **kwargs: dict[str, Any],
  ) -> float:
    r"""Compute the theta value that maximizes the log-likelihood function for the given examinee in a test.

    When this method is used inside a simulator, its arguments are automatically filled. Outside of a simulation, the
    user can also specify the arguments to use the Estimator as a standalone object.

    :param index: index of the current examinee in the simulator
    :param items: a matrix containing item parameters in the format that `catsim` understands
                  (see: :py:func:`catsim.cat.generate_item_bank`)
    :param administered_items: a list containing the indexes of items that were already administered
    :param response_vector: a boolean list containing the examinee's answers to the administered items
    :param est_theta: a float containing the current estimated ability
    :returns: the current :math:`\hat\theta`
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
      **kwargs,
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

    if self.__search_method in {"ternary", "dichotomous"}:
      candidate_theta = self._solve_ternary_dichotomous(
        upper_bound, lower_bound, response_vector, items[administered_items]
      )
    elif self.__search_method == "fibonacci":
      candidate_theta = self._solve_fibonacci(upper_bound, lower_bound, response_vector, items[administered_items])
    elif self.__search_method == "golden2":
      candidate_theta = self._solve_golden_section(upper_bound, lower_bound, response_vector, items[administered_items])
    elif self.__search_method in {"brent", "bounded", "golden"}:
      res = minimize_scalar(
        irt.negative_log_likelihood,
        bracket=(lower_bound, upper_bound),
        bounds=(lower_bound, upper_bound) if self.__search_method == "bounded" else None,
        method=self.__search_method,
        args=(response_vector, items[administered_items]),
        tol=self._tol if self.__search_method != "bounded" else None,
      )
      self._evaluations = res.nfev
      candidate_theta = res.x

    if self._verbose:
      print(f"{self._evaluations} evaluations")

    return candidate_theta

  def _solve_ternary_dichotomous(
    self,
    b: float,
    a: float,
    response_vector: list[bool],
    item_params: numpy.ndarray,
  ) -> float:
    """Find the most likely ability for a given response vector, using the ternary or dichotomous search methods.

    :param upper_bound: the upper bound to search for the ability, in the ability/difficulty scale
    :type upper_bound: float
    :param lower_bound: the lower bound to search for the ability, in the ability/difficulty scale
    :type lower_bound: float
    :param response_vector: the responses given to the answered items
    :type response_vector: List[bool]
    :param item_params: the parameter matrix of the answered items
    :type item_params: numpy.ndarray
    :return: the estimated ability
    :rtype: float
    """
    error = float("inf")
    while error >= self._tol:
      self._evaluations += 2

      if self.__search_method == "ternary":
        c = (b + 2 * a) / 3
        d = (2 * b + a) / 3
      elif self.__search_method == "dichotomous":
        m = (a + b) / 2
        c = m - (self._tol / 2)
        d = m + (self._tol / 2)

      left_side_ll = irt.log_likelihood(c, response_vector, item_params)
      right_side_ll = irt.log_likelihood(d, response_vector, item_params)

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
        print(f"\t\tTheta: {candidate_theta}, LL: {max(left_side_ll, right_side_ll)}")
    return candidate_theta

  def _solve_fibonacci(
    self,
    b: float,
    a: float,
    response_vector: list[bool],
    item_params: numpy.ndarray,
  ) -> float:
    """Find the most likely ability for a given response vector, using the Fibonacci search method.

    :param upper_bound: the upper bound to search for the ability, in the ability/difficulty scale
    :type upper_bound: float
    :param lower_bound: the lower bound to search for the ability, in the ability/difficulty scale
    :type lower_bound: float
    :param response_vector: the responses given to the answered items
    :type response_vector: List[bool]
    :param item_params: the parameter matrix of the answered items
    :type item_params: numpy.ndarray
    :return: the estimated ability
    :rtype: float
    """
    fib = [1, 1]
    n = 1

    # while (upper_bound - lower_bound) / fib[-1] > .001:
    while (b - a) / fib[-1] > self._tol:
      n += 1
      fib.append(fib[-1] + fib[-2])

    c = a + (fib[n - 2] / fib[n]) * (b - a)
    d = a + (fib[n - 1] / fib[n]) * (b - a)

    left_side_ll = irt.log_likelihood(c, response_vector, item_params)
    right_side_ll = irt.log_likelihood(d, response_vector, item_params)
    self._evaluations += 2

    while n != 2:  # noqa: PLR2004
      self._evaluations += 1

      n -= 1

      if left_side_ll >= right_side_ll:
        b = d
        d = c
        c = a + (fib[n - 2] / fib[n]) * (b - a)

        right_side_ll = left_side_ll
        left_side_ll = irt.log_likelihood(c, response_vector, item_params)
      else:
        a = c
        c = d
        d = a + (fib[n - 1] / fib[n]) * (b - a)

        left_side_ll = right_side_ll
        right_side_ll = irt.log_likelihood(d, response_vector, item_params)

      # assert a <= c <= d <= b

      if self._verbose:
        print(f"\t\tTheta: {(b + a) / 2}, LL: {max(left_side_ll, right_side_ll)}")
    return (b + a) / 2

  def _solve_golden_section(
    self,
    b: float,
    a: float,
    response_vector: list[bool],
    item_params: numpy.ndarray,
  ) -> float:
    """Find the most likely ability for a given response vector, using the golden-section search method.

    :param upper_bound: the upper bound to search for the ability, in the ability/difficulty scale
    :type upper_bound: float
    :param lower_bound: the lower bound to search for the ability, in the ability/difficulty scale
    :type lower_bound: float
    :param response_vector: the responses given to the answered items
    :type response_vector: List[bool]
    :param item_params: the parameter matrix of the answered items
    :type item_params: numpy.ndarray
    :return: the estimated ability
    :rtype: float
    """
    c = b + (a - b) / NumericalSearchEstimator.golden_ratio
    d = a + (b - a) / NumericalSearchEstimator.golden_ratio

    left_side_ll = irt.log_likelihood(c, response_vector, item_params)
    right_side_ll = irt.log_likelihood(d, response_vector, item_params)

    while abs(b - a) > self._tol:
      self._evaluations += 1

      if left_side_ll >= right_side_ll:
        b = d
        d = c
        c = b + (a - b) / NumericalSearchEstimator.golden_ratio

        right_side_ll = left_side_ll
        left_side_ll = irt.log_likelihood(c, response_vector, item_params)
      else:
        a = c
        c = d
        d = a + (b - a) / NumericalSearchEstimator.golden_ratio

        left_side_ll = right_side_ll
        right_side_ll = irt.log_likelihood(d, response_vector, item_params)

      assert a < c <= d < b

      if self._verbose:
        print(f"\t\tTheta: {(b + a) / 2}, LL: {max(left_side_ll, right_side_ll)}")
    return (b + a) / 2

  @property
  def dodd(self) -> bool:
    """Whether Dodd's estimation heuristic [Dod90]_ will be used by the estimator.

    Dodd's method is used when the response vector is composed solely of right or wrong answers, to prevent maximum
    likelihood methods to return -infinity or + infinity.

    :returns: boolean value indicating if Dodd's method will be used or not.
    :see: :py:func:`catsim.cat.dodd`
    """
    return self._dodd

  @property
  def method(self) -> str:
    """Get the estimator search method selected during instantiation.

    :returns: search method
    """
    return self.__search_method
