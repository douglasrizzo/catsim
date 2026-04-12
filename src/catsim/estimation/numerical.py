"""Numerical estimation methods for ability estimation."""

import numpy
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from .. import cat, irt
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..item_bank import ItemBank
from .base import BaseEstimator

# Mathematical constants
GOLDEN_RATIO = (1 + 5**0.5) / 2


class NumericalSearchEstimator(BaseEstimator):
  """Estimate ability by numerically maximizing the IRT log-likelihood.

  Parameters
  ----------
  tol : float, optional
      Tolerance for convergence in the optimization algorithm. Default is 1e-6.
  dodd : bool, optional
      Whether to employ Dodd's estimation heuristic :cite:p:`Dod90` when the response vector
      only has one kind of response (all correct or all incorrect, see
      :py:func:`catsim.cat.dodd`). Default is True.
  verbose : bool, optional
      Whether to print detailed information during optimization. Default is False.
  method : str, optional
      The search method to employ. Must be one of: 'ternary', 'dichotomous', 'fibonacci',
      'golden', 'brent', 'bounded', or 'golden2'. Default is 'bounded'.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/estimation/numerical-search-estimator`.
  """

  __methods = frozenset(["ternary", "dichotomous", "fibonacci", "golden", "brent", "bounded", "golden2"])

  @staticmethod
  def available_methods() -> frozenset[str]:
    """Get a set of available estimation methods.

    Returns
    -------
    frozenset[str]
        Set of available estimation methods: {'ternary', 'dichotomous', 'fibonacci',
        'golden', 'brent', 'bounded', 'golden2'}.
    """
    return NumericalSearchEstimator.__methods

  def __str__(self) -> str:
    """Return a string representation of the estimator."""
    return f"Numerical Search Estimator ({self.__search_method})"

  def __init__(
    self,
    tol: float = 1e-6,
    dodd: bool = True,
    verbose: bool = False,
    method: str = "bounded",
  ) -> None:
    """Initialize the estimator.

    Parameters
    ----------
    tol : float, optional
        Tolerance for convergence in the optimization algorithm. Default is 1e-6.
    dodd : bool, optional
        Whether to use Dodd's estimation heuristic for edge cases (all correct or
        all incorrect responses). Default is True.
    verbose : bool, optional
        Whether to print detailed information during optimization. Default is False.
    method : str, optional
        The numerical search method to use for optimization. Default is "bounded".

    Raises
    ------
    ValueError
        If the parameter `method` is not one of the available methods.
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
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,
  ) -> float:
    r"""Compute the theta value that maximizes the log-likelihood function for the given examinee.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    response_vector : list[bool]
        A boolean list containing the examinee's answers to the administered items.
    est_theta : float
        The current estimated ability.

    Returns
    -------
    float
        The current estimated ability :math:`\hat\theta`.

    Raises
    ------
    ValueError
        If required parameters are missing.
    """
    if item_bank is None:
      msg = "item_bank parameter cannot be None"
      raise ValueError(msg)
    if administered_items is None:
      msg = "administered_items parameter cannot be None"
      raise ValueError(msg)
    if response_vector is None:
      msg = "response_vector parameter cannot be None"
      raise ValueError(msg)

    self._calls += 1
    self._last_evaluations = 0

    summarized_answers = set(response_vector)

    # enter here if examinee has only answered correctly or incorrectly
    if len(summarized_answers) == 1:
      answer = summarized_answers.pop()

      # if the estimator was initialized with dodd = True,
      # use Dodd's estimation heuristic to return a theta value
      if self._dodd:
        candidate_theta = cat.dodd(est_theta, item_bank, answer)

      # otherwise, return positive or negative infinity,
      # in accordance with the definition of the MLE
      elif answer:
        candidate_theta = float("inf")
      else:
        candidate_theta = float("-inf")

      return candidate_theta

    if self.__search_method in {"ternary", "dichotomous"}:
      candidate_theta = self._solve_ternary_dichotomous(
        THETA_MAX_EXTENDED, THETA_MIN_EXTENDED, response_vector, item_bank.get_items(administered_items)
      )
    elif self.__search_method == "fibonacci":
      candidate_theta = self._solve_fibonacci(
        THETA_MAX_EXTENDED, THETA_MIN_EXTENDED, response_vector, item_bank.get_items(administered_items)
      )
    elif self.__search_method == "golden2":
      candidate_theta = self._solve_golden_section(
        THETA_MAX_EXTENDED, THETA_MIN_EXTENDED, response_vector, item_bank.get_items(administered_items)
      )
    elif self.__search_method in {"brent", "bounded", "golden"}:
      res = minimize_scalar(
        irt.negative_log_likelihood,
        bracket=(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
        bounds=(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED) if self.__search_method == "bounded" else None,
        method=self.__search_method,
        args=(response_vector, item_bank.get_items(administered_items)),
        tol=self._tol if self.__search_method != "bounded" else None,
      )
      self._last_evaluations = res.nfev
      candidate_theta = res.x

    self._total_evaluations += self._last_evaluations

    if self._verbose:
      print(f"{self._last_evaluations} evaluations")

    return candidate_theta

  def _solve_ternary_dichotomous(
    self,
    b: float,
    a: float,
    response_vector: list[bool],
    item_params: npt.NDArray[numpy.floating],
  ) -> float:
    """Find the most likely ability using ternary or dichotomous search methods.

    Parameters
    ----------
    b : float
        The upper bound to search for the ability, in the ability/difficulty scale.
    a : float
        The lower bound to search for the ability, in the ability/difficulty scale.
    response_vector : list[bool]
        The responses given to the answered items.
    item_params : numpy.ndarray
        The parameter matrix of the answered items.

    Returns
    -------
    float
        The estimated ability.

    Raises
    ------
    ValueError
        If the search interval becomes invalid during iteration.
    """
    error = float("inf")
    while error >= self._tol:
      self._last_evaluations += 2

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

      if not (a <= c <= d <= b):
        msg = f"Invalid interval: a={a}, c={c}, d={d}, b={b}"
        raise ValueError(msg)

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
    item_params: npt.NDArray[numpy.floating],
  ) -> float:
    """Find the most likely ability using the Fibonacci search method.

    Parameters
    ----------
    b : float
        The upper bound to search for the ability, in the ability/difficulty scale.
    a : float
        The lower bound to search for the ability, in the ability/difficulty scale.
    response_vector : list[bool]
        The responses given to the answered items.
    item_params : numpy.ndarray
        The parameter matrix of the answered items.

    Returns
    -------
    float
        The estimated ability.
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
    self._last_evaluations += 2

    while n != 2:  # noqa: PLR2004
      self._last_evaluations += 1

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
    item_params: npt.NDArray[numpy.floating],
  ) -> float:
    """Find the most likely ability using the golden-section search method.

    Parameters
    ----------
    b : float
        The upper bound to search for the ability, in the ability/difficulty scale.
    a : float
        The lower bound to search for the ability, in the ability/difficulty scale.
    response_vector : list[bool]
        The responses given to the answered items.
    item_params : numpy.ndarray
        The parameter matrix of the answered items.

    Returns
    -------
    float
        The estimated ability.

    Raises
    ------
    ValueError
        If the golden section interval becomes invalid during iteration.
    """
    c = b + (a - b) / GOLDEN_RATIO
    d = a + (b - a) / GOLDEN_RATIO

    left_side_ll = irt.log_likelihood(c, response_vector, item_params)
    right_side_ll = irt.log_likelihood(d, response_vector, item_params)
    self._last_evaluations += 2

    while abs(b - a) > self._tol:
      self._last_evaluations += 1

      if left_side_ll >= right_side_ll:
        b = d
        d = c
        c = b + (a - b) / GOLDEN_RATIO

        right_side_ll = left_side_ll
        left_side_ll = irt.log_likelihood(c, response_vector, item_params)
      else:
        a = c
        c = d
        d = a + (b - a) / GOLDEN_RATIO

        left_side_ll = right_side_ll
        right_side_ll = irt.log_likelihood(d, response_vector, item_params)

      if not (a < c <= d < b):
        msg = f"Invalid golden section interval: a={a}, c={c}, d={d}, b={b}"
        raise ValueError(msg)

      if self._verbose:
        print(f"\t\tTheta: {(b + a) / 2}, LL: {max(left_side_ll, right_side_ll)}")
    return (b + a) / 2

  @property
  def dodd(self) -> bool:
    """Whether Dodd's estimation heuristic :cite:p:`Dod90` will be used by the estimator.

    Dodd's method is used when the response vector is composed solely of correct or
    incorrect answers, to prevent maximum likelihood methods from returning -infinity
    or +infinity.

    Returns
    -------
    bool
        Boolean value indicating if Dodd's method will be used or not.

    See Also
    --------
    catsim.cat.dodd : Implementation of Dodd's estimation heuristic.
    """
    return self._dodd

  @property
  def method(self) -> str:
    """Get the estimator search method selected during instantiation.

    Returns
    -------
    str
        The search method ('ternary', 'dichotomous', 'fibonacci', 'golden',
        'brent', 'bounded', or 'golden2').
    """
    return self.__search_method
