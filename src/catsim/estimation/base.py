"""Base class for ability estimators."""

import math
from abc import ABC, abstractmethod

from ..context import StepContext


class BaseEstimator(ABC):
  """Base class for ability estimators.

  An estimator updates the ability estimate from the responses a
  :class:`~catsim.context.StepContext` holds, the newest one included.

  An estimator holds its configuration and caches derived from the item bank, never per-session
  state: one estimator serves every session of a run, an application may interleave the sessions
  of several examinees on it, and look-ahead selectors such as MEI call it with hypothetical
  responses.
  """

  def __init__(self) -> None:
    """Initialize the evaluation counters."""
    self._calls = 0
    self._last_evaluations = 0
    self._total_evaluations = 0

  @abstractmethod
  def estimate(self, context: StepContext) -> float:
    r"""Return the updated ability estimate given the responses so far.

    Once an item has been administered, the estimate is finite and lies within the estimator's
    bounds, whatever the response pattern, all correct and all incorrect included.

    Parameters
    ----------
    context : StepContext
        The session to estimate for, with the newest item and response, and with ``theta`` the
        estimate before them.

    Returns
    -------
    float
        The updated ability estimate :math:`\hat\theta`.
    """

  @staticmethod
  def _check_tol(tol: float) -> float:
    """Return ``tol`` as a float, after checking that it is positive.

    Parameters
    ----------
    tol : float
        The tolerance of the estimator's numerical search.

    Returns
    -------
    float
        The tolerance, as a float.

    Raises
    ------
    ValueError
        If ``tol`` is not positive (NaN included).
    """
    if not tol > 0:
      msg = f"tol must be positive, got {tol}"
      raise ValueError(msg)
    return float(tol)

  @staticmethod
  def _check_bounds(bounds: tuple[float, float]) -> tuple[float, float]:
    """Return ``bounds`` as floats, after checking that they are finite with low < high.

    Parameters
    ----------
    bounds : tuple[float, float]
        The lower and upper bound of the estimate.

    Returns
    -------
    tuple[float, float]
        The bounds, as floats.

    Raises
    ------
    ValueError
        If either bound is not finite, or if the lower bound is not below the upper one.
    """
    low, high = bounds
    if not (math.isfinite(low) and math.isfinite(high) and low < high):
      msg = f"bounds must be finite with low < high, got {bounds}"
      raise ValueError(msg)
    return float(low), float(high)

  @property
  def calls(self) -> int:
    """Get how many times the estimator has been called to maximize/minimize the log-likelihood function.

    Returns
    -------
    int
        Number of times the estimator has been called to maximize/minimize the
        log-likelihood function.
    """
    return self._calls

  @property
  def evaluations(self) -> int:
    """Get the total number of times the estimator has evaluated the log-likelihood function during its existence.

    Returns
    -------
    int
        Number of function evaluations in the most recent estimate call.
    """
    return self._last_evaluations

  @property
  def total_evaluations(self) -> int:
    """Get the total number of evaluations across the estimator lifetime."""
    return self._total_evaluations

  @property
  def avg_evaluations(self) -> float:
    """Get the average number of function evaluations for all tests the estimator has been used.

    Returns
    -------
    float
        Average number of function evaluations per estimate call.
    """
    if self._calls == 0:
      return 0.0
    return self._total_evaluations / self._calls
