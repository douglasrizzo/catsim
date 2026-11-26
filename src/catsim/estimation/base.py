"""Base class for ability estimators."""

import math
from abc import ABC, abstractmethod

from ..context import Estimate, StepContext


class BaseEstimator(ABC):
  """Base class for ability estimators.

  An estimator updates the ability estimate from the responses a :class:`~catsim.StepContext`
  holds, the newest one included, and reports it as an :class:`~catsim.Estimate`: the estimate, its
  standard error by the estimator's own formula, which the stoppers judge, and how many objective
  evaluations it took.

  An estimator holds its configuration and caches derived from the item bank, never per-session
  state: one estimator serves every session of a run, which an application may interleave, and
  look-ahead selectors such as MEI call it with hypothetical responses. The :doc:`/extending` guide
  lists what an estimator receives and returns.
  """

  @abstractmethod
  def estimate(self, context: StepContext) -> Estimate:
    r"""Return the updated ability estimate given the responses so far, with its standard error.

    Once an item has been administered, the estimate is finite and lies within the estimator's
    bounds, whatever the responses, all correct and all incorrect included.

    Parameters
    ----------
    context : StepContext
        The session to estimate for, with the newest item and response, and with ``theta`` the
        estimate before them. When :meth:`CatEngine.answer <catsim.CatEngine.answer>` calls the
        estimator, the context's generator is the one given to ``answer``.

    Returns
    -------
    Estimate
        The updated ability estimate :math:`\hat\theta`, its standard error (positive, possibly
        ``math.inf``) and the number of objective evaluations it took.
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
