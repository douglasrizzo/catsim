"""Stop a test once the ability estimate is precise enough."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy import stats

from .base import BaseStopper

if TYPE_CHECKING:
  from collections.abc import Sequence

  from ..context import StepContext


@dataclass(frozen=True)
class StandardErrorStopper(BaseStopper):
  """Stop once the standard error of the ability estimate is at most a target value.

  The standard error is the one the estimator reported with the current estimate,
  ``context.se`` (ADR-0004): the posterior standard deviation for EAP, the Fisher standard error
  for maximum likelihood. Before any response it is infinite, so the test does not stop.

  Parameters
  ----------
  max_se : float
      Largest standard error at which the test stops. Must be positive. Smaller values ask
      for more precision, and so for more items.

  Raises
  ------
  ValueError
      If `max_se` is not positive (or is NaN).

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/stopping/standard-error-stopper`.
  """

  max_se: float

  def __post_init__(self) -> None:
    """Validate the target standard error."""
    # written so that NaN, which no standard error is ever at most, fails the check too
    if not self.max_se > 0:
      msg = f"max_se must be positive, got {self.max_se}"
      raise ValueError(msg)

  def stop(self, context: StepContext) -> bool:
    """Return whether the standard error at the current estimate is at most `max_se`.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if the standard error of the estimate is at most `max_se`, False otherwise.
    """
    return context.se <= self.max_se


@dataclass(frozen=True)
class ConfidenceIntervalStopper(BaseStopper):
  r"""Stop once the confidence interval for the ability estimate falls within one ability band.

  The bounds cut the ability scale into bands. For example, ``[-2.0, -0.5, 0.5, 2.0]``
  gives five: :math:`(-\infty, -2.0]`, :math:`[-2.0, -0.5]`, :math:`[-0.5, 0.5]`,
  :math:`[0.5, 2.0]` and :math:`[2.0, \infty)`. The test stops once no bound lies strictly
  inside the interval :math:`\hat\theta \pm z \cdot SE`, where :math:`SE` is the standard error
  the estimator reported, ``context.se`` (ADR-0004), and :math:`z` the standard normal quantile
  of the confidence level. Before any response the standard error is infinite, so the test does
  not stop.

  Parameters
  ----------
  interval_bounds : Sequence[float]
      Band boundaries on the ability scale, in strictly increasing order. Must contain at least one
      value. Stored as a tuple.
  confidence : float, optional
      Confidence level of the interval, between 0 and 1. Default is 0.95.

  Raises
  ------
  ValueError
      If `interval_bounds` is empty or not strictly increasing, or if `confidence` is not
      between 0 and 1.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/stopping/confidence-interval-stopper`.
  """

  interval_bounds: Sequence[float]
  confidence: float = 0.95

  def __post_init__(self) -> None:
    """Validate the bounds and the confidence level, and store the bounds as a tuple."""
    bounds = tuple(self.interval_bounds)
    if not bounds:
      msg = "interval_bounds must contain at least one value"
      raise ValueError(msg)
    if any(low >= high for low, high in itertools.pairwise(bounds)):
      msg = "interval_bounds must be strictly increasing"
      raise ValueError(msg)
    if not 0 < self.confidence < 1:
      msg = f"confidence must be between 0 and 1, got {self.confidence}"
      raise ValueError(msg)
    # a tuple keeps the frozen stopper hashable and its bands unchangeable
    object.__setattr__(self, "interval_bounds", bounds)

  def stop(self, context: StepContext) -> bool:
    """Return whether the confidence interval at the current estimate lies within one ability band.

    Parameters
    ----------
    context : StepContext
        The session to check.

    Returns
    -------
    bool
        True if no bound lies strictly inside the confidence interval, False otherwise.
    """
    margin = float(stats.norm.ppf((1 + self.confidence) / 2)) * context.se
    lower, upper = context.theta - margin, context.theta + margin
    return not any(lower < bound < upper for bound in self.interval_bounds)
