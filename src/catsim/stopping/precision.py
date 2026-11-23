"""Stop a test once the ability estimate is precise enough."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .. import irt
from .base import BaseStopper

if TYPE_CHECKING:
  from collections.abc import Sequence

  from ..item_bank import ItemBank


@dataclass(frozen=True)
class StandardErrorStopper(BaseStopper):
  """Stop once the standard error of the ability estimate is at most a target value.

  The standard error is :func:`catsim.irt.see` of the administered items at the current
  estimate. With no administered items it is infinite, so the test does not stop.

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

  def stop(self, item_bank: ItemBank, administered_items: list[int], theta: float) -> bool:
    """Check whether the standard error at `theta` is at most `max_se`.

    Parameters
    ----------
    item_bank : ItemBank
        Item bank used by the session.
    administered_items : list[int]
        Item indices already administered in the session.
    theta : float
        Current ability estimate.

    Returns
    -------
    bool
        True if the standard error of the estimate is at most `max_se`, False otherwise.
    """
    return irt.see(theta, item_bank.get_items(administered_items)) <= self.max_se


@dataclass(frozen=True)
class ConfidenceIntervalStopper(BaseStopper):
  r"""Stop once the confidence interval for the ability estimate falls within one ability band.

  The bounds cut the ability scale into bands. For example, ``[-2.0, -0.5, 0.5, 2.0]``
  gives five: :math:`(-\infty, -2.0]`, :math:`[-2.0, -0.5]`, :math:`[-0.5, 0.5]`,
  :math:`[0.5, 2.0]` and :math:`[2.0, \infty)`. The test stops once no bound lies strictly
  inside the interval from :func:`catsim.irt.confidence_interval`. With no administered
  items the interval is infinite, so the test does not stop.

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

  def stop(self, item_bank: ItemBank, administered_items: list[int], theta: float) -> bool:
    """Check whether the confidence interval at `theta` lies within one ability band.

    Parameters
    ----------
    item_bank : ItemBank
        Item bank used by the session.
    administered_items : list[int]
        Item indices already administered in the session.
    theta : float
        Current ability estimate.

    Returns
    -------
    bool
        True if no bound lies strictly inside the confidence interval, False otherwise.
    """
    lower, upper = irt.confidence_interval(theta, item_bank.get_items(administered_items), self.confidence)
    return not any(lower < bound < upper for bound in self.interval_bounds)
