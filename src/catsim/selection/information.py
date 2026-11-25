"""Information-based selectors: maximum information, Urry's rule and interval information."""

import math

import numpy
from scipy import stats

from .. import irt
from ..compat import override
from ..context import StepContext
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining


class MaxInfoSelector(BaseSelector):
  """Select the non-administered item with maximum information at the current theta estimate.

  Parameters
  ----------
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max will not
      be selected unless no other items are available. Default is 1 (no restriction).

  Raises
  ------
  ValueError
      If r_max is not in (0, 1].

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/max-info-selector`.
  """

  def __init__(self, r_max: float = 1) -> None:
    super().__init__()
    self._r_max = check_r_max(r_max)

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(r_max={self._r_max!r})"

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate for items the selector accepts.

    Returns
    -------
    float
        Maximum exposure rate for items the selector accepts.
    """
    return self._r_max

  @override
  def select(self, context: StepContext) -> int:
    """Return the most informative remaining item at the current estimate, among those below ``r_max``.

    When every remaining item is at or above ``r_max``, all of them are candidates.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.
    """
    candidates = below_cap(remaining(context), context.exposure_rates, self._r_max)
    return pick_best(candidates, context.item_bank.information(context.theta)[candidates], context.rng)


class UrrySelector(BaseSelector):
  """Select the remaining item whose difficulty parameter is closest to the current ability estimate.

  It applies no exposure cap: ``context.exposure_rates`` is ignored.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/urry-selector`.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}()"

  @override
  def select(self, context: StepContext) -> int:
    """Return the remaining item whose difficulty is closest to the current estimate.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.
    """
    candidates = remaining(context)
    difficulty = context.item_bank.difficulty[candidates]
    return pick_best(candidates, -numpy.abs(difficulty - context.theta), context.rng)


# Below this half-width the integral's antiderivative difference loses more precision to rounding than the
# information at the estimate, the interval's limit, loses to the interval's width (an O(width^2) error).
_MIN_HALF_WIDTH = math.sqrt(numpy.finfo(float).eps)


class IntervalInfoSelector(BaseSelector):
  r"""Select the item whose information, integrated over an interval around the estimate, is largest.

  By default the interval is a confidence interval for the estimate, built from the estimator's
  standard error (``context.se``) :cite:p:`Veerkamp1997`. It spans the whole ability scale before the
  first response and narrows as the standard error falls; a standard error of zero, or nearly zero,
  selects by the items' information at the estimate. Setting ``interval`` fixes the half-width. The
  selector applies no exposure cap: ``context.exposure_rates`` is ignored.

  Parameters
  ----------
  interval : float or None, optional
      Fixed half-width of the interval around the estimate. Default is None, which uses the
      confidence interval.
  confidence : float, optional
      Confidence level of the default interval. Default is 0.95.

  Raises
  ------
  ValueError
      If ``interval`` is not positive or ``confidence`` is not between 0 and 1.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/interval-info-selector`.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(interval={self._interval!r}, confidence={self._confidence!r})"

  def __init__(self, interval: float | None = None, confidence: float = 0.95) -> None:
    if interval is not None and interval <= 0:
      msg = f"interval must be positive, got {interval}"
      raise ValueError(msg)
    if not 0 < confidence < 1:
      msg = f"confidence must be between 0 and 1, got {confidence}"
      raise ValueError(msg)

    super().__init__()
    self._interval = interval
    self._confidence = confidence

  @property
  def interval(self) -> float | None:
    """Get the fixed half-width of the interval, or None when the confidence interval is used."""
    return self._interval

  @property
  def confidence(self) -> float:
    """Get the confidence level of the default interval."""
    return self._confidence

  @override
  def select(self, context: StepContext) -> int:
    """Return the remaining item with the largest information integrated over the interval.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.
    """
    candidates = remaining(context)
    item_bank, theta = context.item_bank, context.theta

    if self._interval is None:
      # context.se is infinite before the first response, so the interval starts as the whole scale
      half_width = float(stats.norm.ppf((1 + self._confidence) / 2)) * context.se
    else:
      half_width = self._interval

    items = item_bank.items[candidates]
    if half_width < _MIN_HALF_WIDTH:
      # the interval has collapsed onto the estimate, where every integral is 0 or rounding noise
      scores = irt.information(theta, items)
    else:
      scores = irt.info_integral(theta - half_width, theta + half_width, items)
    return pick_best(candidates, scores, context.rng)
