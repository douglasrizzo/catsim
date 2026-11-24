"""Information-based selectors: maximum information, Urry's rule and interval information."""

import numpy
from scipy import stats
from typing_extensions import override

from .. import irt
from ..context import StepContext
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining


class MaxInfoSelector(BaseSelector):
  """Select the non-administered item with maximum information at the current theta estimate.

  Parameters
  ----------
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max will not
      be selected unless no other items are available. Default is 1 (no restriction).

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/max-info-selector`.
  """

  def __init__(self, r_max: float = 1) -> None:
    """Initialize a MaxInfoSelector object.

    Parameters
    ----------
    r_max : float, optional
        Maximum exposure rate for items. Default is 1.

    Raises
    ------
    ValueError
        If r_max is not in (0, 1].
    """
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


class IntervalInfoSelector(BaseSelector):
  r"""Select the item whose information, integrated over an interval around the estimate, is largest.

  By default the interval is the estimate plus or minus :math:`\lambda` times the estimator's
  standard error (``context.se``), where :math:`\lambda` is the standard normal quantile of
  ``(1 + confidence) / 2`` :cite:p:`Veerkamp1997`. It spans the whole ability scale before the first
  response and narrows as the standard error falls. Setting ``interval`` fixes the half-width. The
  selector applies no exposure cap: ``context.exposure_rates`` is ignored.

  Parameters
  ----------
  interval : float or None, optional
      Fixed half-width of the interval around the estimate. Default is None, which uses the
      confidence interval.
  confidence : float, optional
      Confidence level of the default interval. Default is 0.95.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/interval-info-selector`.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(interval={self._interval!r}, confidence={self._confidence!r})"

  def __init__(self, interval: float | None = None, confidence: float = 0.95) -> None:
    r"""Initialize an IntervalInfoSelector object.

    Parameters
    ----------
    interval : float or None, optional
        Fixed half-width of the interval. Default is None, which uses a confidence interval of
        :math:`\hat\theta`.
    confidence : float, optional
        Confidence level of the default interval. Default is 0.95.

    Raises
    ------
    ValueError
        If ``interval`` is not positive or ``confidence`` is not between 0 and 1.
    """
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

    information_integral = irt.info_integral(theta - half_width, theta + half_width, item_bank.items[candidates])
    return pick_best(candidates, information_integral, context.rng)
