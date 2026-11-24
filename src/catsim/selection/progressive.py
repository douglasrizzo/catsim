"""Progressive and proportional selector implementations."""

import numpy

from ..context import StepContext
from .base import BaseSelector, below_cap, check_r_max, check_test_size, pick_best, remaining


def _position_weight(position: int, test_size: int, acceleration: float) -> float:
  """Return Barrada et al.'s (2008) weight on item information at a test position.

  Parameters
  ----------
  position : int
      1-indexed position of the item about to be selected.
  test_size : int
      Planned number of items in the test, at least 1.
  acceleration : float
      Acceleration parameter of the schedule.

  Returns
  -------
  float
      ``sum(k ** acceleration for k in range(1, position))`` divided by the same sum up to
      ``test_size``: 0 at the first position and 1 at ``test_size``. Past ``test_size``, where the
      stopper may let a session run, the weight stays 1 instead of exceeding it.
  """
  if position >= test_size:
    return 1.0
  steps = numpy.arange(1, test_size, dtype=float)
  # Dividing every term by the largest, the last for a positive acceleration and the first
  # otherwise, keeps the sums from overflowing and leaves their ratio unchanged.
  terms = (steps / steps[-1] if acceleration > 0 else steps) ** acceleration
  return float(terms[: position - 1].sum() / terms.sum())


def _check_acceleration(acceleration: float) -> float:
  """Return ``acceleration`` as a float, after checking that it is finite.

  Parameters
  ----------
  acceleration : float
      Acceleration parameter of the schedule.

  Returns
  -------
  float
      ``acceleration``.

  Raises
  ------
  ValueError
      If ``acceleration`` is infinite or NaN.
  """
  if not numpy.isfinite(acceleration):
    msg = f"acceleration must be finite, got {acceleration}"
    raise ValueError(msg)
  return float(acceleration)


class ProgressiveSelector(BaseSelector):
  """Progressive selector that blends random and information-based scoring.

  Each available item scores a uniform random draw blended with its information, scaled by the
  largest available information; the weight on information follows Barrada et al.'s (2008)
  schedule from 0 at the first position to 1 at ``test_size``, and the item with the largest score
  is selected.

  Parameters
  ----------
  test_size : int
      Planned number of items in the test, at least 1, at which the weight on information reaches 1.
  acceleration : float, optional
      Barrada et al.'s acceleration parameter :math:`t`, any finite number. 0 makes the schedule
      linear, positive values keep the random component longer and negative values drop it sooner.
      Default is 0.
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max will not be selected
      unless no other items are available. Default is 1 (no restriction).

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/progressive-selector`.
  """

  def __init__(
    self,
    test_size: int,
    acceleration: float = 0.0,
    r_max: float = 1.0,
  ) -> None:
    """Initialize a ProgressiveSelector object.

    Raises
    ------
    ValueError
        If ``test_size`` is smaller than 1, ``acceleration`` is not finite or ``r_max`` is not in
        (0, 1].
    """
    super().__init__()
    self._test_size = check_test_size(test_size)
    self._acceleration = _check_acceleration(acceleration)
    self._r_max = check_r_max(r_max)

  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return (
      f"{type(self).__name__}(test_size={self._test_size!r}, acceleration={self._acceleration!r}, "
      f"r_max={self._r_max!r})"
    )

  @property
  def test_size(self) -> int:
    """Return the planned number of items in the test."""
    return self._test_size

  @property
  def acceleration(self) -> float:
    """Return the acceleration parameter of the schedule."""
    return self._acceleration

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate accepted by the selector."""
    return self._r_max

  def select(self, context: StepContext) -> int:
    """Return the available item with the largest blend of a random draw and its scaled information.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.
    """
    item_bank, rng = context.item_bank, context.rng
    candidates = below_cap(remaining(context), context.exposure_rates, self._r_max)
    weight = _position_weight(len(context.administered_items) + 1, self._test_size, self._acceleration)

    # Revuelta and Ponsoda (1998) draw the random component on [0, max information among the
    # available items]; dividing by that maximum puts both components on [0, 1].
    info = numpy.asarray(item_bank.information(context.theta), dtype=float)
    info_max = float(info[candidates].max())
    info_norm = info / info_max if info_max > 0 else numpy.zeros_like(info)

    random_component = rng.uniform(0.0, 1.0, size=item_bank.n_items)
    scores = (1.0 - weight) * random_component + weight * info_norm
    return pick_best(candidates, scores[candidates], rng)


class ProportionalSelector(BaseSelector):
  """Proportional selector that samples items with probability proportional to weighted information.

  Each available item is drawn with probability proportional to its information raised to a
  power that follows Barrada et al.'s (2008) schedule from 0 at the first position, where sampling
  is uniform, to the test length at ``test_size``.

  Parameters
  ----------
  test_size : int
      Planned number of items in the test, at least 1. From that position on, the exponent equals it.
  acceleration : float, optional
      Barrada et al.'s acceleration parameter :math:`s`, any finite number, which shapes the
      exponent's growth as :math:`t` shapes :py:class:`ProgressiveSelector`'s weight. 0 makes the
      growth linear. Default is 0.
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max will not be selected
      unless no other items are available. Default is 1 (no restriction).

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/proportional-selector`.
  """

  def __init__(
    self,
    test_size: int,
    acceleration: float = 0.0,
    r_max: float = 1.0,
  ) -> None:
    """Initialize a ProportionalSelector object.

    Raises
    ------
    ValueError
        If ``test_size`` is smaller than 1, ``acceleration`` is not finite or ``r_max`` is not in
        (0, 1].
    """
    super().__init__()
    self._test_size = check_test_size(test_size)
    self._acceleration = _check_acceleration(acceleration)
    self._r_max = check_r_max(r_max)

  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return (
      f"{type(self).__name__}(test_size={self._test_size!r}, acceleration={self._acceleration!r}, "
      f"r_max={self._r_max!r})"
    )

  @property
  def test_size(self) -> int:
    """Return the planned number of items in the test."""
    return self._test_size

  @property
  def acceleration(self) -> float:
    """Return the acceleration parameter of the schedule."""
    return self._acceleration

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate accepted by the selector."""
    return self._r_max

  def select(self, context: StepContext) -> int:
    """Return an available item drawn with probability proportional to its information raised to the schedule's power.

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
    position = len(context.administered_items) + 1
    exponent = self._test_size * _position_weight(position, self._test_size, self._acceleration)

    info = numpy.maximum(numpy.asarray(context.item_bank.information(context.theta), dtype=float)[candidates], 0.0)
    info_max = float(info.max())
    if info_max <= 0:
      probs = numpy.full(candidates.size, 1.0 / candidates.size, dtype=float)
    else:
      # Scaling by the maximum leaves the probabilities unchanged and keeps a large exponent from
      # underflowing every weight to 0; the best item's weight is 1, so the sum is at least 1.
      weights = numpy.power(info / info_max, exponent)
      probs = weights / weights.sum()

    return int(context.rng.choice(candidates, p=probs))
