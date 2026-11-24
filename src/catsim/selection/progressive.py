"""Progressive and proportional selector implementations."""

import numpy

from ..item_bank import ItemBank
from ..typing import FloatArray
from .base import BaseSelector, below_cap, check_r_max, check_test_size, pick_best, remaining, require_rng


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

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,
    exposure_rates: FloatArray | None = None,
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the next item using blended random and information-based scoring.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered; their count sets
        the test position.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator for the random component. Required.
    exposure_rates : numpy.ndarray or None, optional
        Current exposure rate of every item, compared against ``r_max``.
    response_vector : list[bool] or None, optional
        Unused; accepted for the engine's selector contract.

    Returns
    -------
    int
        Index of the selected item; ties are broken uniformly at random (ADR-0008).

    Raises
    ------
    ValueError
        If ``rng`` is None.
    NoItemsAvailableError
        If every item has been administered.
    """
    rng = require_rng(rng)

    candidates = below_cap(remaining(item_bank.n_items, administered_items), exposure_rates, self._r_max)
    weight = _position_weight(len(administered_items) + 1, self._test_size, self._acceleration)

    # Revuelta and Ponsoda (1998) draw the random component on [0, max information among the
    # available items]; dividing by that maximum puts both components on [0, 1].
    info = numpy.asarray(item_bank.information(est_theta), dtype=float)
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

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,
    exposure_rates: FloatArray | None = None,
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the next item using information-weighted sampling.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered; their count sets
        the test position.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator used to draw the item. Required.
    exposure_rates : numpy.ndarray or None, optional
        Current exposure rate of every item, compared against ``r_max``.
    response_vector : list[bool] or None, optional
        Unused; accepted for the engine's selector contract.

    Returns
    -------
    int
        Index of the selected item.

    Raises
    ------
    ValueError
        If ``rng`` is None.
    NoItemsAvailableError
        If every item has been administered.
    """
    rng = require_rng(rng)

    candidates = below_cap(remaining(item_bank.n_items, administered_items), exposure_rates, self._r_max)
    exponent = self._test_size * _position_weight(len(administered_items) + 1, self._test_size, self._acceleration)

    info = numpy.maximum(numpy.asarray(item_bank.information(est_theta), dtype=float)[candidates], 0.0)
    info_max = float(info.max())
    if info_max <= 0:
      probs = numpy.full(candidates.size, 1.0 / candidates.size, dtype=float)
    else:
      # Scaling by the maximum leaves the probabilities unchanged and keeps a large exponent from
      # underflowing every weight to 0; the best item's weight is 1, so the sum is at least 1.
      weights = numpy.power(info / info_max, exponent)
      probs = weights / weights.sum()

    return int(rng.choice(candidates, p=probs))
