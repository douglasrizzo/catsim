"""Progressive and proportional selector implementations."""

import numpy

from .._typing import FloatArray
from ..item_bank import ItemBank
from .base import FiniteSelector, below_cap, pick_best, remaining


def _position_weight(position: int, test_size: int, acceleration: float) -> float:
  """Return the progression weight ``((position - 1) / (test_size - 1)) ** acceleration``.

  The ratio is clamped to 1 so that sessions running past ``test_size`` (``FiniteSelector`` does
  not enforce it) keep a fully information-based weight instead of extrapolating beyond 1.
  """
  if test_size <= 1:
    return 1.0
  return min(1.0, (position - 1) / (test_size - 1)) ** acceleration


class ProgressiveSelector(FiniteSelector):
  """Progressive selector that blends random and information-based scoring.

  Each available item scores a uniform random draw blended with its information, scaled by the
  largest available information; the weight on information grows from 0 at the first position
  to 1 at ``test_size``, and the item with the largest score is selected.

  Parameters
  ----------
  test_size : int
      Planned number of items in the test, at which the weight on information reaches 1.
  acceleration : float, optional
      Exponent of the position schedule. Values above 1 delay the shift toward information and
      values below 1 hasten it. Default is 1 (linear).
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
    acceleration: float = 1.0,
    r_max: float = 1.0,
  ) -> None:
    """Initialize a ProgressiveSelector object.

    Raises
    ------
    ValueError
        If ``acceleration`` is not positive or ``r_max`` is not between 0 and 1.
    """
    if acceleration <= 0:
      msg = f"acceleration must be positive, got {acceleration}"
      raise ValueError(msg)
    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)
    super().__init__(test_size=test_size)
    self._s = float(acceleration)
    self._r_max = float(r_max)

  def __str__(self) -> str:
    """Return the selector name."""
    return f"Progressive Selector (s={self._s})"

  def _weight(self, position: int) -> float:
    """Return the progression weight for the given 1-indexed test position."""
    return _position_weight(position, self._test_size, self._s)

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
    if rng is None:
      msg = "rng parameter cannot be None"
      raise ValueError(msg)

    candidates = below_cap(remaining(item_bank.n_items, administered_items), exposure_rates, self._r_max)
    position = len(administered_items) + 1
    weight = self._weight(position)

    # Revuelta and Ponsoda (1998) draw the random component on [0, max information among the
    # available items]; dividing by that maximum puts both components on [0, 1].
    info = numpy.asarray(item_bank.information(est_theta), dtype=float)
    info_max = float(info[candidates].max())
    info_norm = info / info_max if info_max > 0 else numpy.zeros_like(info)

    random_component = rng.uniform(0.0, 1.0, size=item_bank.n_items)
    scores = (1.0 - weight) * random_component + weight * info_norm
    return pick_best(candidates, scores[candidates], rng)


class ProportionalSelector(FiniteSelector):
  """Proportional selector that samples items with probability proportional to weighted information.

  Each available item is drawn with probability proportional to its information raised to
  ``sharpness`` times the position weight, which grows from 0 at the first position (uniform
  sampling) to 1 at ``test_size``.

  Parameters
  ----------
  test_size : int
      Planned number of items in the test, at which the exponent reaches ``sharpness``.
  acceleration : float, optional
      Exponent of the position schedule, as in :py:class:`ProgressiveSelector`. Default is 1.
  sharpness : float, optional
      Largest exponent applied to the information. 0 keeps sampling uniform; the test length
      reproduces Barrada et al.'s end point. Default is 6.
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
    acceleration: float = 1.0,
    sharpness: float = 6.0,
    r_max: float = 1.0,
  ) -> None:
    """Initialize a ProportionalSelector object.

    Raises
    ------
    ValueError
        If ``acceleration`` is not positive, ``sharpness`` is negative, or ``r_max`` is not
        between 0 and 1.
    """
    if acceleration <= 0:
      msg = f"acceleration must be positive, got {acceleration}"
      raise ValueError(msg)
    if sharpness < 0:
      msg = f"sharpness must be non-negative, got {sharpness}"
      raise ValueError(msg)
    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)
    super().__init__(test_size=test_size)
    self._s = float(acceleration)
    self._k = float(sharpness)
    self._r_max = float(r_max)

  def __str__(self) -> str:
    """Return the selector name."""
    return f"Proportional Selector (s={self._s}, k={self._k})"

  def _exponent(self, position: int) -> float:
    """Return the information exponent for the given 1-indexed test position."""
    return self._k * _position_weight(position, self._test_size, self._s)

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
    if rng is None:
      msg = "rng parameter cannot be None"
      raise ValueError(msg)

    candidates = below_cap(remaining(item_bank.n_items, administered_items), exposure_rates, self._r_max)
    position = len(administered_items) + 1
    exponent = self._exponent(position)

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
