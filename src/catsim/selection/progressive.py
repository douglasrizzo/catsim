"""Progressive and proportional selector implementations."""

import numpy
import numpy.typing as npt

from ..item_bank import ItemBank
from .base import FiniteSelector


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

  For full algorithmic details, see :doc:`/api/techniques/selection/progressive-selector`.
  """

  def __init__(
    self,
    test_size: int,
    acceleration: float = 1.0,
    r_max: float = 1.0,
  ) -> None:
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
    exposure_rates: npt.NDArray[numpy.floating] | None = None,
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the next item using blended random and information-based scoring."""
    if rng is None:
      msg = "rng parameter cannot be None"
      raise ValueError(msg)

    candidates = self._exposure_filtered_candidates(item_bank.n_items, administered_items, exposure_rates, self._r_max)
    position = len(administered_items) + 1
    weight = self._weight(position)

    # Revuelta and Ponsoda (1998) draw the random component on [0, max information among the
    # available items]; dividing by that maximum puts both components on [0, 1].
    info = numpy.asarray(item_bank.information(est_theta), dtype=float)
    info_max = float(info[candidates].max())
    info_norm = info / info_max if info_max > 0 else numpy.zeros_like(info)

    random_component = rng.uniform(0.0, 1.0, size=item_bank.n_items)
    scores = (1.0 - weight) * random_component + weight * info_norm
    return int(candidates[numpy.argmax(scores[candidates])])


class ProportionalSelector(FiniteSelector):
  """Proportional selector that samples items with probability proportional to weighted information.

  For full algorithmic details, see :doc:`/api/techniques/selection/proportional-selector`.
  """

  def __init__(
    self,
    test_size: int,
    acceleration: float = 1.0,
    sharpness: float = 6.0,
    r_max: float = 1.0,
  ) -> None:
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
    exposure_rates: npt.NDArray[numpy.floating] | None = None,
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the next item using information-weighted sampling."""
    if rng is None:
      msg = "rng parameter cannot be None"
      raise ValueError(msg)

    candidates = self._exposure_filtered_candidates(item_bank.n_items, administered_items, exposure_rates, self._r_max)
    position = len(administered_items) + 1
    exponent = self._exponent(position)

    info = numpy.asarray(item_bank.information(est_theta), dtype=float)
    weights = numpy.power(numpy.maximum(info[candidates], 0.0), exponent)
    total = float(weights.sum())
    probs = numpy.full(candidates.size, 1.0 / candidates.size, dtype=float) if total <= 0 else weights / total

    return int(rng.choice(candidates, p=probs))
