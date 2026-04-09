"""Progressive and proportional selector implementations."""

import numpy
import numpy.typing as npt

from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import FiniteSelector


def _available_candidates(
  item_bank: ItemBank,
  administered_items: list[int],
  exposure_rates: npt.NDArray[numpy.floating] | None,
  r_max: float,
) -> npt.NDArray[numpy.int_]:
  """Return the non-administered candidate items honoring the exposure cap when possible."""
  candidate_mask = numpy.ones(item_bank.n_items, dtype=bool)
  candidate_mask[administered_items] = False
  candidates = numpy.flatnonzero(candidate_mask)
  if candidates.size == 0:
    msg = "There are no more items to apply."
    raise NoItemsAvailableError(msg)

  if exposure_rates is None:
    exposure_rates = numpy.zeros(item_bank.n_items, dtype=float)

  low_exposure_candidates = candidates[exposure_rates[candidates] < r_max]
  return low_exposure_candidates if low_exposure_candidates.size > 0 else candidates


class ProgressiveSelector(FiniteSelector):
  """Progressive selector that blends random and information-based scoring."""

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
    if self._test_size <= 1:
      return 1.0
    return ((position - 1) / (self._test_size - 1)) ** self._s

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,
    exposure_rates: npt.NDArray[numpy.floating] | None = None,
  ) -> int | None:
    """Return the next item using blended random and information-based scoring."""
    if rng is None:
      msg = "rng parameter cannot be None"
      raise ValueError(msg)

    candidates = _available_candidates(item_bank, administered_items, exposure_rates, self._r_max)
    position = len(administered_items) + 1
    weight = self._weight(position)

    info = numpy.asarray(item_bank.information(est_theta), dtype=float)
    info_max = float(info.max()) if info.size > 0 else 0.0
    info_norm = info / info_max if info_max > 0 else numpy.zeros_like(info)

    random_component = rng.uniform(0.0, 1.0, size=item_bank.n_items)
    scores = (1.0 - weight) * random_component + weight * info_norm
    return int(candidates[numpy.argmax(scores[candidates])])


class ProportionalSelector(FiniteSelector):
  """Proportional selector that samples items with probability proportional to weighted information."""

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
    if self._test_size <= 1:
      return self._k
    weight = ((position - 1) / (self._test_size - 1)) ** self._s
    return self._k * weight

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,
    exposure_rates: npt.NDArray[numpy.floating] | None = None,
  ) -> int | None:
    """Return the next item using information-weighted sampling."""
    if rng is None:
      msg = "rng parameter cannot be None"
      raise ValueError(msg)

    candidates = _available_candidates(item_bank, administered_items, exposure_rates, self._r_max)
    position = len(administered_items) + 1
    exponent = self._exponent(position)

    info = numpy.asarray(item_bank.information(est_theta), dtype=float)
    weights = numpy.power(numpy.maximum(info[candidates], 0.0), exponent)
    total = float(weights.sum())
    probs = numpy.full(candidates.size, 1.0 / candidates.size, dtype=float) if total <= 0 else weights / total

    return int(rng.choice(candidates, p=probs))
