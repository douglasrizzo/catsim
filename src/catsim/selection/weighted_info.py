"""Likelihood-weighted information selector implementations."""

from typing import Any

import numpy
import numpy.typing as npt

from .. import irt
from ..exceptions import NoItemsAvailableError
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..item_bank import ItemBank
from .base import BaseSelector

_MIN_NODES = 5


class MLWISelector(BaseSelector):
  """Maximum Likelihood Weighted Information selector.

  For full algorithmic details, see :doc:`/specs/selection/mlwi-selector`.
  """

  def __init__(
    self,
    n_nodes: int = 41,
    bounds: tuple[float, float] = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
    r_max: float = 1.0,
  ) -> None:
    if n_nodes < _MIN_NODES:
      msg = f"n_nodes must be >= {_MIN_NODES}, got {n_nodes}"
      raise ValueError(msg)
    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)

    super().__init__()
    self._nodes = numpy.linspace(bounds[0], bounds[1], n_nodes)
    self._delta = (bounds[1] - bounds[0]) / (n_nodes - 1)
    self._r_max = float(r_max)

  def __str__(self) -> str:
    """Return the selector name."""
    return f"Maximum Likelihood Weighted Information Selector (Q={len(self._nodes)})"

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate accepted by the selector."""
    return self._r_max

  def _likelihood_weights(
    self,
    response_vector: list[bool],
    administered_items: npt.NDArray[numpy.floating[Any]],
  ) -> npt.NDArray[numpy.floating[Any]]:
    log_likelihood = numpy.array(
      [irt.log_likelihood(float(theta), response_vector, administered_items) for theta in self._nodes],
      dtype=float,
    )
    log_likelihood -= float(log_likelihood.max())
    return numpy.exp(log_likelihood)

  def _weighted_information(
    self,
    item: npt.NDArray[numpy.floating[Any]],
    weights: npt.NDArray[numpy.floating[Any]],
  ) -> float:
    info = numpy.array([irt.inf(float(theta), item[0], item[1], item[2], item[3]) for theta in self._nodes])
    return float(numpy.sum(info * weights) * self._delta)

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,  # noqa: ARG002
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: npt.NDArray[numpy.floating[Any]] | None = None,
    response_vector: list[bool] | None = None,
  ) -> int | None:
    """Select the item with the highest likelihood-weighted information."""
    if response_vector is None:
      msg = "MLWISelector requires the response_vector for likelihood weighting."
      raise ValueError(msg)

    candidate_mask = numpy.ones(item_bank.n_items, dtype=bool)
    candidate_mask[administered_items] = False
    all_candidates = numpy.flatnonzero(candidate_mask)
    if all_candidates.size == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    if exposure_rates is None:
      exposure_rates = numpy.zeros(item_bank.n_items, dtype=float)
    low_exposure_mask = candidate_mask & (exposure_rates < self._r_max)

    candidates = numpy.flatnonzero(low_exposure_mask)
    if candidates.size == 0:
      candidates = all_candidates

    if administered_items:
      weights = self._likelihood_weights(response_vector, item_bank.items[administered_items, :4])
    else:
      weights = numpy.ones_like(self._nodes, dtype=float)

    scores = numpy.array(
      [self._weighted_information(item_bank.items[int(idx), :4], weights) for idx in candidates],
      dtype=float,
    )
    return int(candidates[int(scores.argmax())])
