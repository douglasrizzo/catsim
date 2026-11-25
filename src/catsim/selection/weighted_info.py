"""Likelihood-weighted information selector implementations."""

import math

from typing_extensions import override

from .. import irt
from ..context import StepContext
from ..estimation import QuadratureGrid, posterior, uniform_log_prior
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..typing import FloatArray, IntArray
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining

_MIN_NODES = 5


class MLWISelector(BaseSelector):
  """Maximum Likelihood Weighted Information selector.

  Scores each candidate item by its information integrated over ability, weighted by the
  likelihood of the responses so far, and selects the highest score. The integral is approximated
  on a fixed grid of equally spaced nodes.

  Parameters
  ----------
  n_nodes : int, optional
      Number of grid nodes, including both bounds. Default is 41.
  bounds : tuple[float, float], optional
      First and last grid node. Default is ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``.
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max are only selected when no
      other item remains. Default is 1 (no restriction).

  Raises
  ------
  ValueError
      If n_nodes is below 5, the bounds are not finite or not increasing, or r_max is not in (0, 1].

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/mlwi-selector`.
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
    if not (math.isfinite(bounds[0]) and math.isfinite(bounds[1])):
      msg = f"bounds must be finite, got {bounds}"
      raise ValueError(msg)
    if not bounds[0] < bounds[1]:
      msg = f"bounds must be increasing, got {bounds}"
      raise ValueError(msg)

    super().__init__()
    self._n_nodes = n_nodes
    self._bounds = (float(bounds[0]), float(bounds[1]))
    self._grid = QuadratureGrid.uniform(n_nodes, *self._bounds)
    self._log_prior = uniform_log_prior(*self._bounds)
    self._r_max = check_r_max(r_max)

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(n_nodes={self._n_nodes!r}, bounds={self._bounds!r}, r_max={self._r_max!r})"

  @property
  def n_nodes(self) -> int:
    """Return the number of grid nodes, including both bounds."""
    return self._n_nodes

  @property
  def bounds(self) -> tuple[float, float]:
    """Return the first and last grid node."""
    return self._bounds

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate accepted by the selector."""
    return self._r_max

  def _score(self, context: StepContext, candidates: IntArray) -> FloatArray:
    """Return each candidate's information averaged over the grid, weighted by the likelihood of the responses."""
    item_bank = context.item_bank
    # under a flat prior the posterior over the grid is the likelihood, normalized to sum to 1
    weights = posterior(context.responses, item_bank.get_items(context.administered_items), self._grid, self._log_prior)
    information = irt.information(self._grid.nodes, item_bank.get_items(candidates))
    # a sum over the node axis adds each column in the same order, so identical items tie exactly
    # (ADR-0017); a matrix product may round columns differently
    return (weights[:, None] * information).sum(axis=0)

  @override
  def select(self, context: StepContext) -> int:
    """Return the available item with the largest information integrated against the likelihood of the responses.

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
    return pick_best(candidates, self._score(context, candidates), context.rng)
