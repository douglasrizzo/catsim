"""Likelihood-weighted information selector implementations."""

import numpy

from .. import irt
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..item_bank import ItemBank
from ..typing import FloatArray
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining, require_responses

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
    """Initialize a MLWISelector object.

    Raises
    ------
    ValueError
        If n_nodes is below 5, the bounds are not increasing, or r_max is not in (0, 1].
    """
    if n_nodes < _MIN_NODES:
      msg = f"n_nodes must be >= {_MIN_NODES}, got {n_nodes}"
      raise ValueError(msg)
    if not bounds[0] < bounds[1]:
      msg = f"bounds must be increasing, got {bounds}"
      raise ValueError(msg)

    super().__init__()
    self._n_nodes = n_nodes
    self._bounds = (float(bounds[0]), float(bounds[1]))
    self._nodes = numpy.linspace(bounds[0], bounds[1], n_nodes)
    self._delta = (bounds[1] - bounds[0]) / (n_nodes - 1)
    self._r_max = check_r_max(r_max)

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

  def _likelihood_weights(
    self,
    response_vector: list[bool],
    administered_items: FloatArray,
  ) -> FloatArray:
    log_likelihood = irt.log_likelihood_hpc(self._nodes, response_vector, administered_items)
    log_likelihood -= float(numpy.max(log_likelihood))
    return numpy.exp(log_likelihood)

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,  # noqa: ARG002
    rng: numpy.random.Generator | None = None,
    exposure_rates: FloatArray | None = None,
    response_vector: list[bool] | None = None,
  ) -> int | None:
    """Return the index of the item with the highest likelihood-weighted information.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        Unused; MLWI weights by the whole likelihood rather than a point estimate.
    rng : numpy.random.Generator or None, optional
        Random number generator, used only to break ties between candidates with equal scores
        (ADR-0008).
    exposure_rates : numpy.ndarray or None, optional
        Run-level exposure-rate snapshot. ``None`` treats every item as unexposed.
    response_vector : list[bool] or None, optional
        Responses to the administered items. Required.

    Returns
    -------
    int
        Index of the next item to be applied.

    Raises
    ------
    ValueError
        If ``response_vector`` is None, or if candidates tie for the best score and ``rng`` is None.
    NoItemsAvailableError
        If every item has already been administered.
    """
    response_vector = require_responses(response_vector)

    candidates = below_cap(remaining(item_bank.n_items, administered_items), exposure_rates, self._r_max)

    if administered_items:
      weights = self._likelihood_weights(response_vector, item_bank.get_items(administered_items))
    else:
      weights = numpy.ones_like(self._nodes, dtype=float)

    # information of every candidate at every node, one row per node
    info = irt.inf_hpc(self._nodes[:, None], item_bank.get_items(candidates))
    scores = (weights @ info) * self._delta
    return pick_best(candidates, scores, rng)
