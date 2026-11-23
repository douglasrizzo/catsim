"""Likelihood-weighted information selector implementations."""

import numpy
import numpy.typing as npt

from .. import irt
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..item_bank import ItemBank
from .base import BaseSelector

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
        If n_nodes is below 5, the bounds are not increasing, or r_max is not between 0 and 1.
    """
    if n_nodes < _MIN_NODES:
      msg = f"n_nodes must be >= {_MIN_NODES}, got {n_nodes}"
      raise ValueError(msg)
    if not bounds[0] < bounds[1]:
      msg = f"bounds must be increasing, got {bounds}"
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
    administered_items: npt.NDArray[numpy.floating],
  ) -> npt.NDArray[numpy.floating]:
    log_likelihood = numpy.array(
      [irt.log_likelihood(float(theta), response_vector, administered_items) for theta in self._nodes],
      dtype=float,
    )
    log_likelihood -= float(log_likelihood.max())
    return numpy.exp(log_likelihood)

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,  # noqa: ARG002
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: npt.NDArray[numpy.floating] | None = None,
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
        Unused; MLWI is deterministic.
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
        If ``response_vector`` is None.
    NoItemsAvailableError
        If every item has already been administered.
    """
    if response_vector is None:
      msg = "MLWISelector requires the response_vector for likelihood weighting."
      raise ValueError(msg)

    candidates = self._exposure_filtered_candidates(item_bank.n_items, administered_items, exposure_rates, self._r_max)

    if administered_items:
      weights = self._likelihood_weights(response_vector, item_bank.items[administered_items, :4])
    else:
      weights = numpy.ones_like(self._nodes, dtype=float)

    # information of every candidate at every node, one row per node
    info = irt.inf_hpc(self._nodes[:, None], item_bank.items[candidates, :4])
    scores = (weights @ info) * self._delta
    return int(candidates[int(scores.argmax())])
