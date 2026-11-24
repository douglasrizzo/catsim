"""Kullback-Leibler item selector."""

import functools

import numpy
from typing_extensions import override

from .. import irt
from ..context import StepContext
from ..typing import FloatArray
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining

# The integrand's complex singularities lie about pi / a from the real axis, so the Gauss-Legendre
# error falls like exp(-2 pi n / (a h)) for n nodes, discrimination a and half-width h. Four nodes
# per unit of a * h, for the steepest item, keep every integral within a relative 1e-10 of
# adaptive quadrature up to a * h = 60 (a = 10 over theta_hat +- 6); integrals below about 1e-6 are
# limited by rounding in the integrand instead. Past a * h = 60 the count stays at the cap and the
# error grows: about 3e-9 at a * h = 90 and 2e-7 at 120.
_NODES_PER_UNIT = 4.0
_MIN_NODES = 24
_MAX_NODES = 240


@functools.cache
def _gauss_legendre(n_nodes: int) -> tuple[FloatArray, FloatArray]:
  """Return the nodes on [-1, 1] and the weights of the n-node Gauss-Legendre rule, read-only."""
  nodes, weights = numpy.polynomial.legendre.leggauss(n_nodes)
  nodes.flags.writeable = weights.flags.writeable = False
  return nodes, weights


class KLSelector(BaseSelector):
  r"""Kullback-Leibler global-information item selector.

  Each candidate's KL divergence between the response distributions at the current estimate
  and at nearby abilities is integrated over :math:`\hat\theta \pm c / \sqrt{\max(1, n)}`,
  where :math:`c` is ``window`` and :math:`n` the number of administered items, and the item with
  the largest integral is selected.

  Parameters
  ----------
  window : float, optional
      Chang and Ying's constant :math:`c`, the half-width of the integration window before it
      narrows as items are administered. Default is 3.
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max will not be selected
      unless no other items are available. Default is 1 (no restriction).

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/kl-selector`.
  """

  def __init__(self, window: float = 3.0, r_max: float = 1.0) -> None:
    """Initialize a KLSelector object.

    Raises
    ------
    ValueError
        If window is not positive, or if r_max is not in (0, 1].
    """
    if window <= 0:
      msg = f"window must be positive, got {window}"
      raise ValueError(msg)
    super().__init__()
    self._window = float(window)
    self._r_max = check_r_max(r_max)

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(window={self._window!r}, r_max={self._r_max!r})"

  @property
  def window(self) -> float:
    r"""Return Chang and Ying's constant :math:`c`: the integration half-width is :math:`c / \sqrt{\max(1, n)}`."""
    return self._window

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate accepted by the selector."""
    return self._r_max

  @staticmethod
  def _kl_global(theta_hat: float, items: FloatArray, half_width: float) -> FloatArray:
    """Integrate each item's KL divergence over theta_hat +- half_width, with one rule sized for the steepest item."""
    steepness = float(numpy.max(items[:, 0])) * half_width
    n_nodes = int(numpy.clip(numpy.ceil(_NODES_PER_UNIT * steepness), _MIN_NODES, _MAX_NODES))
    nodes, weights = _gauss_legendre(n_nodes)
    log_p_hat, log_q_hat = irt.log_response_probabilities(theta_hat, items)
    log_p, log_q = irt.log_response_probabilities(theta_hat + half_width * nodes, items)
    divergence = numpy.exp(log_p_hat) * (log_p_hat - log_p) + numpy.exp(log_q_hat) * (log_q_hat - log_q)
    # one rule for every item, and a sum over the node axis that adds each column in the same order,
    # so identical items tie exactly (ADR-0008); a matrix product may round columns differently
    return half_width * (weights[:, None] * divergence).sum(axis=0)

  @override
  def select(self, context: StepContext) -> int:
    """Return the available item with the largest KL divergence integrated over the window.

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
    half_width = self._window / numpy.sqrt(max(1, len(context.administered_items)))

    scores = self._kl_global(context.theta, context.item_bank.get_items(candidates), half_width)
    return pick_best(candidates, scores, context.rng)
