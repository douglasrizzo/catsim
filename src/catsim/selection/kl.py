"""Kullback-Leibler item selector."""

import math

import numpy
from scipy.integrate import quad
from scipy.special import log_expit
from typing_extensions import override

from ..context import StepContext
from ..typing import FloatArray
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining


def _log_probabilities(theta: float, a: float, b: float, c: float, d: float) -> tuple[float, float]:
  """Return the log-probabilities of a correct and an incorrect response at ``theta``.

  Both come from the logit ``a * (theta - b)``, as in :py:func:`catsim.irt.log_likelihood`, so
  the log-probability of an incorrect response stays exact where the probability of a correct one
  rounds to 1.

  Returns
  -------
  tuple[float, float]
      The log-probabilities of a correct and of an incorrect response.
  """
  z = a * (theta - b)
  log_dmc = math.log(d - c)
  log_c = math.log(c) if c > 0 else -math.inf
  log_1md = math.log1p(-d) if d < 1 else -math.inf
  return (
    float(numpy.logaddexp(log_c, log_dmc + log_expit(z))),
    float(numpy.logaddexp(log_1md, log_dmc + log_expit(-z))),
  )


def _kl_integrand(
  theta: float,
  theta_hat: float,
  a: float,
  b: float,
  c: float,
  d: float,
) -> float:
  """Return the KL divergence between response distributions at two ability values."""
  log_p_hat, log_q_hat = _log_probabilities(theta_hat, a, b, c, d)
  log_p, log_q = _log_probabilities(theta, a, b, c, d)
  return math.exp(log_p_hat) * (log_p_hat - log_p) + math.exp(log_q_hat) * (log_q_hat - log_q)


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
  def _kl_global(theta_hat: float, item: FloatArray, half_width: float) -> float:
    """Integrate the KL divergence for one item over the local ability window."""
    val, _ = quad(
      _kl_integrand,
      theta_hat - half_width,
      theta_hat + half_width,
      args=(theta_hat, float(item[0]), float(item[1]), float(item[2]), float(item[3])),
    )
    return float(val)

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

    items = context.item_bank.items
    kl_values = numpy.array([self._kl_global(context.theta, items[int(idx)], half_width) for idx in candidates])
    return pick_best(candidates, kl_values, context.rng)
