"""Kullback-Leibler item selector."""

import math

import numpy
from scipy.integrate import quad
from scipy.special import log_expit

from .._typing import FloatArray
from ..item_bank import ItemBank
from .base import BaseSelector, below_cap, pick_best, remaining


def _log_probabilities(theta: float, a: float, b: float, c_param: float, d: float) -> tuple[float, float]:
  """Return the log-probabilities of a correct and an incorrect response at ``theta``.

  Both come from the logit ``a * (theta - b)``, as in :py:func:`catsim.irt.log_likelihood_hpc`, so
  the log-probability of an incorrect response stays exact where the probability of a correct one
  rounds to 1.
  """
  z = a * (theta - b)
  log_dmc = math.log(d - c_param)
  log_c = math.log(c_param) if c_param > 0 else -math.inf
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
  c_param: float,
  d: float,
) -> float:
  """Return the KL divergence between response distributions at two ability values."""
  log_p_hat, log_q_hat = _log_probabilities(theta_hat, a, b, c_param, d)
  log_p, log_q = _log_probabilities(theta, a, b, c_param, d)
  return math.exp(log_p_hat) * (log_p_hat - log_p) + math.exp(log_q_hat) * (log_q_hat - log_q)


class KLSelector(BaseSelector):
  r"""Kullback-Leibler global-information item selector.

  Each candidate's KL divergence between the response distributions at the current estimate
  and at nearby abilities is integrated over :math:`\hat\theta \pm c / \sqrt{\max(1, n)}`,
  where :math:`n` is the number of administered items, and the item with the largest integral
  is selected.

  Parameters
  ----------
  c : float, optional
      Width constant of the integration window, which narrows as items are administered.
      Default is 3.
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max will not be selected
      unless no other items are available. Default is 1 (no restriction).

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/kl-selector`.
  """

  def __init__(self, c: float = 3.0, r_max: float = 1.0) -> None:
    """Initialize a KLSelector object.

    Raises
    ------
    ValueError
        If c is not positive, or if r_max is not between 0 and 1.
    """
    if c <= 0:
      msg = f"c must be positive, got {c}"
      raise ValueError(msg)
    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)
    super().__init__()
    self._c = float(c)
    self._r_max = float(r_max)

  def __str__(self) -> str:
    """Return the name of the selector."""
    return f"Kullback-Leibler Selector (c={self._c})"

  @property
  def c(self) -> float:
    """Return the schedule constant used to define the integration window."""
    return self._c

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

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,
    exposure_rates: FloatArray | None = None,
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability, the center of the integration window.
    rng : numpy.random.Generator or None, optional
        Random number generator, used only to break ties between candidates with equal KL
        divergence (ADR-0008).
    exposure_rates : numpy.ndarray or None, optional
        Current exposure rate of every item, compared against ``r_max``.
    response_vector : list[bool] or None, optional
        Unused; accepted for the engine's selector contract.

    Returns
    -------
    int
        Index of the item with the largest integrated KL divergence; ties are broken uniformly at
        random (ADR-0008).

    Raises
    ------
    ValueError
        If candidates tie for the largest divergence and ``rng`` is None.
    NoItemsAvailableError
        If every item has been administered.
    """
    candidates = below_cap(remaining(item_bank.n_items, administered_items), exposure_rates, self._r_max)
    half_width = self._c / numpy.sqrt(max(1, len(administered_items)))

    kl_values = numpy.array([self._kl_global(est_theta, item_bank.items[int(idx)], half_width) for idx in candidates])
    return pick_best(candidates, kl_values, rng)
