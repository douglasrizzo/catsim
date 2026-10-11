"""Kullback-Leibler item selector."""

import numpy
import numpy.typing as npt
from scipy.integrate import quad

from .. import irt
from ..item_bank import ItemBank
from .base import BaseSelector

FloatArray = npt.NDArray[numpy.floating]


def _kl_integrand(
  theta: float,
  theta_hat: float,
  a: float,
  b: float,
  c_param: float,
  d: float,
) -> float:
  """Return the KL divergence between response distributions at two ability values."""
  p_hat = irt.icc(theta_hat, a, b, c_param, d)
  p = irt.icc(theta, a, b, c_param, d)

  eps = 1e-12
  p_hat = numpy.clip(p_hat, eps, 1.0 - eps)
  p = numpy.clip(p, eps, 1.0 - eps)

  return float(p_hat * numpy.log(p_hat / p) + (1.0 - p_hat) * numpy.log((1.0 - p_hat) / (1.0 - p)))


class KLSelector(BaseSelector):
  """Kullback-Leibler global-information item selector.

  For full algorithmic details, see :doc:`/api/techniques/selection/kl-selector`.
  """

  def __init__(self, c: float = 3.0, r_max: float = 1.0) -> None:
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
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: npt.NDArray[numpy.floating] | None = None,
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered."""
    candidates = self._exposure_filtered_candidates(item_bank.n_items, administered_items, exposure_rates, self._r_max)
    half_width = self._c / numpy.sqrt(max(1, len(administered_items)))

    kl_values = numpy.array([self._kl_global(est_theta, item_bank.items[int(idx)], half_width) for idx in candidates])
    return int(candidates[int(kl_values.argmax())])
