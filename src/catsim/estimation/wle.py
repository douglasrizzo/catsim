"""Warm's Weighted Likelihood Estimator for CAT."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from .. import irt
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from .base import BaseEstimator
from .numerical import NumericalSearchEstimator

if TYPE_CHECKING:
  from ..item_bank import ItemBank

FloatArray = npt.NDArray[numpy.floating[Any]]


class WarmLikelihoodEstimator(BaseEstimator):
  """Warm (1989) bias-corrected ability estimator.

  The estimator reuses the existing numerical MLE implementation and applies
  Warm's first-order bias correction when the MLE is finite. If the numerical
  MLE diverges for an extreme response pattern, the estimator falls back to a
  direct bounded maximization of the weighted likelihood.

  For full algorithmic details, see :doc:`/specs/estimation/warm-wle`.
  """

  def __init__(self, tol: float = 1e-6, verbose: bool = False) -> None:
    super().__init__(verbose=verbose)
    self._tol = float(tol)
    self._mle = NumericalSearchEstimator(tol=tol, dodd=False, verbose=verbose, method="bounded")

  def __str__(self) -> str:
    """Return a string representation of the estimator."""
    return "Warm Weighted Likelihood Estimator"

  @staticmethod
  def _bias_correction(theta: float, items: FloatArray) -> float:
    """Return Warm's first-order correction term J(theta) / [2 I(theta)^2]."""
    if items.size == 0:
      return 0.0

    a = items[:, 0]
    b = items[:, 1]
    c = items[:, 2]
    d = items[:, 3]

    z = numpy.exp(-a * (theta - b))
    one_plus_z = 1.0 + z

    p = c + (d - c) / one_plus_z
    p_prime = a * (d - c) * z / (one_plus_z**2)
    p_double = a * a * (d - c) * z * (z - 1.0) / (one_plus_z**3)

    q = 1.0 - p
    eps = numpy.finfo(float).eps
    denom = numpy.clip(p * q, eps, None)
    info_i = (p_prime**2) / denom
    j_i = (p_prime * p_double) / denom

    info = float(numpy.sum(info_i))
    if not numpy.isfinite(info) or info <= 0.0:
      return 0.0
    j = float(numpy.sum(j_i))
    if not numpy.isfinite(j):
      return 0.0
    return j / (2.0 * info * info)

  @staticmethod
  def _weighted_negative_log_likelihood(
    theta: float,
    response_vector: list[bool],
    items: FloatArray,
  ) -> float:
    log_lik = irt.log_likelihood(theta, response_vector, items)
    info = irt.test_info(theta, items)
    if info <= 0.0 or not numpy.isfinite(info):
      return float("inf")
    return -(log_lik + 0.5 * numpy.log(info))

  def _fallback_weighted_mle(
    self,
    response_vector: list[bool],
    items: FloatArray,
  ) -> float:
    result = minimize_scalar(
      self._weighted_negative_log_likelihood,
      args=(response_vector, items),
      bounds=(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
      method="bounded",
      options={"xatol": self._tol},
    )
    self._last_evaluations = int(result.nfev)
    self._total_evaluations += self._last_evaluations
    return float(result.x)

  def estimate(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,
  ) -> float:
    """Estimate ability using Warm's weighted likelihood correction."""
    if item_bank is None:
      msg = "item_bank parameter cannot be None"
      raise ValueError(msg)
    if administered_items is None:
      msg = "administered_items parameter cannot be None"
      raise ValueError(msg)
    if response_vector is None:
      msg = "response_vector parameter cannot be None"
      raise ValueError(msg)

    self._calls += 1
    self._last_evaluations = 0

    if not administered_items:
      return float(est_theta)

    items = item_bank.items[administered_items, :4]
    theta_mle = self._mle.estimate(item_bank, administered_items, response_vector, est_theta)

    if numpy.isfinite(theta_mle):
      theta = float(theta_mle + self._bias_correction(theta_mle, items))
      self._last_evaluations = self._mle.evaluations
      self._total_evaluations += self._last_evaluations
      return theta

    return self._fallback_weighted_mle(response_vector, items)
