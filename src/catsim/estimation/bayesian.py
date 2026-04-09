"""Shared Bayesian helpers for CAT estimation."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy
import numpy.typing as npt
from scipy.optimize import minimize_scalar

from .. import irt
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..item_bank import ItemBank
from .base import BaseEstimator

FloatArray = npt.NDArray[numpy.floating[Any]]
LogPrior = Callable[[FloatArray], FloatArray]
MIN_QUADRATURE_NODES = 2


def normal_log_prior(mean: float = 0.0, sd: float = 1.0) -> LogPrior:
  """Return the log-density of a normal prior."""
  if sd <= 0:
    msg = f"sd must be positive, got {sd}"
    raise ValueError(msg)

  var = sd * sd
  norm = -0.5 * numpy.log(2.0 * numpy.pi * var)

  def _log_prior(theta: FloatArray) -> FloatArray:
    return norm - 0.5 * (theta - mean) ** 2 / var

  return _log_prior


def uniform_log_prior(low: float = THETA_MIN_EXTENDED, high: float = THETA_MAX_EXTENDED) -> LogPrior:
  """Return the log-density of a uniform prior."""
  if high <= low:
    msg = f"high must be greater than low, got low={low}, high={high}"
    raise ValueError(msg)

  inv_width = 1.0 / (high - low)

  def _log_prior(theta: FloatArray) -> FloatArray:
    inside = (theta >= low) & (theta <= high)
    return numpy.where(inside, numpy.log(inv_width), -numpy.inf)

  return _log_prior


@dataclass(frozen=True)
class QuadratureGrid:
  """A fixed-node grid used for posterior computation."""

  nodes: FloatArray
  weights: FloatArray

  @classmethod
  def uniform(
    cls,
    n_nodes: int = 41,
    low: float = THETA_MIN_EXTENDED,
    high: float = THETA_MAX_EXTENDED,
  ) -> "QuadratureGrid":
    """Build a uniformly spaced quadrature grid."""
    if n_nodes < MIN_QUADRATURE_NODES:
      msg = f"n_nodes must be at least 2, got {n_nodes}"
      raise ValueError(msg)
    if high <= low:
      msg = f"high must be greater than low, got low={low}, high={high}"
      raise ValueError(msg)

    nodes = numpy.linspace(low, high, n_nodes)
    weights = numpy.full(n_nodes, (high - low) / (n_nodes - 1), dtype=float)
    return cls(nodes=nodes, weights=weights)


def log_likelihood_grid(
  response_vector: list[bool],
  administered_items: FloatArray,
  nodes: FloatArray,
) -> FloatArray:
  """Evaluate the response log-likelihood at each quadrature node."""
  if administered_items.size == 0:
    return numpy.zeros_like(nodes, dtype=float)
  return numpy.array(
    [irt.log_likelihood(float(theta), response_vector, administered_items) for theta in nodes],
    dtype=float,
  )


def posterior(
  response_vector: list[bool],
  administered_items: FloatArray,
  grid: QuadratureGrid,
  log_prior: LogPrior,
) -> FloatArray:
  """Compute a discrete posterior distribution on the provided grid."""
  log_lik = log_likelihood_grid(response_vector, administered_items, grid.nodes)
  log_post = log_prior(grid.nodes) + log_lik + numpy.log(grid.weights)
  m = numpy.max(log_post)
  unnorm = numpy.exp(log_post - m)
  return unnorm / unnorm.sum()


def posterior_mean(post: FloatArray, nodes: FloatArray) -> float:
  """Return the posterior mean on the supplied grid."""
  return float(numpy.sum(nodes * post))


def posterior_variance(post: FloatArray, nodes: FloatArray) -> float:
  """Return the posterior variance on the supplied grid."""
  mean = posterior_mean(post, nodes)
  return float(numpy.sum(((nodes - mean) ** 2) * post))


class MAPEstimator(BaseEstimator):
  """Maximum a Posteriori ability estimator."""

  def __init__(
    self,
    log_prior: LogPrior | None = None,
    tol: float = 1e-6,
    bounds: tuple[float, float] = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
    verbose: bool = False,
  ) -> None:
    super().__init__(verbose=verbose)
    self._log_prior = log_prior if log_prior is not None else normal_log_prior()
    self._tol = float(tol)
    self._bounds = bounds

  def __str__(self) -> str:
    """Return a human-readable estimator name."""
    return "Maximum a Posteriori Estimator"

  def _negative_log_posterior(self, theta: float, response_vector: list[bool], items: FloatArray) -> float:
    """Return the negative log-posterior at a candidate theta."""
    log_lik = irt.log_likelihood(theta, response_vector, items)
    log_pr = float(self._log_prior(numpy.array([theta], dtype=float))[0])
    return -(log_lik + log_pr)

  def estimate(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,  # noqa: ARG002 - required by the estimator contract
  ) -> float:
    """Estimate ability by maximizing the posterior density."""
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
      grid = numpy.linspace(self._bounds[0], self._bounds[1], 201)
      idx = int(numpy.argmax(self._log_prior(grid)))
      return float(grid[idx])

    items = item_bank.items[administered_items, :4]
    result = minimize_scalar(
      self._negative_log_posterior,
      args=(response_vector, items),
      bounds=self._bounds,
      method="bounded",
      options={"xatol": self._tol},
    )
    self._last_evaluations = int(result.nfev)
    self._total_evaluations += self._last_evaluations
    return float(result.x)
