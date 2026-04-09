"""Shared Bayesian helpers for ability estimation in CAT."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy
import numpy.typing as npt

from .. import irt
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED

FloatArray = npt.NDArray[numpy.floating[Any]]
LogPrior = Callable[[FloatArray], FloatArray]
MIN_QUADRATURE_NODES = 2

__all__ = [
  "FloatArray",
  "LogPrior",
  "QuadratureGrid",
  "normal_log_prior",
  "posterior",
  "posterior_mean",
  "posterior_variance",
  "uniform_log_prior",
]


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
  """Return the log-density of a uniform prior on ``[low, high]``."""
  if high <= low:
    msg = f"high must be greater than low, got low={low}, high={high}"
    raise ValueError(msg)

  inv_width = 1.0 / (high - low)
  log_density = float(numpy.log(inv_width))

  def _log_prior(theta: FloatArray) -> FloatArray:
    inside = (theta >= low) & (theta <= high)
    return numpy.where(inside, log_density, -numpy.inf)

  return _log_prior


@dataclass(frozen=True, slots=True)
class QuadratureGrid:
  """A fixed quadrature grid for posterior computations."""

  nodes: FloatArray
  weights: FloatArray

  @classmethod
  def uniform(
    cls,
    n_nodes: int = 41,
    low: float = THETA_MIN_EXTENDED,
    high: float = THETA_MAX_EXTENDED,
  ) -> QuadratureGrid:
    """Create a uniform grid spanning ``[low, high]``."""
    if n_nodes < MIN_QUADRATURE_NODES:
      msg = f"n_nodes must be at least {MIN_QUADRATURE_NODES}, got {n_nodes}"
      raise ValueError(msg)
    if high <= low:
      msg = f"high must be greater than low, got low={low}, high={high}"
      raise ValueError(msg)

    nodes = numpy.linspace(low, high, n_nodes, dtype=float)
    weights = numpy.full(n_nodes, (high - low) / (n_nodes - 1), dtype=float)
    return cls(nodes=nodes, weights=weights)


def log_likelihood_grid(
  response_vector: list[bool],
  administered_items: FloatArray,
  nodes: FloatArray,
) -> FloatArray:
  """Evaluate the log-likelihood at each node in a quadrature grid."""
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
  """Compute a normalized discrete posterior over ``grid.nodes``."""
  log_lik = log_likelihood_grid(response_vector, administered_items, grid.nodes)
  log_post = log_prior(grid.nodes) + log_lik + numpy.log(grid.weights)
  finite = numpy.isfinite(log_post)
  if not finite.any():
    msg = "posterior is undefined on the provided grid"
    raise ValueError(msg)

  max_log_post = float(numpy.max(log_post[finite]))
  unnorm = numpy.exp(log_post - max_log_post)
  total = float(unnorm.sum())
  if total <= 0:
    msg = "posterior normalization failed"
    raise ValueError(msg)
  return unnorm / total


def posterior_mean(post: FloatArray, nodes: FloatArray) -> float:
  """Return the posterior mean over a discrete grid."""
  return float(numpy.sum(nodes * post))


def posterior_variance(post: FloatArray, nodes: FloatArray) -> float:
  """Return the posterior variance over a discrete grid."""
  mean = posterior_mean(post, nodes)
  return float(numpy.sum(((nodes - mean) ** 2) * post))
