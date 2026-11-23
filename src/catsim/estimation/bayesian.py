"""Shared Bayesian helpers for ability estimation in CAT.

For the full posterior specification and quadrature contract, see
:doc:`/api/techniques/estimation/bayesian-infrastructure`.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy

from .. import irt
from .._typing import FloatArray
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from .base import BaseEstimator

if TYPE_CHECKING:
  from ..item_bank import ItemBank

LogPrior = Callable[[FloatArray], FloatArray]
_MIN_QUADRATURE_NODES = 2

__all__ = [
  "EAPEstimator",
  "LogPrior",
  "QuadratureGrid",
  "normal_log_prior",
  "posterior",
  "posterior_mean",
  "posterior_variance",
  "uniform_log_prior",
]


def normal_log_prior(mean: float = 0.0, sd: float = 1.0) -> LogPrior:
  r"""Return the log-density of a normal prior.

  Parameters
  ----------
  mean : float, optional
      Mean of the prior. Default is 0.
  sd : float, optional
      Standard deviation of the prior. Default is 1.

  Returns
  -------
  LogPrior
      A vectorized function mapping ability values to their log-density under
      :math:`\mathcal{N}(\text{mean}, \text{sd}^2)`.

  Raises
  ------
  ValueError
      If ``mean`` is not finite, or ``sd`` is not finite and positive.
  """
  if not math.isfinite(mean):
    msg = f"mean must be finite, got {mean}"
    raise ValueError(msg)
  if not (math.isfinite(sd) and sd > 0):
    msg = f"sd must be positive and finite, got {sd}"
    raise ValueError(msg)

  var = sd * sd
  norm = -0.5 * numpy.log(2.0 * numpy.pi * var)

  def _log_prior(theta: FloatArray) -> FloatArray:
    return norm - 0.5 * (theta - mean) ** 2 / var

  return _log_prior


def uniform_log_prior(low: float = THETA_MIN_EXTENDED, high: float = THETA_MAX_EXTENDED) -> LogPrior:
  r"""Return the log-density of a uniform prior on ``[low, high]``.

  Parameters
  ----------
  low : float, optional
      Lower bound of the prior's support. Default is ``THETA_MIN_EXTENDED``.
  high : float, optional
      Upper bound of the prior's support. Default is ``THETA_MAX_EXTENDED``.

  Returns
  -------
  LogPrior
      A vectorized function returning :math:`-\log(\text{high} - \text{low})` inside the
      bounds and :math:`-\infty` outside them.

  Raises
  ------
  ValueError
      If either bound is not finite or ``high <= low``.
  """
  if not (math.isfinite(low) and math.isfinite(high)):
    msg = f"low and high must be finite, got low={low}, high={high}"
    raise ValueError(msg)
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
  r"""A fixed quadrature grid for posterior computations.

  The posterior at node :math:`\theta_q` is proportional to
  :math:`w_q\,p(\theta_q)\,L(\mathbf{u} \mid \theta_q)`, so the weights take part in the
  posterior itself. Equal weights, as built by :meth:`uniform`, cancel in the normalization.

  For the full posterior specification and quadrature contract, see
  :doc:`/api/techniques/estimation/bayesian-infrastructure`.

  Parameters
  ----------
  nodes : numpy.ndarray
      One-dimensional array of finite ability values.
  weights : numpy.ndarray
      One-dimensional array of finite, positive quadrature weights, one per node.

  Raises
  ------
  ValueError
      If the arrays are not one-dimensional, differ in length, have fewer than
      two entries, or hold non-finite nodes or non-positive weights.
  """

  nodes: FloatArray
  weights: FloatArray

  def __post_init__(self) -> None:
    """Validate the nodes and weights."""
    if self.nodes.ndim != 1 or self.weights.ndim != 1:
      msg = "nodes and weights must be one-dimensional"
      raise ValueError(msg)
    if self.nodes.shape != self.weights.shape:
      msg = f"nodes and weights must have the same length, got {self.nodes.size} and {self.weights.size}"
      raise ValueError(msg)
    if self.nodes.size < _MIN_QUADRATURE_NODES:
      msg = f"a grid needs at least {_MIN_QUADRATURE_NODES} nodes, got {self.nodes.size}"
      raise ValueError(msg)
    if not numpy.isfinite(self.nodes).all():
      msg = "nodes must be finite"
      raise ValueError(msg)
    if not (numpy.isfinite(self.weights).all() and (self.weights > 0).all()):
      msg = "weights must be finite and positive"
      raise ValueError(msg)

  @classmethod
  def uniform(
    cls,
    n_nodes: int = 121,
    low: float = THETA_MIN_EXTENDED,
    high: float = THETA_MAX_EXTENDED,
  ) -> QuadratureGrid:
    """Create a grid of equally spaced nodes with equal weights spanning ``[low, high]``.

    Parameters
    ----------
    n_nodes : int, optional
        Number of nodes, including both bounds. Default is 121, a spacing of 0.1 on the
        default bounds.
    low : float, optional
        First node. Default is ``THETA_MIN_EXTENDED``.
    high : float, optional
        Last node. Default is ``THETA_MAX_EXTENDED``.

    Returns
    -------
    QuadratureGrid
        The grid, with every weight equal to the node spacing.

    Raises
    ------
    ValueError
        If ``n_nodes < 2`` or ``high <= low``.
    """
    if n_nodes < _MIN_QUADRATURE_NODES:
      msg = f"n_nodes must be at least {_MIN_QUADRATURE_NODES}, got {n_nodes}"
      raise ValueError(msg)
    if high <= low:
      msg = f"high must be greater than low, got low={low}, high={high}"
      raise ValueError(msg)

    nodes = numpy.linspace(low, high, n_nodes, dtype=float)
    weights = numpy.full(n_nodes, (high - low) / (n_nodes - 1), dtype=float)
    return cls(nodes=nodes, weights=weights)


def posterior(
  response_vector: list[bool],
  administered_items: FloatArray,
  grid: QuadratureGrid,
  log_prior: LogPrior,
) -> FloatArray:
  r"""Compute the normalized discrete posterior over ``grid.nodes``.

  Each node gets mass proportional to :math:`w_q\,p(\theta_q)\,L(\mathbf{u} \mid \theta_q)`,
  computed in log space and shifted by its maximum before exponentiating, so long response
  vectors do not underflow.

  Parameters
  ----------
  response_vector : list[bool]
      Responses to the administered items, in the same order as ``administered_items``.
  administered_items : numpy.ndarray
      Parameter matrix of the administered items, one row per item.
  grid : QuadratureGrid
      Nodes and weights of the discrete approximation.
  log_prior : LogPrior
      Vectorized log-density of the prior.

  Returns
  -------
  numpy.ndarray
      Posterior probabilities, one per node, summing to 1.

  Raises
  ------
  ValueError
      If the log-prior or log-likelihood is NaN or :math:`+\infty` at any node, if every node has
      zero posterior mass, if ``response_vector`` and ``administered_items`` differ in length, or
      if ``response_vector`` contains non-Boolean elements.
  """
  log_lik = irt.log_likelihood_hpc(grid.nodes, response_vector, administered_items)
  log_post = log_prior(grid.nodes) + log_lik + numpy.log(grid.weights)
  if numpy.isnan(log_post).any() or numpy.isposinf(log_post).any():
    msg = "posterior is undefined: the log-prior or log-likelihood is NaN or +inf at some node"
    raise ValueError(msg)
  finite = numpy.isfinite(log_post)
  if not finite.any():
    msg = "posterior is undefined on the provided grid: every node has zero mass"
    raise ValueError(msg)

  unnorm = numpy.exp(log_post - float(numpy.max(log_post[finite])))
  return unnorm / float(unnorm.sum())


def posterior_mean(post: FloatArray, nodes: FloatArray) -> float:
  r"""Return the mean of a discrete posterior.

  Parameters
  ----------
  post : numpy.ndarray
      Normalized posterior probabilities, as returned by :func:`posterior`.
  nodes : numpy.ndarray
      The nodes the probabilities refer to.

  Returns
  -------
  float
      :math:`\sum_q \theta_q\,\tilde{p}_q`.
  """
  return float(numpy.sum(nodes * post))


def posterior_variance(post: FloatArray, nodes: FloatArray) -> float:
  r"""Return the variance of a discrete posterior.

  Parameters
  ----------
  post : numpy.ndarray
      Normalized posterior probabilities, as returned by :func:`posterior`.
  nodes : numpy.ndarray
      The nodes the probabilities refer to.

  Returns
  -------
  float
      :math:`\sum_q (\theta_q - \bar{\theta})^2\,\tilde{p}_q`, where :math:`\bar{\theta}` is
      :func:`posterior_mean`.
  """
  mean = posterior_mean(post, nodes)
  return float(numpy.sum(((nodes - mean) ** 2) * post))


class EAPEstimator(BaseEstimator):
  """Expected a Posteriori ability estimator.

  The estimator keeps no posterior between calls: one instance serves every session of a
  simulation, and look-ahead selectors such as MEI call it with hypothetical responses. Compute
  a session's posterior with :py:func:`posterior` when it is needed.

  For full algorithmic details, see :doc:`/api/techniques/estimation/eap-estimator`.

  Parameters
  ----------
  grid : QuadratureGrid or None, optional
      Quadrature grid over which the posterior is evaluated. Default is
      ``QuadratureGrid.uniform()``.
  log_prior : LogPrior or None, optional
      Vectorized log-density of the prior. Default is ``normal_log_prior()``, the standard normal.
  verbose : bool, optional
      Diagnostics flag inherited from :py:class:`BaseEstimator`. Default is False.
  """

  def __init__(
    self,
    grid: QuadratureGrid | None = None,
    log_prior: LogPrior | None = None,
    verbose: bool = False,
  ) -> None:
    super().__init__(verbose=verbose)
    self._grid = grid if grid is not None else QuadratureGrid.uniform()
    self._log_prior = log_prior if log_prior is not None else normal_log_prior()

  def __str__(self) -> str:
    """Return a human-readable name for the estimator."""
    return "Expected a Posteriori Estimator"

  def estimate(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,  # noqa: ARG002
  ) -> float:
    r"""Return the posterior mean for the administered response pattern.

    Each call evaluates the likelihood once per grid node, which the ``evaluations`` counters
    report.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        Indices of the items administered so far. With none, the estimate is the prior mean.
    response_vector : list[bool]
        The examinee's answers to the administered items, in the same order.
    est_theta : float
        Unused; the posterior mean does not depend on a previous estimate.

    Returns
    -------
    float
        The posterior mean of :math:`\theta`.

    Raises
    ------
    ValueError
        If ``response_vector`` and ``administered_items`` have different lengths, or if
        ``response_vector`` contains non-Boolean elements.
    """
    post = posterior(response_vector, item_bank.items[administered_items, :4], self._grid, self._log_prior)
    self._calls += 1
    self._last_evaluations = self._grid.nodes.size
    self._total_evaluations += self._last_evaluations
    return posterior_mean(post, self._grid.nodes)

  @property
  def grid(self) -> QuadratureGrid:
    """Return the quadrature grid used by the estimator."""
    return self._grid
