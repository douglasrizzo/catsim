"""Shared Bayesian helpers for ability estimation in CAT.

For the full posterior specification and quadrature contract, see
:doc:`/api/techniques/estimation/bayesian-infrastructure`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy
from scipy.optimize import minimize_scalar

from .. import irt
from ..context import Estimate
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..typing import FloatArray
from .base import BaseEstimator

if TYPE_CHECKING:
  from ..context import StepContext

LogPrior = Callable[[FloatArray], FloatArray]
_MIN_QUADRATURE_NODES = 2
# Step of the central difference that gives a prior's curvature at the MAP estimate.
_CURVATURE_STEP = 1e-4

__all__ = [
  "EAPEstimator",
  "LogPrior",
  "MAPEstimator",
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
  responses: Sequence[bool],
  items: FloatArray,
  grid: QuadratureGrid,
  log_prior: LogPrior,
) -> FloatArray:
  r"""Compute the normalized discrete posterior over ``grid.nodes``.

  Each node gets mass proportional to :math:`w_q\,p(\theta_q)\,L(\mathbf{u} \mid \theta_q)`,
  computed in log space and shifted by its maximum before exponentiating, so the likelihood of
  many responses does not underflow.

  Parameters
  ----------
  responses : Sequence[bool]
      Responses to the administered items, in the same order as ``items``.
  items : numpy.ndarray
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
      zero posterior mass, if ``responses`` and ``items`` differ in length, or if ``responses``
      contains non-Boolean elements.
  """
  log_lik = irt.log_likelihood(grid.nodes, responses, items)
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
  """

  def __init__(
    self,
    grid: QuadratureGrid | None = None,
    log_prior: LogPrior | None = None,
  ) -> None:
    # Kept as given for the repr, where a default shows as None: the default grid's repr would list every
    # node, and a prior function has no source form.
    self._grid_argument = grid
    self._log_prior_argument = log_prior
    self._grid = grid if grid is not None else QuadratureGrid.uniform()
    self._log_prior = log_prior if log_prior is not None else normal_log_prior()

  def __repr__(self) -> str:
    """Return the estimator as a constructor call that names every argument, with None for a default."""
    return f"{type(self).__name__}(grid={self._grid_argument!r}, log_prior={self._log_prior_argument!r})"

  def estimate(self, context: StepContext) -> Estimate:
    r"""Return the posterior mean and the posterior standard deviation.

    With no administered item they are the prior's mean and standard deviation on the grid. The
    standard error is the posterior standard deviation (:cite:t:`Magis12`, Eq. 11, p. 6), as in
    catR. Each call evaluates the likelihood once per grid node, which is its evaluation count.

    Parameters
    ----------
    context : StepContext
        The session to estimate for. Its ``theta`` and ``se`` are not used.

    Returns
    -------
    Estimate
        The posterior mean of :math:`\theta`, the posterior standard deviation and the number of
        grid nodes.
    """
    items = context.item_bank.get_items(context.administered_items)
    nodes = self._grid.nodes
    post = posterior(context.responses, items, self._grid, self._log_prior)
    return Estimate(
      theta=posterior_mean(post, nodes),
      se=math.sqrt(posterior_variance(post, nodes)),
      evaluations=int(nodes.size),
    )

  @property
  def grid(self) -> QuadratureGrid:
    """Return the quadrature grid used by the estimator."""
    return self._grid


# Points on which MAPEstimator locates the posterior mode before refining it (0.06 apart on [-6, 6]).
_MODE_SEARCH_POINTS = 201
# Equal maxima on this many search points mean a flat posterior; two can flank a symmetric peak.
_FLAT_MAXIMUM_POINTS = 3


class MAPEstimator(BaseEstimator):
  """Maximum a Posteriori ability estimator.

  The posterior mode is located on 201 equally spaced points over the search bounds, then
  refined with a bounded scalar search between the best point's neighbors. The grid keeps the
  estimate inside the prior's support and finds the global mode of a multimodal posterior.

  For full algorithmic details, see :doc:`/api/techniques/estimation/map-estimator`.

  Parameters
  ----------
  log_prior : LogPrior or None, optional
      Vectorized log-density of the prior. Default is ``normal_log_prior()``, the standard normal.
  tol : float, optional
      Absolute tolerance of the refinement step. Default is 1e-6.
  bounds : tuple[float, float], optional
      Search range for the mode. Default is ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``.

  Raises
  ------
  ValueError
      If ``tol`` is not positive, or if ``bounds`` is not a finite interval with low < high.
  """

  def __init__(
    self,
    log_prior: LogPrior | None = None,
    tol: float = 1e-6,
    bounds: tuple[float, float] = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
  ) -> None:
    # Kept as given for the repr, where the default shows as None: a prior function has no source form.
    self._log_prior_argument = log_prior
    self._log_prior = log_prior if log_prior is not None else normal_log_prior()
    self._tol = self._check_tol(tol)
    self._bounds = self._check_bounds(bounds)

  def __repr__(self) -> str:
    """Return the estimator as a constructor call that names every argument, with None for a default prior."""
    return f"{type(self).__name__}(log_prior={self._log_prior_argument!r}, tol={self._tol!r}, bounds={self._bounds!r})"

  def _negative_log_posterior(self, theta: float, responses: Sequence[bool], items: FloatArray) -> float:
    """Return the negative log-posterior at a candidate theta."""
    log_lik = irt.log_likelihood(theta, responses, items)
    log_pr = float(self._log_prior(numpy.array([theta], dtype=float))[0])
    return -(log_lik + log_pr)

  def _standard_error(self, theta: float, items: FloatArray) -> float:
    r"""Return the standard error of a posterior mode, :math:`1/\sqrt{I(\hat\theta) - (\log p)''(\hat\theta)}`.

    The prior's curvature is a central difference, so any :data:`LogPrior` works. Where it is not
    finite, as at the edge of a bounded prior's support, the prior adds nothing and the error is
    the Fisher one. Where the sum is not positive, the error is ``math.inf``.

    Parameters
    ----------
    theta : float
        The posterior mode.
    items : numpy.ndarray
        Parameters of the administered items.

    Returns
    -------
    float
        The standard error, finite and positive, or ``math.inf``.
    """
    h = _CURVATURE_STEP
    below, at, above = self._log_prior(numpy.array([theta - h, theta, theta + h], dtype=float))
    curvature = float((above - 2.0 * at + below) / h**2)
    precision = irt.total_information(theta, items) - (curvature if math.isfinite(curvature) else 0.0)
    return 1.0 / math.sqrt(precision) if precision > 0.0 and math.isfinite(precision) else math.inf

  def estimate(self, context: StepContext) -> Estimate:
    r"""Return the posterior mode within the search bounds, the prior mode when no item has been administered.

    The standard error is :math:`1/\sqrt{I(\hat\theta) - (\log p)''(\hat\theta)}`, the test
    information plus the prior's negative curvature at the mode, taken by a central difference so
    any prior works. For a normal prior with variance :math:`\sigma^2` that is
    :math:`1/\sqrt{I(\hat\theta) + 1/\sigma^2}` (:cite:t:`Magis12`, Eq. 6, p. 5; catR's
    default), and inside a uniform prior's support the Fisher standard error. The evaluation count
    is the number of search points plus the refinement's posterior evaluations.

    Parameters
    ----------
    context : StepContext
        The session to estimate for. Its ``theta`` and ``se`` are not used: the search covers the
        whole bounds on every call.

    Returns
    -------
    Estimate
        The posterior mode of :math:`\theta` within the search bounds, its standard error and the
        number of posterior evaluations.

    Raises
    ------
    ValueError
        If the log-posterior has no finite maximum within the search bounds.
    """
    responses = context.responses
    items = context.item_bank.get_items(context.administered_items)
    nodes = numpy.linspace(self._bounds[0], self._bounds[1], _MODE_SEARCH_POINTS)
    log_post = irt.log_likelihood(nodes, responses, items) + self._log_prior(nodes)
    peak = log_post.max()
    if not numpy.isfinite(peak):
      msg = (
        f"the log-posterior is not finite at any of the {nodes.size} search points in {self._bounds}; "
        "the prior's support must contain at least one of them"
      )
      raise ValueError(msg)

    ties = numpy.flatnonzero(log_post == peak)
    best = int(ties[ties.size // 2])
    # A flat maximum, such as a uniform prior with no responses, has no unique mode
    if ties.size >= _FLAT_MAXIMUM_POINTS:
      theta = float(nodes[best])
      return Estimate(theta=theta, se=self._standard_error(theta, items), evaluations=int(nodes.size))

    # Past a narrow prior's support the objective is +inf, and SciPy's parabolic step then
    # subtracts infinities; the result is checked against the best point below
    with numpy.errstate(invalid="ignore"):
      result = minimize_scalar(
        self._negative_log_posterior,
        args=(responses, items),
        bounds=(nodes[max(best - 1, 0)], nodes[min(best + 1, nodes.size - 1)]),
        method="bounded",
        options={"xatol": self._tol},
      )
    # A neighbour can lie outside a narrow prior's support, so keep the refinement only if it
    # did at least as well as the best node; the estimate then always lies inside the support
    theta = float(result.x) if result.fun <= -peak else float(nodes[best])
    return Estimate(theta=theta, se=self._standard_error(theta, items), evaluations=int(nodes.size + result.nfev))
