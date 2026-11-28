"""Shared Bayesian helpers for ability estimation in CAT.

For the full posterior specification and quadrature contract, see
:doc:`/api/techniques/estimation/bayesian-infrastructure`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, override

import numpy
from irtlib import irt
from irtlib.irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from irtlib.typing import FloatArray
from scipy.optimize import minimize_scalar

from ..context import Estimate
from .base import BaseEstimator

if TYPE_CHECKING:
  from ..context import StepContext

LogPrior = Callable[[FloatArray], FloatArray]
"""A log prior density: maps an array of abilities to their log-densities, elementwise."""

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
      A vectorized callable mapping ability values to their log-density under
      :math:`\mathcal{N}(\text{mean}, \text{sd}^2)`. It pickles, equals any prior this function
      returns for the same arguments, and its repr is the call that builds it.

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
  return _NormalLogPrior(float(mean), float(sd))


# The priors are module-level classes rather than closures so that an estimator holding one pickles, as an engine sent
# to a worker process must (ADR-0015).
@dataclass(frozen=True, slots=True)
class _NormalLogPrior:
  """The normal log-density that :func:`normal_log_prior` returns."""

  mean: float
  sd: float

  def __call__(self, theta: FloatArray) -> FloatArray:
    """Return the log-density at each ability."""
    var = self.sd * self.sd
    return -0.5 * numpy.log(2.0 * numpy.pi * var) - 0.5 * (theta - self.mean) ** 2 / var

  @override
  def __repr__(self) -> str:
    """Return the call to :func:`normal_log_prior` that builds this prior."""
    return f"normal_log_prior(mean={self.mean!r}, sd={self.sd!r})"


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
      A vectorized callable returning :math:`-\log(\text{high} - \text{low})` inside the
      bounds and :math:`-\infty` outside them. It pickles, equals any prior this function returns
      for the same arguments, and its repr is the call that builds it.

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
  return _UniformLogPrior(float(low), float(high))


@dataclass(frozen=True, slots=True)
class _UniformLogPrior:
  """The uniform log-density that :func:`uniform_log_prior` returns."""

  low: float
  high: float

  def __call__(self, theta: FloatArray) -> FloatArray:
    """Return the log-density at each ability: minus the log of the width inside the bounds, -inf outside."""
    inside = (theta >= self.low) & (theta <= self.high)
    return numpy.where(inside, float(numpy.log(1.0 / (self.high - self.low))), -numpy.inf)

  @override
  def __repr__(self) -> str:
    """Return the call to :func:`uniform_log_prior` that builds this prior."""
    return f"uniform_log_prior(low={self.low!r}, high={self.high!r})"


# eq=False: generated equality and hashing would compare the arrays, which raises; a grid is compared by identity
@dataclass(frozen=True, slots=True, eq=False)
class QuadratureGrid:
  r"""A fixed quadrature grid for posterior computations.

  A node's posterior mass is proportional to its weight times the prior density and the likelihood
  there, so the weights take part in the posterior itself; equal weights, as :meth:`uniform` builds,
  cancel in the normalization. Two grids are equal only if they are the same object, and a grid can
  be a set member or a dictionary key.

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

  Each node's mass is proportional to its weight times the prior density and the likelihood there,
  computed in log space and shifted by its maximum before exponentiating, so the likelihood of many
  responses does not underflow.

  Parameters
  ----------
  responses : Sequence[bool]
      Responses to the administered items, in the same order as ``items``: each a Boolean or the
      integer 0 or 1.
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
      zero posterior mass, if ``responses`` and ``items`` differ in length, or if an integer response
      is not 0 or 1.
  TypeError
      If a response is neither a Boolean nor an integer.
  """
  log_lik = irt.log_likelihood(grid.nodes, [responses], items)[0]
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
      The probability-weighted mean of the nodes.
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
      The probability-weighted mean of the squared distances between the nodes and
      :func:`posterior_mean`.
  """
  mean = posterior_mean(post, nodes)
  return float(numpy.sum(((nodes - mean) ** 2) * post))


class EAPEstimator(BaseEstimator):
  """Expected a Posteriori ability estimator.

  The mean of the ability's posterior over a quadrature grid, with the posterior standard deviation
  as its standard error; see :doc:`/api/techniques/estimation/eap-estimator`. The estimator keeps no
  posterior between calls: :py:func:`posterior` computes a session's posterior when it is needed.

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
    # Kept as given for the repr, where a default shows as None: the default grid's repr would list every node.
    self._grid_argument = grid
    self._log_prior_argument = log_prior
    self._grid = grid if grid is not None else QuadratureGrid.uniform()
    self._log_prior = log_prior if log_prior is not None else normal_log_prior()

  @override
  def __repr__(self) -> str:
    """Return the estimator as a constructor call that names every argument, with None for a default."""
    return f"{type(self).__name__}(grid={self._grid_argument!r}, log_prior={self._log_prior_argument!r})"

  @override
  def estimate(self, context: StepContext) -> Estimate:
    r"""Return the posterior mean and the posterior standard deviation.

    With no administered item they are the prior's mean and standard deviation on the grid. Each
    call evaluates the likelihood once per grid node, which is its evaluation count.

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

  The mode of the ability's posterior within the search bounds; see
  :doc:`/api/techniques/estimation/map-estimator`. The search finds the global mode of a multimodal
  posterior and keeps the estimate inside the prior's support.

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
    # Kept as given for the repr, which shows the default as None, as EAPEstimator's does.
    self._log_prior_argument = log_prior
    self._log_prior = log_prior if log_prior is not None else normal_log_prior()
    self._tol = self._check_tol(tol)
    self._bounds = self._check_bounds(bounds)

  @override
  def __repr__(self) -> str:
    """Return the estimator as a constructor call that names every argument, with None for a default prior."""
    return f"{type(self).__name__}(log_prior={self._log_prior_argument!r}, tol={self._tol!r}, bounds={self._bounds!r})"

  def _negative_log_posterior(
    self, theta: float, log_likelihood: Callable[[float | FloatArray], float | FloatArray]
  ) -> float:
    """Return the negative log-posterior at a candidate theta."""
    log_lik = float(log_likelihood(theta))
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

  @override
  def estimate(self, context: StepContext) -> Estimate:
    r"""Return the posterior mode within the search bounds, the prior mode when no item has been administered.

    The standard error is the inverse square root of the test information plus the prior's negative
    curvature at the mode, which a central difference gives for any prior; inside a uniform prior's
    support it is the Fisher standard error. The evaluation count is the number of search points
    plus the refinement's posterior evaluations.

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
    items = context.item_bank.get_items(context.administered_items)
    examinee_log_likelihood = irt.log_likelihood_function([context.responses], items)

    def log_likelihood(theta: float | FloatArray) -> FloatArray:
      return examinee_log_likelihood(theta)[0]

    nodes = numpy.linspace(self._bounds[0], self._bounds[1], _MODE_SEARCH_POINTS)
    log_post = log_likelihood(nodes) + self._log_prior(nodes)
    peak = numpy.max(log_post)
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
        args=(log_likelihood,),
        bounds=(nodes[max(best - 1, 0)], nodes[min(best + 1, nodes.size - 1)]),
        method="bounded",
        options={"xatol": self._tol},
      )
    # A neighbour can lie outside a narrow prior's support, so keep the refinement only if it
    # did at least as well as the best node; the estimate then always lies inside the support
    theta = float(result.x) if result.fun <= -peak else float(nodes[best])
    return Estimate(theta=theta, se=self._standard_error(theta, items), evaluations=int(nodes.size + result.nfev))
