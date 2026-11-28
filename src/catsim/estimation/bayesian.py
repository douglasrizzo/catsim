"""The Bayesian ability estimators, EAP and MAP, over irtlib's quadrature grid, priors and posterior."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, override

import numpy
from irtlib import irt
from irtlib.irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from irtlib.quadrature import LogPrior, QuadratureGrid, normal_log_prior, posterior, posterior_mean, posterior_variance
from scipy.optimize import minimize_scalar

from ..context import Estimate
from .base import BaseEstimator

if TYPE_CHECKING:
  from collections.abc import Callable

  from irtlib.typing import FloatArray

  from ..context import StepContext

# Step of the central difference that gives a prior's curvature at the MAP estimate.
_CURVATURE_STEP = 1e-4

__all__ = ["EAPEstimator", "MAPEstimator"]


class EAPEstimator(BaseEstimator):
  """Expected a Posteriori ability estimator.

  The mean of the ability's posterior over a quadrature grid, with the posterior standard deviation
  as its standard error; see :doc:`/api/techniques/estimation/eap-estimator`. The estimator keeps no
  posterior between calls: it computes a session's posterior when it is needed.

  Parameters
  ----------
  grid : QuadratureGrid or None, optional
      irtlib's quadrature grid over which the posterior is evaluated. Default is
      ``QuadratureGrid.uniform()``, 121 nodes on [-6, 6].
  log_prior : LogPrior or None, optional
      Vectorized log-density of the prior, such as irtlib's ``normal_log_prior`` or
      ``uniform_log_prior``. Default is ``normal_log_prior()``, the standard normal.
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

    Raises
    ------
    ValueError
        If the prior is NaN or :math:`+\infty` at a node of the grid, or the posterior has no mass at
        any node: a prior whose support misses the grid, or responses whose likelihood is 0 at every
        node.
    """
    items = context.item_bank.get_items(context.administered_items)
    nodes = self._grid.nodes
    post = posterior(irt.log_likelihood(nodes, [context.responses], items), self._grid, self._log_prior)
    return Estimate(
      theta=float(posterior_mean(post, nodes)[0]),
      se=math.sqrt(posterior_variance(post, nodes)[0]),
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

    The prior's curvature is a central difference, so any ``LogPrior`` works. Where it is not
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
