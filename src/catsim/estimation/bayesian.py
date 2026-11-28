"""The Bayesian ability estimators, EAP and MAP, which irtlib's estimators compute for a session."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from irtlib import ability
from irtlib.irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from irtlib.quadrature import GridPrior, LogPrior, QuadratureGrid, normal_log_prior

from .base import BaseEstimator, _session_estimate

if TYPE_CHECKING:
  from ..context import Estimate, StepContext

__all__ = ["EAPEstimator", "MAPEstimator"]


class EAPEstimator(BaseEstimator):
  r"""Expected a Posteriori ability estimator.

  The mean of the ability's posterior over a quadrature grid, with the posterior standard deviation
  as its standard error; see :doc:`/api/techniques/estimation/eap-estimator`. The estimator copies
  the grid's nodes and evaluates the prior on them once, when it is built, so changing the grid's
  arrays afterwards does not change it. It keeps no posterior between calls: it computes a session's
  posterior when it is needed.

  Parameters
  ----------
  grid : QuadratureGrid or None, optional
      irtlib's quadrature grid over which the posterior is evaluated. Default is
      ``QuadratureGrid.uniform()``, 121 nodes on [-6, 6].
  log_prior : LogPrior or None, optional
      Vectorized log-density of the prior, such as irtlib's ``normal_log_prior`` or
      ``uniform_log_prior``. Default is ``normal_log_prior()``, the standard normal.

  Raises
  ------
  ValueError
      If the prior is NaN or :math:`+\infty` at a node of the grid, or has no mass at any node: a
      prior whose support misses the grid.
  """

  def __init__(
    self,
    grid: QuadratureGrid | None = None,
    log_prior: LogPrior | None = None,
  ) -> None:
    # Kept as given for the repr, where a default shows as None: the default grid's repr would list every node.
    self._grid_argument = grid
    self._log_prior_argument = log_prior
    self._grid_prior = GridPrior(
      grid if grid is not None else QuadratureGrid.uniform(),
      log_prior if log_prior is not None else normal_log_prior(),
    )

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
        If the posterior has no mass at any node: responses whose likelihood is 0 at every node.
    """
    items = context.item_bank.get_items(context.administered_items)
    estimates = ability.expected_a_posteriori([context.responses], items, grid_prior=self._grid_prior)
    return _session_estimate(estimates)

  @property
  def grid(self) -> QuadratureGrid:
    """Return the quadrature grid used by the estimator."""
    return self._grid_prior.grid


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

  @override
  def estimate(self, context: StepContext) -> Estimate:
    r"""Return the posterior mode within the search bounds, the prior mode when no item has been administered.

    The standard error is the inverse square root of the test information plus the prior's negative
    curvature at the mode, which a central difference gives for any prior; inside a uniform prior's
    support it is the Fisher standard error. The evaluation count is the number of evaluations of
    the log-posterior or of its derivatives.

    Parameters
    ----------
    context : StepContext
        The session to estimate for. Its ``theta`` and ``se`` are not used: the search covers the
        whole bounds on every call.

    Returns
    -------
    Estimate
        The posterior mode of :math:`\theta` within the search bounds, its standard error and the
        number of evaluations of the log-posterior or of its derivatives.

    Raises
    ------
    ValueError
        If the log-posterior has no finite maximum within the search bounds.
    """
    items = context.item_bank.get_items(context.administered_items)
    estimates = ability.maximum_a_posteriori(
      [context.responses], items, log_prior=self._log_prior, bounds=self._bounds, tol=self._tol
    )
    return _session_estimate(estimates)
