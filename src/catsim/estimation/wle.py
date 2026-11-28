"""Warm's weighted likelihood estimator, which irtlib's estimator computes for a session."""

from __future__ import annotations

from typing import TYPE_CHECKING, override

from irtlib import ability
from irtlib.irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED

from ..context import Estimate
from .base import BaseEstimator, _session_estimate

if TYPE_CHECKING:
  from ..context import StepContext


class WarmLikelihoodEstimator(BaseEstimator):
  r"""Weighted likelihood estimator (WLE) of :cite:t:`Warm1989`.

  The root of Warm's modified score equation, which removes the MLE's first-order bias and stays
  finite for all-correct and all-incorrect responses; see
  :doc:`/api/techniques/estimation/warm-wle`. If the equation has several roots within ``bounds``,
  the one with the largest weighted likelihood is returned; if it has none, the bound the modified
  score points to.

  Parameters
  ----------
  tol : float, optional
      Absolute tolerance of the root refinement. Must be positive. Default is 1e-6.
  bounds : tuple[float, float], optional
      Range of the estimate. Default is ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``.

  Raises
  ------
  ValueError
      If ``tol`` is not positive, or if ``bounds`` is not a finite interval with low < high.
  """

  def __init__(
    self,
    tol: float = 1e-6,
    bounds: tuple[float, float] = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
  ) -> None:
    self._tol = self._check_tol(tol)
    self._bounds = self._check_bounds(bounds)

  @override
  def __repr__(self) -> str:
    """Return the estimator as a constructor call that names every argument."""
    return f"{type(self).__name__}(tol={self._tol!r}, bounds={self._bounds!r})"

  @override
  def estimate(self, context: StepContext) -> Estimate:
    r"""Return the root of Warm's modified score equation, or ``context.theta`` when no item has been administered.

    The standard error is the inverse square root of the test information at the estimate, without
    Warm's correction (the technique page compares the two), or ``math.inf`` where the information
    is 0. The evaluation count is the number of modified-score evaluations. A ranking integral that
    does not converge warns with ``scipy.integrate.IntegrationWarning``. With no administered
    item, the estimate is ``context.theta`` and ``context.se``, unchanged, after no evaluation.

    Parameters
    ----------
    context : StepContext
        The session to estimate for.

    Returns
    -------
    Estimate
        The weighted likelihood estimate, within ``bounds``, its standard error and the number of
        modified-score evaluations.
    """
    if not context.administered_items:
      return Estimate(theta=float(context.theta), se=context.se, evaluations=0)

    items = context.item_bank.get_items(context.administered_items)
    return _session_estimate(
      ability.weighted_likelihood([context.responses], items, bounds=self._bounds, tol=self._tol)
    )
