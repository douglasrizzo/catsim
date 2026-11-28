"""The maximum likelihood estimator, with Dodd's step while the responses are all correct or all incorrect."""

from typing import override

import numpy
from irtlib import ItemBank, ability, irt
from irtlib.irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED

from ..context import Estimate, StepContext
from .base import BaseEstimator, _session_estimate


def _dodd(theta: float, item_bank: ItemBank, *, correct: bool, bounds: tuple[float, float]) -> float:
  r"""Return Dodd's estimate after responses that are all correct or all incorrect.

  Dodd's variable step size :cite:p:`Dod90` moves the estimate half the distance to the bank's
  largest difficulty after correct answers, or to its smallest after incorrect ones. Beyond that
  difficulty the step would point back toward it, so the estimate stays where it is instead,
  and the result is clipped to ``bounds``:

  .. math::

      \hat{\theta}_{t+1} = \left\lbrace \begin{array}{ll}
      \max\left(\hat{\theta}_t, \hat{\theta}_t + \frac{b_{max} - \hat{\theta}_t}{2}\right)
        & \text{if all responses are correct} \\
      \min\left(\hat{\theta}_t, \hat{\theta}_t - \frac{\hat{\theta}_t - b_{min}}{2}\right)
        & \text{if all responses are incorrect}
      \end{array} \right.

  Parameters
  ----------
  theta : float
      The current ability estimate.
  item_bank : ItemBank
      The item bank, whose largest and smallest difficulties are the targets of the step.
  correct : bool
      Whether every response so far is correct (True) or incorrect (False).
  bounds : tuple[float, float]
      The interval the estimate is clipped to.

  Returns
  -------
  float
      The updated ability estimate, inside ``bounds``.
  """
  difficulty = item_bank.difficulty
  if correct:
    estimate = max(theta, theta + (float(numpy.max(difficulty)) - theta) / 2)
  else:
    estimate = min(theta, theta - (theta - float(numpy.min(difficulty))) / 2)
  low, high = bounds
  return float(min(max(estimate, low), high))


class MLEstimator(BaseEstimator):
  """Estimate ability by numerically maximizing the IRT log-likelihood.

  The maximum is irtlib's maximum likelihood estimate within ``bounds``, ``irtlib.ability.maximum_likelihood``,
  which finds the highest of several maxima. While every response is correct, or every response is
  incorrect, the likelihood has no maximum, and the estimate takes Dodd's step :cite:p:`Dod90`
  toward the bank's hardest or easiest item instead.

  Parameters
  ----------
  tol : float, optional
      Absolute tolerance of the search, in the ability scale. Default is 1e-6.
  bounds : tuple[float, float], optional
      Range of the estimate. Default is ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``.

  Raises
  ------
  ValueError
      If ``tol`` is not positive, or if ``bounds`` is not a finite interval with low < high.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/estimation/ml-estimator`.
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
    r"""Return the maximum likelihood estimate, or Dodd's step from ``context.theta`` while every response is the same.

    The standard error is the inverse square root of the test information at the estimate, or
    ``math.inf`` where the information is 0. The evaluation count is the number of evaluations of
    the log-likelihood or of its derivatives; Dodd's step makes none. With no administered item,
    the estimate is ``context.theta`` and ``context.se``, unchanged, after no evaluation.

    Parameters
    ----------
    context : StepContext
        The session to estimate for.

    Returns
    -------
    Estimate
        The updated ability estimate :math:`\hat\theta`, inside ``bounds``, its standard error and
        the number of evaluations of the log-likelihood or of its derivatives.
    """
    responses = context.responses
    if not responses:
      return Estimate(theta=float(context.theta), se=context.se, evaluations=0)

    items = context.item_bank.get_items(context.administered_items)
    if len(set(responses)) == 1:
      theta = _dodd(context.theta, context.item_bank, correct=responses[0], bounds=self._bounds)
      return Estimate(theta=theta, se=irt.see(theta, items), evaluations=0)

    return _session_estimate(ability.maximum_likelihood([responses], items, bounds=self._bounds, tol=self._tol))
