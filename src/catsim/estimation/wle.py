"""Warm's Weighted Likelihood Estimator for CAT."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy
import numpy.typing as npt
from scipy.integrate import IntegrationWarning, quad
from scipy.optimize import brentq
from scipy.special import expit

from .. import irt
from ..context import Estimate
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from .base import BaseEstimator

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence

  from ..context import StepContext
  from ..typing import FloatArray

# Points on which the modified score is scanned to bracket its roots before Brent's method refines
# them (0.1 apart on [-6, 6]).
_SCAN_POINTS = 121


class WarmLikelihoodEstimator(BaseEstimator):
  r"""Warm's (1989) weighted likelihood estimator (WLE).

  The estimate is the root of Warm's modified score equation

  .. math::

    \sum_i \frac{(u_i - P_i) P_i'}{P_i Q_i} + \frac{J(\theta)}{2 I(\theta)} = 0,
    \qquad J(\theta) = \sum_i \frac{P_i' P_i''}{P_i Q_i},

  which removes the first-order (:math:`O(n^{-1})`) bias of the MLE and stays finite for
  all-correct and all-incorrect response patterns. For models without guessing or slipping
  (:math:`c = 0`, :math:`d = 1`), :math:`J = I'` and the estimate maximizes
  :math:`L(\theta)\sqrt{I(\theta)}`.

  The modified score is scanned at 121 equally spaced points over ``bounds`` (0.1 apart on the
  default bounds) to bracket its roots, which are then refined with Brent's method. If several
  roots exist, the one with the largest weighted likelihood is returned; if none does, the
  estimate is the bound the modified score points to.

  For full algorithmic details, see :doc:`/api/techniques/estimation/warm-wle`.

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

  def __repr__(self) -> str:
    """Return the estimator as a constructor call that names every argument."""
    return f"{type(self).__name__}(tol={self._tol!r}, bounds={self._bounds!r})"

  @staticmethod
  def _modified_score(theta: npt.ArrayLike, responses: Sequence[bool], items: FloatArray) -> FloatArray:
    r"""Return Warm's modified score at one or more ability values.

    Parameters
    ----------
    theta : array_like
        Ability value(s) at which to evaluate the modified score.
    responses : Sequence[bool]
        Responses to the items, in the same order as ``items``.
    items : numpy.ndarray
        Item parameter matrix with columns ``a, b, c, d``.

    Returns
    -------
    numpy.ndarray
        The likelihood score plus :math:`J(\theta) / [2 I(\theta)]`, one value per ``theta``.
    """
    thetas = numpy.atleast_1d(numpy.asarray(theta, dtype=float))[:, None]
    p, q, dp, d2p = _response_curves(thetas, items)
    pq = numpy.maximum(p * q, numpy.finfo(float).tiny)

    # u - P is Q for a correct response and -P for an incorrect one
    residual = numpy.where(numpy.asarray(responses, dtype=bool), q, -p)
    score = numpy.sum(residual * dp / pq, axis=1)
    info = numpy.sum(dp**2 / pq, axis=1)
    j = numpy.sum(dp * d2p / pq, axis=1)
    correction = numpy.divide(j, 2.0 * info, out=numpy.zeros_like(j), where=info > 0.0)
    return score + correction

  def estimate(self, context: StepContext) -> Estimate:
    r"""Return the root of Warm's modified score equation, or ``context.theta`` when no item has been administered.

    The standard error is :math:`1/\sqrt{I(\hat\theta)}`, the inverse square root of the test
    information at the estimate, or ``math.inf`` where the information is 0. This is catR's
    default for the WLE (``semTheta`` with ``sem.type = "classic"``). :cite:t:`Magis12` (Eq. 14,
    p. 6) add a term to the information, :math:`(I'J - IJ')/(2I^2)`, which catR has not used since
    version 3.0 and which, with guessing or slipping parameters, can make the sum negative; catsim
    follows catR (ADR-0004). The evaluation count is
    the number of modified-score evaluations: the scan, Brent's refinement and, when several roots
    compete, the integration that ranks them. With no administered item, the estimate is
    ``context.theta`` and ``context.se``, unchanged, after no evaluation.

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

    responses = context.responses
    items = context.item_bank.get_items(context.administered_items)

    def score(theta: float) -> float:
      return float(self._modified_score(theta, responses, items)[0])

    low, high = self._bounds
    grid = numpy.linspace(low, high, _SCAN_POINTS)
    values = self._modified_score(grid, responses, items)
    evaluations = grid.size

    # Maxima of the weighted likelihood: + to - sign changes of its derivative (a zero at the
    # right end of a bracket is a valid Brent endpoint), plus any bound the modified score
    # points outward from.
    candidates: list[float] = []
    for idx in numpy.flatnonzero((values[:-1] > 0.0) & (values[1:] <= 0.0)).tolist():
      root, result = brentq(score, grid[idx], grid[idx + 1], xtol=self._tol, full_output=True)
      candidates.append(float(root))
      evaluations += result.function_calls
    if values[0] <= 0.0:
      candidates.append(low)
    if values[-1] > 0.0:
      candidates.append(high)

    if len(candidates) == 1:
      theta_hat = candidates[0]
    else:
      theta_hat, ranking_evaluations = _most_likely(score, low, candidates)
      evaluations += ranking_evaluations

    theta_hat = float(theta_hat)
    return Estimate(theta=theta_hat, se=irt.see(theta_hat, items), evaluations=int(evaluations))


def _response_curves(thetas: FloatArray, items: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, FloatArray]:
  """Return P, Q and the first and second derivatives of P, for every ability against every item."""
  a, b, c, d = items[:, 0], items[:, 1], items[:, 2], items[:, 3]
  z = a * (thetas - b)
  # The logistic and its complement both come from the logit, so Q and P Q stay exact where P
  # rounds to 1 instead of losing their digits to 1 - P
  logistic, complement = expit(z), expit(-z)
  p = c + (d - c) * logistic
  q = (1.0 - d) + (d - c) * complement
  dp = a * (d - c) * logistic * complement
  return p, q, dp, a * dp * (complement - logistic)


def _most_likely(score: Callable[[float], float], low: float, candidates: list[float]) -> tuple[float, int]:
  """Return the candidate of highest weighted likelihood and the score evaluations spent ranking the candidates."""
  # The modified score is the derivative of the weighted log-likelihood, so integrating it
  # from the lower bound ranks the candidates by weighted likelihood.
  heights = []
  evaluations = 0
  for candidate in candidates:
    # With full_output, quad returns its convergence warning instead of raising it; stacklevel 3
    # points the warning at the caller of WarmLikelihoodEstimator.estimate.
    height, _, info, *message = quad(score, low, candidate, limit=200, full_output=1)
    heights.append(height)
    evaluations += info["neval"]
    if message:
      warnings.warn(message[0], IntegrationWarning, stacklevel=3)
  return candidates[int(numpy.argmax(heights))], evaluations
