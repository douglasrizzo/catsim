"""Warm's Weighted Likelihood Estimator for CAT."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy
import numpy.typing as npt
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import expit

from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from .base import BaseEstimator

if TYPE_CHECKING:
  from ..item_bank import ItemBank

FloatArray = npt.NDArray[numpy.floating[Any]]

# Grid used to bracket the roots of the modified score before refining them with Brent's method.
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

  The modified score is scanned on a grid over the search interval to bracket its roots, which
  are then refined with Brent's method. If several roots exist, the one with the largest weighted
  likelihood is returned; if none does, the estimate is the search bound the modified score
  points to.

  For full algorithmic details, see :doc:`/api/techniques/estimation/warm-wle`.

  Parameters
  ----------
  tol : float, optional
      Absolute tolerance of the root refinement. Default is 1e-6.
  verbose : bool, optional
      Whether to be verbose during execution. Default is False.
  """

  def __init__(self, tol: float = 1e-6, verbose: bool = False) -> None:
    super().__init__(verbose=verbose)
    self._tol = float(tol)

  def __str__(self) -> str:
    """Return a string representation of the estimator."""
    return "Warm Weighted Likelihood Estimator"

  @staticmethod
  def _modified_score(theta: npt.ArrayLike, response_vector: list[bool], items: FloatArray) -> FloatArray:
    r"""Return Warm's modified score at one or more ability values.

    Parameters
    ----------
    theta : array_like
        Ability value(s) at which to evaluate the modified score.
    response_vector : list[bool]
        Responses to the items, in the same order as ``items``.
    items : numpy.ndarray
        Item parameter matrix with columns ``a, b, c, d``.

    Returns
    -------
    numpy.ndarray
        The likelihood score plus :math:`J(\theta) / [2 I(\theta)]`, one value per ``theta``.
    """
    thetas = numpy.atleast_1d(numpy.asarray(theta, dtype=float))[:, None]
    a, b, c, d = items[:, 0], items[:, 1], items[:, 2], items[:, 3]
    logistic = expit(a * (thetas - b))
    p = c + (d - c) * logistic
    dp = a * (d - c) * logistic * (1.0 - logistic)
    d2p = a * dp * (1.0 - 2.0 * logistic)
    pq = numpy.clip(p * (1.0 - p), numpy.finfo(float).eps, None)

    responses = numpy.asarray(response_vector, dtype=float)
    score = numpy.sum((responses - p) * dp / pq, axis=1)
    info = numpy.sum(dp**2 / pq, axis=1)
    j = numpy.sum(dp * d2p / pq, axis=1)
    correction = numpy.divide(j, 2.0 * info, out=numpy.zeros_like(j), where=info > 0.0)
    return score + correction

  def estimate(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,
  ) -> float:
    """Return the root of Warm's modified score equation.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        Indices of the administered items.
    response_vector : list[bool]
        The examinee's responses to the administered items.
    est_theta : float
        The current ability estimate, returned unchanged when no item has been administered.

    Returns
    -------
    float
        The weighted likelihood estimate, within ``[THETA_MIN_EXTENDED, THETA_MAX_EXTENDED]``.
    """
    self._require_inputs(item_bank, administered_items, response_vector)

    self._calls += 1
    self._last_evaluations = 0

    if not administered_items:
      return float(est_theta)

    items = item_bank.items[administered_items, :4]

    def score(theta: float) -> float:
      return float(self._modified_score(theta, response_vector, items)[0])

    grid = numpy.linspace(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED, _SCAN_POINTS)
    values = self._modified_score(grid, response_vector, items)
    evaluations = grid.size

    # Maxima of the weighted likelihood: + to - sign changes of its derivative (a zero at the
    # right end of a bracket is a valid Brent endpoint), plus any bound the modified score
    # points outward from.
    candidates: list[float] = []
    for idx in numpy.flatnonzero((values[:-1] > 0.0) & (values[1:] <= 0.0)):
      root, result = brentq(score, grid[idx], grid[idx + 1], xtol=self._tol, full_output=True)
      candidates.append(float(root))
      evaluations += result.function_calls
    if values[0] <= 0.0:
      candidates.append(THETA_MIN_EXTENDED)
    if values[-1] > 0.0:
      candidates.append(THETA_MAX_EXTENDED)

    if len(candidates) == 1:
      theta_hat = candidates[0]
    else:
      # The modified score is the derivative of the weighted log-likelihood, so integrating it
      # from the lower bound ranks the candidates by weighted likelihood.
      heights = [quad(score, THETA_MIN_EXTENDED, candidate, limit=200)[0] for candidate in candidates]
      theta_hat = candidates[int(numpy.argmax(heights))]

    self._last_evaluations = evaluations
    self._total_evaluations += evaluations
    return float(theta_hat)
