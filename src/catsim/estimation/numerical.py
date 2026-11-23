"""Numerical estimation methods for ability estimation."""

from scipy.optimize import minimize_scalar

from .. import irt
from ..irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from ..item_bank import ItemBank
from .base import BaseEstimator


def _dodd(theta: float, item_bank: ItemBank, correct: bool, bounds: tuple[float, float]) -> float:
  r"""Return Dodd's estimate after a response pattern that is all correct or all incorrect.

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
    estimate = max(theta, theta + (float(difficulty.max()) - theta) / 2)
  else:
    estimate = min(theta, theta - (theta - float(difficulty.min())) / 2)
  low, high = bounds
  return float(min(max(estimate, low), high))


class NumericalSearchEstimator(BaseEstimator):
  """Estimate ability by numerically maximizing the IRT log-likelihood.

  The maximum is searched for with SciPy's bounded scalar minimizer on ``bounds``. While every
  response is correct, or every response is incorrect, the likelihood has no maximum, and the
  estimate takes Dodd's step :cite:p:`Dod90` toward the bank's hardest or easiest item instead.

  Parameters
  ----------
  tol : float, optional
      Absolute tolerance of the bounded search, in the ability scale. Default is 1e-6.
  bounds : tuple[float, float], optional
      Range of the estimate. Default is ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/estimation/numerical-search-estimator`.
  """

  def __str__(self) -> str:
    """Return a string representation of the estimator."""
    return "Numerical Search Estimator"

  def __init__(
    self,
    tol: float = 1e-6,
    bounds: tuple[float, float] = (THETA_MIN_EXTENDED, THETA_MAX_EXTENDED),
  ) -> None:
    """Initialize the estimator.

    Parameters
    ----------
    tol : float, optional
        Absolute tolerance of the bounded search, in the ability scale. Default is 1e-6.
    bounds : tuple[float, float], optional
        Range of the estimate. Default is ``(THETA_MIN_EXTENDED, THETA_MAX_EXTENDED)``.

    Raises
    ------
    ValueError
        If ``tol`` is not positive, or if ``bounds`` is not a finite interval with low < high.
    """
    super().__init__()
    self._tol = self._check_tol(tol)
    self._bounds = self._check_bounds(bounds)

  def estimate(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,
  ) -> float:
    r"""Return the maximum likelihood estimate, or Dodd's step while every response is the same.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    response_vector : list[bool]
        A boolean list containing the examinee's answers to the administered items.
    est_theta : float
        The current estimated ability: the start of Dodd's step, and the estimate returned
        unchanged when no item has been administered.

    Returns
    -------
    float
        The updated ability estimate :math:`\hat\theta`, inside ``bounds``.

    Raises
    ------
    ValueError
        If required parameters are missing, if ``response_vector`` and ``administered_items`` have
        different lengths, or if ``response_vector`` contains non-Boolean elements.
    """
    self._require_inputs(item_bank, administered_items, response_vector)

    self._calls += 1
    self._last_evaluations = 0

    if len(administered_items) == 0:
      return float(est_theta)

    if len(set(response_vector)) == 1:
      return _dodd(est_theta, item_bank, response_vector[0], self._bounds)

    res = minimize_scalar(
      irt.negative_log_likelihood,
      bounds=self._bounds,
      method="bounded",
      args=(response_vector, item_bank.get_items(administered_items)),
      # SciPy's bounded method takes its absolute tolerance only through options
      options={"xatol": self._tol},
    )
    self._last_evaluations = res.nfev
    self._total_evaluations += self._last_evaluations

    return res.x
