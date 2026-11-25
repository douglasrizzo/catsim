"""Expected information selector implementations."""

import dataclasses

import numpy

from .. import irt
from ..compat import override
from ..context import StepContext
from ..estimation.base import BaseEstimator
from ..estimation.numerical import NumericalSearchEstimator
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining


class MEISelector(BaseSelector):
  """Maximum Expected Information selector.

  For each candidate item, the selector re-estimates ability as if the candidate were answered
  correctly and as if it were answered incorrectly. It scores the candidate by the information of
  the administered test plus the candidate at each updated estimate, weighted by the probability
  of each response at the current estimate, and selects the highest score.

  Parameters
  ----------
  estimator : BaseEstimator or None, optional
      Estimator used for the one-step-ahead ability updates. Default is a new
      :py:class:`catsim.estimation.NumericalSearchEstimator`. It may be the engine's own
      estimator: the look-ahead estimates are never recorded in the session.
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max are only selected when no
      other item remains. Default is 1 (no restriction).

  Raises
  ------
  ValueError
      If r_max is not in (0, 1].

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/mei-selector`.
  """

  def __init__(self, estimator: BaseEstimator | None = None, r_max: float = 1.0) -> None:
    super().__init__()
    self._estimator = estimator if estimator is not None else NumericalSearchEstimator()
    self._r_max = check_r_max(r_max)

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return f"{type(self).__name__}(estimator={self._estimator!r}, r_max={self._r_max!r})"

  @property
  def estimator(self) -> BaseEstimator:
    """Return the estimator used for one-step-ahead scoring."""
    return self._estimator

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate accepted by the selector."""
    return self._r_max

  def _expected_information(self, context: StepContext, candidate: int) -> float:
    """Return the candidate's information, expected over its two responses, at the one-step-ahead estimates."""
    administered_items = (*context.administered_items, candidate)
    correct = dataclasses.replace(context, administered_items=administered_items, responses=(*context.responses, True))
    incorrect = dataclasses.replace(correct, responses=(*context.responses, False))
    theta_correct = self._estimator.estimate(correct).theta
    theta_incorrect = self._estimator.estimate(incorrect).theta
    p_correct = irt.icc(context.theta, context.item_bank.items[candidate])
    info_correct = context.item_bank.total_information(theta_correct, administered_items)
    info_incorrect = context.item_bank.total_information(theta_incorrect, administered_items)
    return float(p_correct * info_correct + (1.0 - p_correct) * info_incorrect)

  @override
  def select(self, context: StepContext) -> int:
    """Return the available item with the largest expected information after its response.

    The one-step-ahead estimates start from ``context.theta``, which also gives the response
    probabilities.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.
    """
    candidates = below_cap(remaining(context), context.exposure_rates, self._r_max)
    scores = numpy.array([self._expected_information(context, int(idx)) for idx in candidates], dtype=float)
    return pick_best(candidates, scores, context.rng)
