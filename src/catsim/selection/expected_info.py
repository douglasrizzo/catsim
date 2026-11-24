"""Expected information selector implementations."""

import numpy

from .. import irt
from ..estimation.base import BaseEstimator
from ..estimation.numerical import NumericalSearchEstimator
from ..item_bank import ItemBank
from ..typing import FloatArray
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining, require_responses


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
      :py:class:`catsim.estimation.NumericalSearchEstimator`. Give the selector its own instance:
      it calls ``estimate`` twice per candidate item, which would inflate the call and evaluation
      counters of an estimator shared with the engine.
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max are only selected when no
      other item remains. Default is 1 (no restriction).

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/mei-selector`.
  """

  def __init__(self, estimator: BaseEstimator | None = None, r_max: float = 1.0) -> None:
    """Initialize a MEISelector object.

    Raises
    ------
    ValueError
        If r_max is not in (0, 1].
    """
    super().__init__()
    self._estimator = estimator if estimator is not None else NumericalSearchEstimator()
    self._r_max = check_r_max(r_max)

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

  def _expected_information(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    candidate_idx: int,
    est_theta: float,
  ) -> float:
    next_administered = [*administered_items, candidate_idx]
    theta_correct = self._estimator.estimate(
      item_bank=item_bank,
      administered_items=next_administered,
      response_vector=[*response_vector, True],
      est_theta=est_theta,
    )
    theta_incorrect = self._estimator.estimate(
      item_bank=item_bank,
      administered_items=next_administered,
      response_vector=[*response_vector, False],
      est_theta=est_theta,
    )
    item = item_bank.items[candidate_idx]
    p_correct = irt.icc(est_theta, item[0], item[1], item[2], item[3])
    info_correct = item_bank.test_information(theta_correct, next_administered)
    info_incorrect = item_bank.test_information(theta_incorrect, next_administered)
    return float(p_correct * info_correct + (1.0 - p_correct) * info_incorrect)

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,
    exposure_rates: FloatArray | None = None,
    response_vector: list[bool] | None = None,
  ) -> int | None:
    """Return the index of the item with the highest expected post-response information.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability, used for the response probabilities and as the starting
        point of the one-step-ahead estimates.
    rng : numpy.random.Generator or None, optional
        Random number generator, used only to break ties between candidates with equal scores
        (ADR-0008).
    exposure_rates : numpy.ndarray or None, optional
        Run-level exposure-rate snapshot. ``None`` treats every item as unexposed.
    response_vector : list[bool] or None, optional
        Responses to the administered items. Required.

    Returns
    -------
    int
        Index of the next item to be applied.

    Raises
    ------
    ValueError
        If ``response_vector`` is None, or if candidates tie for the best score and ``rng`` is None.
    NoItemsAvailableError
        If every item has already been administered.
    """
    response_vector = require_responses(response_vector)

    candidates = below_cap(remaining(item_bank.n_items, administered_items), exposure_rates, self._r_max)

    scores = numpy.array(
      [
        self._expected_information(item_bank, administered_items, response_vector, int(idx), est_theta)
        for idx in candidates
      ],
      dtype=float,
    )
    return pick_best(candidates, scores, rng)
