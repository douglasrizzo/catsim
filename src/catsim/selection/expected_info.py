"""Expected information selector implementations."""

from typing import Any

import numpy
import numpy.typing as npt

from .. import irt
from ..estimation.base import BaseEstimator
from ..estimation.numerical import NumericalSearchEstimator
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector


class MEISelector(BaseSelector):
  """Maximum Expected Information selector.

  For full algorithmic details, see :doc:`/specs/selection/mei-selector`.
  """

  def __init__(self, estimator: BaseEstimator | None = None, r_max: float = 1.0) -> None:
    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)

    super().__init__()
    self._estimator = estimator if estimator is not None else NumericalSearchEstimator()
    self._r_max = float(r_max)

  def __str__(self) -> str:
    """Return the selector name."""
    return f"Maximum Expected Information Selector ({self._estimator})"

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
    item = item_bank.items[candidate_idx]
    p_correct = irt.icc(est_theta, item[0], item[1], item[2], item[3])

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

    info_correct = item_bank.test_information(theta_correct, next_administered)
    info_incorrect = item_bank.test_information(theta_incorrect, next_administered)
    return float(p_correct * info_correct + (1.0 - p_correct) * info_incorrect)

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,  # noqa: ARG002
    exposure_rates: npt.NDArray[numpy.floating[Any]] | None = None,
    response_vector: list[bool] | None = None,
  ) -> int | None:
    """Select the item with the highest expected post-response information."""
    if response_vector is None:
      msg = "MEISelector requires the response_vector to compute one-step-ahead estimates."
      raise ValueError(msg)

    candidate_mask = numpy.ones(item_bank.n_items, dtype=bool)
    candidate_mask[administered_items] = False
    all_candidates = numpy.flatnonzero(candidate_mask)
    if all_candidates.size == 0:
      msg = "There are no more items to apply."
      raise NoItemsAvailableError(msg)

    if exposure_rates is None:
      exposure_rates = numpy.zeros(item_bank.n_items, dtype=float)
    low_exposure_mask = candidate_mask & (exposure_rates < self._r_max)

    candidates = numpy.flatnonzero(low_exposure_mask)
    if candidates.size == 0:
      candidates = all_candidates

    scores = numpy.array(
      [
        self._expected_information(item_bank, administered_items, response_vector, int(idx), est_theta)
        for idx in candidates
      ],
      dtype=float,
    )
    return int(candidates[int(scores.argmax())])
