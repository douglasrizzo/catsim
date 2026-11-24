"""Tests for MEISelector."""

import numpy as np
import pytest
from typing_extensions import override

from catsim import Estimate, StepContext
from catsim.estimation import BaseEstimator, NumericalSearchEstimator
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector, MEISelector
from tests.contexts import make_context


class RecordingEstimator(BaseEstimator):
  """Estimator stub that records the history it receives."""

  def __init__(self) -> None:
    self.records: list[tuple[tuple[int, ...], tuple[bool, ...], float]] = []

  @override
  def estimate(self, context: StepContext) -> Estimate:
    self.records.append((context.administered_items, context.responses, context.theta))
    return Estimate(theta=float(len(context.responses)), se=1.0)

  def __repr__(self) -> str:
    return "RecordingEstimator()"


class ShiftEstimator(BaseEstimator):
  """Estimator stub that moves the estimate 0.5 toward the last response."""

  @override
  def estimate(self, context: StepContext) -> Estimate:
    return Estimate(theta=context.theta + (0.5 if context.responses[-1] else -0.5), se=1.0)


class TestMEISelector:
  """Tests for MEISelector."""

  def test_select_forwards_response_history_to_estimator(self) -> None:
    """Each look-ahead estimate sees the session's items and responses, then the candidate and one response.

    A look-ahead that dropped the history, appended one hypothetical response only, or started from another estimate
    would score the candidates on the wrong pattern.
    """
    item_bank = ItemBank.generate(10, rng=42)
    estimator = RecordingEstimator()

    MEISelector(estimator=estimator).select(make_context(item_bank, [0], [True], theta=0.25))

    expected = [((0, candidate), (True, response), 0.25) for candidate in range(1, 10) for response in (True, False)]
    assert sorted(estimator.records) == sorted(expected)

  def test_select_falls_back_when_exposure_cap_blocks_all_candidates(self) -> None:
    """With every remaining item over r_max, the cap is ignored and the best remaining item is still chosen."""
    item_bank = ItemBank.generate(10, rng=42)
    selector = MEISelector(estimator=RecordingEstimator(), r_max=0.5)

    selected = selector.select(
      make_context(
        item_bank,
        [1],
        [True],
        theta=0.0,
        rng=np.random.default_rng(42),
        exposure_rates=np.ones(item_bank.n_items, dtype=float),
      )
    )

    # RecordingEstimator puts both one-step-ahead estimates at 2.0 (the number of responses), so each score is
    # the information of item 1 plus the candidate's at 2.0, where item 9 (0.1417) just beats item 5 (0.1410)
    assert selected == 9

  def test_select_diverges_from_max_info_when_one_step_estimates_shift(self) -> None:
    """MEI should not collapse to MaxInfo when the response history changes theta.

    This is the key behavioral check for a one-step-ahead selector: if MEI
    always agreed with MaxInfo here, it would be ignoring the estimated post-response shift.
    """
    # Frozen bank parameters captured from a verified divergence scenario.
    item_bank = ItemBank(
      np.array(
        [
          [1.16697378, 0.12573022, 0.23911482, 0.97692311],
          [1.22622503, 0.64042265, 0.243674, 0.96302065],
          [1.29039876, -0.53566937, 0.25823261, 0.9998326],
          [1.43677024, 1.30400005, 0.27085027, 0.99885012],
          [0.88364463, -0.70373524, 0.24742931, 0.98113252],
          [1.21033149, -0.62327446, 0.27732927, 0.97902756],
          [1.14530208, -2.32503077, 0.23669611, 0.9813068],
          [1.01693316, -1.24591095, 0.2570302, 0.96333529],
        ],
        dtype=float,
      )
    )
    administered_items = [0]
    responses = [True]
    theta = 0.0

    mei = MEISelector(estimator=NumericalSearchEstimator())
    max_info = MaxInfoSelector()

    mei_choice = mei.select(
      make_context(item_bank, administered_items, responses, theta=theta, rng=np.random.default_rng(0))
    )
    max_info_choice = max_info.select(
      make_context(item_bank, administered_items, theta=theta, rng=np.random.default_rng(0))
    )

    assert mei_choice == 1
    assert max_info_choice == 2
    assert mei_choice != max_info_choice

  def test_expected_information_matches_the_closed_form_on_1pl_items(self) -> None:
    """Under the 1PL model an item's information is P(1 - P), so each MEI score has a closed form."""
    difficulties = np.array([0.0, 0.8, -0.6, 1.5])
    item_bank = ItemBank(np.column_stack([np.ones(4), difficulties, np.zeros(4), np.ones(4)]))
    selector = MEISelector(estimator=ShiftEstimator())
    theta = 0.2

    def probability(theta: float, b: float) -> float:
      return 1.0 / (1.0 + np.exp(-(theta - b)))

    def total_information(theta: float, indices: list[int]) -> float:
      return sum(probability(theta, difficulties[j]) * (1.0 - probability(theta, difficulties[j])) for j in indices)

    context = make_context(item_bank, [0], [True], theta=theta)
    expected = {}
    for candidate in (1, 2, 3):
      p_correct = probability(theta, difficulties[candidate])
      expected[candidate] = p_correct * total_information(theta + 0.5, [0, candidate]) + (
        1.0 - p_correct
      ) * total_information(theta - 0.5, [0, candidate])
      score = selector._expected_information(context, candidate)
      assert score == pytest.approx(expected[candidate], rel=1e-12)

    selected = selector.select(context)
    assert selected == max(expected, key=expected.__getitem__)
