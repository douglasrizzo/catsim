"""Tests for catsim.selection module."""

import importlib
import inspect
from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad

from catsim import irt
from catsim.estimation import BaseEstimator
from catsim.estimation.numerical import NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  BaseSelector,
  ClusterSelector,
  IntervalInfoSelector,
  KLSelector,
  LinearSelector,
  MaxInfoBBlockSelector,
  MaxInfoSelector,
  MaxInfoStratSelector,
  MEISelector,
  MLWISelector,
  RandomesqueSelector,
  RandomSelector,
  The54321Selector,
  UrrySelector,
)


class RecordingEstimator(BaseEstimator):
  """Estimator stub that records the history it receives."""

  def __init__(self) -> None:
    super().__init__()
    self.records: list[tuple[list[int], list[bool], float]] = []

  def estimate(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,
  ) -> float:
    del item_bank
    self.records.append((list(administered_items), list(response_vector), est_theta))
    return float(len(response_vector))


class ShiftEstimator(BaseEstimator):
  """Estimator stub that moves the estimate 0.5 toward the last response."""

  def estimate(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,
  ) -> float:
    del item_bank, administered_items
    return est_theta + (0.5 if response_vector[-1] else -0.5)


def _two_pl_bank(discriminations: list[float], difficulties: list[float]) -> ItemBank:
  """Build a 2PL bank, whose item information at theta is a^2 P (1 - P)."""
  n_items = len(discriminations)
  return ItemBank(np.column_stack([discriminations, difficulties, np.zeros(n_items), np.ones(n_items)]))


class TestMaxInfoSelector:
  """Tests for MaxInfoSelector."""

  # all difficulties are 0, so at theta = 0 information is a^2 / 4 and ranks items 1, 2, 0, 3
  BANK_A = (1.0, 2.0, 1.5, 0.5)

  def test_init_invalid_r_max_raises(self) -> None:
    """Test that invalid r_max raises ValueError."""
    with pytest.raises(ValueError, match="between 0 and 1"):
      MaxInfoSelector(r_max=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
      MaxInfoSelector(r_max=-0.1)

  def test_select_returns_the_most_informative_remaining_item(self) -> None:
    """The pick is the remaining item with the largest information at the estimate."""
    item_bank = _two_pl_bank(list(self.BANK_A), [0.0] * 4)
    selector = MaxInfoSelector()

    assert selector.select(item_bank=item_bank, administered_items=[], est_theta=0.0) == 1
    assert selector.select(item_bank=item_bank, administered_items=[1], est_theta=0.0) == 2

  def test_select_skips_items_at_the_exposure_cap(self) -> None:
    """An item whose exposure rate has reached r_max is passed over for the next most informative one."""
    item_bank = _two_pl_bank(list(self.BANK_A), [0.0] * 4)
    exposure_rates = np.array([0.0, 0.5, 0.0, 0.0])

    selected = MaxInfoSelector(r_max=0.5).select(
      item_bank=item_bank, administered_items=[], est_theta=0.0, exposure_rates=exposure_rates
    )

    assert selected == 2

  def test_select_ignores_the_cap_when_every_item_is_over_it(self) -> None:
    """With every remaining item at or over r_max, the most informative one is still administered."""
    item_bank = _two_pl_bank(list(self.BANK_A), [0.0] * 4)

    selected = MaxInfoSelector(r_max=0.5).select(
      item_bank=item_bank, administered_items=[], est_theta=0.0, exposure_rates=np.ones(4)
    )

    assert selected == 1

  def test_select_raises_when_exhausted(self) -> None:
    """Test that select raises NoItemsAvailableError when all items administered."""
    item_bank = ItemBank.generate_item_bank(5, seed=42)
    selector = MaxInfoSelector()

    administered = list(range(5))  # All items
    with pytest.raises(NoItemsAvailableError, match="no more items"):
      selector.select(
        item_bank=item_bank,
        administered_items=administered,
        est_theta=0.0,
        rng=np.random.default_rng(42),
      )


class TestRandomSelector:
  """Tests for RandomSelector."""

  def test_select_is_random(self) -> None:
    """Test that selection is random across calls with different RNGs."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    selector = RandomSelector()

    selections = set()
    for seed in range(20):
      selected = selector.select(
        item_bank=item_bank,
        administered_items=[],
        est_theta=0.0,
        rng=np.random.default_rng(seed),
      )
      if selected is not None:
        selections.add(selected)

    # With 20 different seeds, we should get multiple different selections
    assert len(selections) > 1


class TestLinearSelector:
  """Tests for LinearSelector."""

  def test_select_returns_items_in_order(self) -> None:
    """Test that select returns items in the specified order."""
    indices = [5, 10, 15]
    selector = LinearSelector(indices)
    item_bank = ItemBank.generate_item_bank(50, seed=42)

    # First selection
    selected1 = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )
    assert selected1 == 5

    # Second selection
    selected2 = selector.select(
      item_bank=item_bank,
      administered_items=[5],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )
    assert selected2 == 10

    # Third selection
    selected3 = selector.select(
      item_bank=item_bank,
      administered_items=[5, 10],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )
    assert selected3 == 15


class TestUrrySelector:
  """Tests for UrrySelector."""

  def test_select_matches_difficulty_to_the_estimate_whatever_the_discrimination(self) -> None:
    """The pick is the remaining item whose difficulty is closest to the estimate, not the most informative one."""
    # at theta = 0.8, item 0 is the most informative but item 1's difficulty is the closest
    item_bank = _two_pl_bank([2.5, 0.5, 1.0, 1.0], [0.3, 0.9, -1.0, 2.5])
    selector = UrrySelector()

    assert MaxInfoSelector().select(item_bank=item_bank, administered_items=[], est_theta=0.8) == 0
    assert selector.select(item_bank=item_bank, administered_items=[], est_theta=0.8) == 1
    assert selector.select(item_bank=item_bank, administered_items=[1], est_theta=0.8) == 0


class TestRandomesqueSelector:
  """Tests for RandomesqueSelector."""

  def test_draws_from_the_bin_of_most_informative_remaining_items(self) -> None:
    """Every pick is one of the bin_size most informative remaining items, and each of them gets picked."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    administered = [3, 17, 25]
    by_information = [int(i) for i in np.argsort(-item_bank.information(-0.4)) if i not in administered]
    selector = RandomesqueSelector(4)
    rng = np.random.default_rng(0)

    picks = {
      selector.select(item_bank=item_bank, administered_items=administered, est_theta=-0.4, rng=rng) for _ in range(200)
    }

    assert picks == set(by_information[:4])


class TestThe54321Selector:
  """The 5-4-3-2-1 rule of McBride & Martin (1983)."""

  def test_init_default_and_validation(self) -> None:
    """The first item is drawn from the 5 most informative items unless configured otherwise."""
    assert The54321Selector().initial_bin_size == 5
    assert The54321Selector(3).initial_bin_size == 3
    with pytest.raises(ValueError, match="initial_bin_size"):
      The54321Selector(0)

  @pytest.mark.parametrize(("n_administered", "bin_size"), [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (12, 1)])
  def test_draws_from_a_shrinking_bin_of_most_informative_items(self, n_administered: int, bin_size: int) -> None:
    """Item k is drawn from the 5 - k most informative remaining items, and from the fifth on it is the best one.

    Earlier versions sized the bin as test_size - k, so a 20-item test drew its first item from the top
    20, and past test_size the negative bin drew from almost the whole bank.
    """
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    administered = list(range(n_administered))
    by_information = [int(i) for i in np.argsort(-item_bank.information(0.3)) if i not in administered]
    selector = The54321Selector()
    rng = np.random.default_rng(0)

    picks = {
      selector.select(item_bank=item_bank, administered_items=administered, est_theta=0.3, rng=rng) for _ in range(200)
    }

    assert picks == set(by_information[:bin_size])


STRATIFIED_SELECTORS = [AStratSelector, AStratBBlockSelector, MaxInfoStratSelector, MaxInfoBBlockSelector]


def _stratified_keys(item_bank: ItemBank, selector_class: type) -> tuple[np.ndarray, np.ndarray]:
  """Return the (stratification key, location) pair a stratified selector uses: (a, b) or (I_max, theta_max)."""
  if issubclass(selector_class, MaxInfoStratSelector):
    return item_bank.max_info_values, item_bank.max_info_thetas
  return item_bank.discrimination, item_bank.difficulty


class TestStratifiedSelectors:
  """Stratified selection as in Chang & Ying (1999), Chang et al. (2001) and Barrada et al. (2006)."""

  @pytest.mark.parametrize("selector_class", STRATIFIED_SELECTORS)
  def test_selection_adapts_to_the_ability_estimate(self, selector_class: type) -> None:
    """Each pick is the item of the current stratum whose location is closest to the estimate.

    Earlier versions of AStratSelector and AStratBBlockSelector ignored the estimate and gave every
    examinee the same test, and MaxInfoStratSelector maximized information instead of matching theta_max.
    """
    item_bank = ItemBank.generate_item_bank(200, itemtype="3PL", seed=1)
    _, location = _stratified_keys(item_bank, selector_class)
    tests = {}
    for theta in (-2.0, 2.0):
      selector = selector_class(10)
      administered: list[int] = []
      for position in range(10):
        item = selector.select(item_bank=item_bank, administered_items=administered, est_theta=theta)
        remaining = [i for i in selector.strata(item_bank)[position] if i not in administered]
        assert item in remaining
        assert abs(location[item] - theta) == pytest.approx(np.abs(location[remaining] - theta).min())
        administered.append(item)
      tests[theta] = administered

    assert set(tests[-2.0]).isdisjoint(tests[2.0])

  @pytest.mark.parametrize("selector_class", [AStratSelector, MaxInfoStratSelector])
  def test_strata_ascend_in_the_stratification_key(self, selector_class: type) -> None:
    """Without blocking, the bank is split into contiguous strata of increasing a (or maximum information)."""
    item_bank = ItemBank.generate_item_bank(103, itemtype="3PL", seed=2)
    key, _ = _stratified_keys(item_bank, selector_class)
    strata = selector_class(30, n_strata=5).strata(item_bank)

    assert sorted(np.concatenate(strata).tolist()) == list(range(item_bank.n_items))
    assert [len(stratum) for stratum in strata] == [21, 21, 21, 20, 20]
    for lower, upper in pairwise(strata):
      assert key[lower].max() <= key[upper].min()

  @pytest.mark.parametrize("selector_class", [AStratBBlockSelector, MaxInfoBBlockSelector])
  def test_blocking_spreads_each_location_block_across_the_strata(self, selector_class: type) -> None:
    """In each block of n_strata items ordered by location, the k-th item by key goes to stratum k.

    Earlier versions made each location block a stratum, so every test moved from the easiest block to
    the hardest regardless of the estimate.
    """
    item_bank = ItemBank.generate_item_bank(102, itemtype="3PL", corr=0.5, seed=3)
    key, location = _stratified_keys(item_bank, selector_class)
    n_strata = 4
    strata = selector_class(20, n_strata=n_strata).strata(item_bank)
    stratum_of = np.empty(item_bank.n_items, dtype=int)
    for index, stratum in enumerate(strata):
      stratum_of[stratum] = index

    by_location = np.argsort(location, kind="stable")
    for start in range(0, item_bank.n_items, n_strata):
      block = by_location[start : start + n_strata]
      by_key = block[np.argsort(key[block], kind="stable")]
      # the trailing block of 2 items is spread over strata 0 and 2
      expected = [rank * n_strata // len(block) for rank in range(len(block))]
      assert stratum_of[by_key].tolist() == expected

  def test_stages_span_several_items_with_fewer_strata(self) -> None:
    """With 5 strata in a 30-item test (Barrada et al., 2006), each stage administers 6 items from its stratum."""
    item_bank = ItemBank.generate_item_bank(250, itemtype="3PL", seed=4)
    selector = AStratSelector(30, n_strata=5)
    strata = selector.strata(item_bank)
    administered: list[int] = []
    for position in range(30):
      item = selector.select(item_bank=item_bank, administered_items=administered, est_theta=0.5)
      assert item in strata[position // 6]
      administered.append(item)

  def test_stays_in_the_last_stratum_past_test_size(self) -> None:
    """Sessions longer than test_size keep drawing from the last stratum instead of raising RuntimeError."""
    item_bank = ItemBank.generate_item_bank(60, seed=5)
    selector = MaxInfoStratSelector(5)
    administered: list[int] = []
    for _ in range(8):
      administered.append(selector.select(item_bank=item_bank, administered_items=administered, est_theta=0.0))

    assert set(administered[4:]) <= set(selector.strata(item_bank)[-1].tolist())

  def test_raises_when_the_current_stratum_is_exhausted(self) -> None:
    """A stage whose stratum has no items left raises NoItemsAvailableError."""
    item_bank = ItemBank.generate_item_bank(6, seed=6)
    selector = AStratSelector(10, n_strata=2)
    first_stratum = selector.strata(item_bank)[0].tolist()

    with pytest.raises(NoItemsAvailableError, match="stratum 0"):
      selector.select(item_bank=item_bank, administered_items=first_stratum, est_theta=0.0)

  def test_n_strata_must_be_between_one_and_test_size(self) -> None:
    """n_strata defaults to test_size and cannot exceed it."""
    assert AStratSelector(10).n_strata == 10
    with pytest.raises(ValueError, match="n_strata"):
      AStratSelector(10, n_strata=0)
    with pytest.raises(ValueError, match="n_strata"):
      AStratSelector(10, n_strata=11)


class TestIntervalInfoSelector:
  """Tests for IntervalInfoSelector."""

  @pytest.mark.parametrize(("kwargs", "match"), [({"interval": 0.0}, "interval"), ({"confidence": 1.0}, "confidence")])
  def test_init_rejects_invalid_arguments(self, kwargs: dict[str, float], match: str) -> None:
    """The half-width must be positive and the confidence level strictly between 0 and 1."""
    with pytest.raises(ValueError, match=match):
      IntervalInfoSelector(**kwargs)

  def test_default_interval_is_the_confidence_interval_of_the_estimate(self) -> None:
    """Veerkamp & Berger (1997) integrate over theta_hat +/- z / sqrt(I(theta_hat)) of the administered items."""
    item_bank = ItemBank.generate_item_bank(60, itemtype="3PL", seed=7)
    administered = [3, 11, 25, 40, 52]
    theta = 0.7
    half_width = 1.959963984540054 / np.sqrt(irt.test_info(theta, item_bank.items[administered]))
    remaining = [i for i in range(item_bank.n_items) if i not in administered]
    integrals = [
      quad(irt.inf, theta - half_width, theta + half_width, args=tuple(item_bank.items[i, :4]))[0] for i in remaining
    ]

    selected = IntervalInfoSelector().select(item_bank=item_bank, administered_items=administered, est_theta=theta)

    assert selected == remaining[int(np.argmax(integrals))]

  def test_default_selection_adapts_to_the_estimate(self) -> None:
    """The default no longer integrates over the whole ability scale, which ignored the estimate.

    Those integrals were NaN, so earlier versions administered items in bank order. Before any item is
    administered the confidence interval is still unbounded, and the 2PL integral over it is a. With
    equal discriminations, the item whose difficulty is nearest the estimate has the largest integral
    over any interval centered on it.
    """
    items = np.column_stack([np.linspace(0.6, 2.0, 41), np.zeros(41), np.zeros(41), np.ones(41)])
    assert IntervalInfoSelector().select(item_bank=ItemBank(items), administered_items=[], est_theta=0.0) == 40

    items[:, 0], items[:, 1] = 1.2, np.linspace(-3.0, 3.0, 41)
    item_bank = ItemBank(items)
    administered = [18, 19, 20, 21, 22]
    low = IntervalInfoSelector().select(item_bank=item_bank, administered_items=administered, est_theta=-2.0)
    high = IntervalInfoSelector().select(item_bank=item_bank, administered_items=administered, est_theta=2.0)

    assert item_bank.difficulty[low] == pytest.approx(-1.95)
    assert item_bank.difficulty[high] == pytest.approx(1.95)

  def test_fixed_interval_matches_numerical_integration(self) -> None:
    """A fixed half-width integrates information over theta_hat +/- interval."""
    item_bank = ItemBank.generate_item_bank(40, seed=8)
    integrals = [quad(irt.inf, -1.3, -0.3, args=tuple(item[:4]))[0] for item in item_bank.items]

    selected = IntervalInfoSelector(interval=0.5).select(item_bank=item_bank, administered_items=[], est_theta=-0.8)

    assert selected == int(np.argmax(integrals))

  def test_select_excludes_administered_and_raises_when_exhausted(self) -> None:
    """Administered items are never returned; an exhausted bank raises NoItemsAvailableError."""
    item_bank = ItemBank.generate_item_bank(3, seed=9)
    selector = IntervalInfoSelector(interval=1.0)

    assert selector.select(item_bank=item_bank, administered_items=[0, 2], est_theta=0.0) == 1
    with pytest.raises(NoItemsAvailableError):
      selector.select(item_bank=item_bank, administered_items=[0, 1, 2], est_theta=0.0)


class TestMEISelector:
  """Tests for MEISelector."""

  def test_select_requires_response_vector(self) -> None:
    """The selector needs the current response history."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    selector = MEISelector()

    with pytest.raises(ValueError, match="response_vector"):
      selector.select(
        item_bank=item_bank,
        administered_items=[],
        est_theta=0.0,
        rng=np.random.default_rng(42),
      )

  def test_select_forwards_response_history_to_estimator(self) -> None:
    """One-step-ahead scoring should append both hypothetical responses."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = RecordingEstimator()
    selector = MEISelector(estimator=estimator)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[0],
      est_theta=0.0,
      rng=np.random.default_rng(42),
      response_vector=[True],
    )

    assert selected is not None
    assert selected != 0
    assert estimator.records
    assert all(call[0][:1] == [0] for call in estimator.records)
    assert {tuple(call[1]) for call in estimator.records} == {(True, True), (True, False)}

  def test_select_falls_back_when_exposure_cap_blocks_all_candidates(self) -> None:
    """MEI should still choose from non-administered items when every candidate exceeds r_max."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    selector = MEISelector(estimator=RecordingEstimator(), r_max=0.0)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[0],
      est_theta=0.0,
      rng=np.random.default_rng(42),
      exposure_rates=np.ones(item_bank.n_items, dtype=float),
      response_vector=[True],
    )

    assert selected is not None
    assert selected != 0

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
    response_vector = [True]
    est_theta = 0.0

    mei = MEISelector(estimator=NumericalSearchEstimator())
    max_info = MaxInfoSelector()

    mei_choice = mei.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      rng=np.random.default_rng(0),
      response_vector=response_vector,
    )
    max_info_choice = max_info.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      rng=np.random.default_rng(0),
    )

    assert mei_choice == 1
    assert max_info_choice == 2
    assert mei_choice != max_info_choice

  def test_select_matches_manual_expected_information(self) -> None:
    """MEI scores should match the manual P*I_correct + (1-P)*I_incorrect calculation.

    The manual score comparison makes sure both response branches are weighted
    correctly instead of merely exercising the estimator wiring.
    """
    item_bank = ItemBank(
      np.array(
        [
          [1.5, 0.0, 0.0, 1.0],
          [1.2, 0.5, 0.0, 1.0],
          [0.8, -0.5, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    administered_items = [0]
    response_vector = [True]
    est_theta = 0.4
    estimator = NumericalSearchEstimator()
    selector = MEISelector(estimator=estimator)

    candidate_mask = np.ones(item_bank.n_items, dtype=bool)
    candidate_mask[administered_items] = False
    candidates = np.flatnonzero(candidate_mask)
    manual_scores = np.array(
      [
        float(
          (
            irt.icc(est_theta, *item_bank.items[int(idx), :4])
            * item_bank.test_information(
              estimator.estimate(
                item_bank=item_bank,
                administered_items=[*administered_items, int(idx)],
                response_vector=[*response_vector, True],
                est_theta=est_theta,
              ),
              [*administered_items, int(idx)],
            )
          )
          + (
            (1.0 - irt.icc(est_theta, *item_bank.items[int(idx), :4]))
            * item_bank.test_information(
              estimator.estimate(
                item_bank=item_bank,
                administered_items=[*administered_items, int(idx)],
                response_vector=[*response_vector, False],
                est_theta=est_theta,
              ),
              [*administered_items, int(idx)],
            )
          )
        )
        for idx in candidates
      ],
      dtype=float,
    )

    selected = selector.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      rng=np.random.default_rng(0),
      response_vector=response_vector,
    )

    assert selected == 1
    assert selected == int(candidates[int(manual_scores.argmax())])

  def test_expected_information_matches_the_closed_form_on_1pl_items(self) -> None:
    """Under the 1PL model an item's information is P(1 - P), so each MEI score has a closed form."""
    difficulties = np.array([0.0, 0.8, -0.6, 1.5])
    item_bank = ItemBank(np.column_stack([np.ones(4), difficulties, np.zeros(4), np.ones(4)]))
    selector = MEISelector(estimator=ShiftEstimator())
    est_theta = 0.2

    def probability(theta: float, b: float) -> float:
      return 1.0 / (1.0 + np.exp(-(theta - b)))

    def test_information(theta: float, items: list[int]) -> float:
      return sum(probability(theta, difficulties[j]) * (1.0 - probability(theta, difficulties[j])) for j in items)

    expected = {}
    for candidate in (1, 2, 3):
      p_correct = probability(est_theta, difficulties[candidate])
      expected[candidate] = p_correct * test_information(est_theta + 0.5, [0, candidate]) + (
        1.0 - p_correct
      ) * test_information(est_theta - 0.5, [0, candidate])
      score = selector._expected_information(item_bank, [0], [True], candidate, est_theta)
      assert score == pytest.approx(expected[candidate], rel=1e-12)

    selected = selector.select(item_bank=item_bank, administered_items=[0], est_theta=est_theta, response_vector=[True])
    assert selected == max(expected, key=expected.__getitem__)

  def test_select_falls_back_to_max_info_when_a_look_ahead_is_infinite(self) -> None:
    """Without Dodd's heuristic an all-correct look-ahead is +inf; MEI should then match MaxInfo, not pick item 0."""
    item_bank = ItemBank(
      np.array(
        [[1.0, 2.0, 0.0, 1.0], [1.0, 1.5, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0], [1.5, 0.1, 0.0, 1.0]],
        dtype=float,
      )
    )
    selector = MEISelector(estimator=NumericalSearchEstimator(dodd=False))

    selected = selector.select(item_bank=item_bank, administered_items=[0], est_theta=0.0, response_vector=[True])

    assert selected == MaxInfoSelector().select(item_bank=item_bank, administered_items=[0], est_theta=0.0) == 3


class TestMLWISelector:
  """Tests for MLWISelector."""

  def test_select_requires_response_vector(self) -> None:
    """The selector needs the current response history."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    selector = MLWISelector()

    with pytest.raises(ValueError, match="response_vector"):
      selector.select(
        item_bank=item_bank,
        administered_items=[],
        est_theta=0.0,
        rng=np.random.default_rng(42),
      )

  def test_select_matches_manual_weighted_information(self) -> None:
    """Likelihood-weighted scores should match the manual integration."""
    item_bank = ItemBank.generate_item_bank(8, seed=42)
    selector = MLWISelector(n_nodes=11)
    administered_items = [0, 1]
    response_vector = [True, False]
    candidates = np.array([idx for idx in range(item_bank.n_items) if idx not in administered_items])

    selected = selector.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=0.0,
      rng=np.random.default_rng(42),
      response_vector=response_vector,
    )

    nodes = np.linspace(-6.0, 6.0, 11)
    delta = (6.0 - (-6.0)) / (11 - 1)
    log_likelihood = np.array(
      [irt.log_likelihood(float(theta), response_vector, item_bank.items[administered_items, :4]) for theta in nodes],
      dtype=float,
    )
    log_likelihood -= float(log_likelihood.max())
    weights = np.exp(log_likelihood)
    manual_scores = np.array(
      [
        float(
          np.sum(np.array([irt.inf(float(theta), *item_bank.items[int(idx), :4]) for theta in nodes]) * weights) * delta
        )
        for idx in candidates
      ],
      dtype=float,
    )
    expected = int(candidates[int(manual_scores.argmax())])

    assert selected == expected

  def test_select_falls_back_when_exposure_cap_blocks_all_candidates(self) -> None:
    """MLWI should still choose from non-administered items when every candidate exceeds r_max."""
    item_bank = ItemBank.generate_item_bank(8, seed=42)
    selector = MLWISelector(n_nodes=11, r_max=0.0)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[0, 1],
      est_theta=0.0,
      rng=np.random.default_rng(42),
      exposure_rates=np.ones(item_bank.n_items, dtype=float),
      response_vector=[True, False],
    )

    assert selected is not None
    assert selected not in {0, 1}

  def test_select_with_empty_response_vector_reduces_to_information_integral(self) -> None:
    """With no responses, MLWI should reduce to the integral of item information.

    This verifies the no-data degeneracy case where likelihood weights become
    uniform; if that fallback changed, MLWI would stop matching its mathematical definition.
    """
    item_bank = ItemBank.generate_item_bank(8, seed=0)
    n_nodes = 21
    selector = MLWISelector(n_nodes=n_nodes)
    nodes = np.linspace(-6.0, 6.0, n_nodes)
    delta = (6.0 - (-6.0)) / (n_nodes - 1)

    manual_scores = np.array(
      [
        float(np.sum(np.array([irt.inf(float(theta), *item_bank.items[idx, :4]) for theta in nodes])) * delta)
        for idx in range(item_bank.n_items)
      ],
      dtype=float,
    )

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      response_vector=[],
    )

    assert selected == int(manual_scores.argmax())

  def test_select_with_no_data_prefers_the_most_discriminating_item(self) -> None:
    """With flat weights a 2PL item's integrated information is its discrimination a, whatever its difficulty."""
    item_bank = ItemBank(
      np.array(
        [[0.8, 2.0, 0.0, 1.0], [1.9, -2.5, 0.0, 1.0], [1.2, 0.0, 0.0, 1.0], [1.5, 1.0, 0.0, 1.0]],
        dtype=float,
      )
    )

    selected = MLWISelector().select(item_bank=item_bank, administered_items=[], est_theta=0.0, response_vector=[])

    assert selected == 1

  def test_select_stays_finite_on_a_long_test(self) -> None:
    """Normalizing the log-likelihood keeps 300-item likelihood weights from underflowing."""
    item_bank = ItemBank.generate_item_bank(320, seed=7)
    rng = np.random.default_rng(7)
    administered_items = list(range(300))
    response_vector = [bool(rng.random() < irt.icc(0.4, *item_bank.items[idx, :4])) for idx in administered_items]
    selector = MLWISelector()

    weights = selector._likelihood_weights(response_vector, item_bank.items[administered_items, :4])
    selected = selector.select(
      item_bank=item_bank, administered_items=administered_items, est_theta=0.0, response_vector=response_vector
    )

    assert np.isfinite(weights).all()
    assert weights.max() == pytest.approx(1.0)
    assert selected in range(300, 320)

  def test_likelihood_weights_stay_finite_for_extreme_items(self) -> None:
    """Items far from every node should give finite weights, not NaN.

    With log(1 - P) computed after P rounded to 1, these two responses made the log-likelihood
    -inf at every node, and subtracting its maximum gave NaN weights.
    """
    items = np.array([[200.0, -6.0, 0.0, 1.0], [200.0, 6.0, 0.0, 1.0]])

    weights = MLWISelector()._likelihood_weights([False, True], items)

    assert np.isfinite(weights).all()
    assert weights.max() == pytest.approx(1.0)

  def test_init_rejects_non_increasing_bounds(self) -> None:
    """Reversed bounds would flip the sign of every score and select the least informative item."""
    with pytest.raises(ValueError, match="bounds must be increasing"):
      MLWISelector(bounds=(4.0, -4.0))
    with pytest.raises(ValueError, match="bounds must be increasing"):
      MLWISelector(bounds=(1.0, 1.0))

  def test_select_converges_to_max_info_on_long_test(self) -> None:
    """On a long test, MLWI should agree with MaxInfo near the MLE.

    As the likelihood sharpens, MLWI should concentrate on the MLE neighborhood,
    so disagreement here would signal broken weighting or integration.
    """
    item_bank = ItemBank.generate_item_bank(30, seed=1)
    theta_true = 0.7
    rng = np.random.default_rng(5)
    administered_items = list(range(20))
    response_vector = [
      bool(rng.random() < irt.icc(theta_true, *item_bank.items[idx, :4])) for idx in administered_items
    ]
    mle_theta = NumericalSearchEstimator().estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=theta_true,
    )

    mlwi = MLWISelector(n_nodes=81)
    max_info = MaxInfoSelector()

    mlwi_choice = mlwi.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=mle_theta,
      response_vector=response_vector,
    )
    max_info_choice = max_info.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=mle_theta,
    )

    assert mlwi_choice == max_info_choice


class TestKLSelector:
  """Tests for KLSelector."""

  @pytest.mark.parametrize(
    ("kwargs", "match"),
    [
      pytest.param({"c": 0.0}, "c must be positive", id="zero_c"),
      pytest.param({"c": -1.0}, "c must be positive", id="negative_c"),
      pytest.param({"r_max": 1.5}, "between 0 and 1", id="r_max_above_one"),
    ],
  )
  def test_init_rejects_invalid_arguments(self, kwargs: dict[str, float], match: str) -> None:
    """A window of zero width would make every item's KL integral 0 and the pick a random tie."""
    with pytest.raises(ValueError, match=match):
      KLSelector(**kwargs)

  def test_kl_integrand_is_exact_where_the_probability_rounds_to_one(self) -> None:
    """Test the KL integrand for an item whose correct-response probability rounds to 1.

    At theta_hat = b, P = 1/2, and at a logit of z = 40 the divergence is
    -log 2 + z/2 + log(1 + e^-z). Computing log(1 - P) after P rounds to 1 caps the
    second term, so this guards the logit-based log-probabilities.
    """
    kl_module = importlib.import_module("catsim.selection.kl")
    z = 40.0
    value = kl_module.__dict__["_kl_integrand"](4.0, 0.0, 10.0, 0.0, 0.0, 1.0)
    assert value == pytest.approx(-np.log(2.0) + z / 2 + np.log1p(np.exp(-z)), rel=1e-12)

  @pytest.mark.parametrize(
    "item",
    [
      pytest.param((1.0, 0.0, 0.0, 1.0), id="2pl"),
      pytest.param((1.5, 0.3, 0.2, 1.0), id="3pl"),
      pytest.param((1.2, -0.5, 0.0, 0.9), id="4pl_without_guessing"),
      pytest.param((1.8, 0.4, 0.15, 0.85), id="4pl"),
    ],
  )
  @pytest.mark.parametrize(("theta", "theta_hat"), [(-2.0, 1.0), (0.5, -0.5), (3.0, 2.5)])
  def test_kl_integrand_matches_the_bernoulli_divergence(
    self, item: tuple[float, float, float, float], theta: float, theta_hat: float
  ) -> None:
    """Test the logit-based integrand against P-hat log(P-hat/P) + Q-hat log(Q-hat/Q).

    The c = 0 and d = 1 cases reach the logaddexp branches with an infinite term, the
    others with two finite ones, so this covers every path of the log-probabilities.
    """
    kl_module = importlib.import_module("catsim.selection.kl")
    p_hat, p = irt.icc(theta_hat, *item), irt.icc(theta, *item)
    expected = p_hat * np.log(p_hat / p) + (1 - p_hat) * np.log((1 - p_hat) / (1 - p))

    value = kl_module.__dict__["_kl_integrand"](theta, theta_hat, *item)

    assert value == pytest.approx(expected, rel=1e-10, abs=1e-14)

  @pytest.mark.parametrize("n_administered", [0, 1, 4, 9])
  def test_select_maximizes_kl_over_a_window_shrinking_with_the_test(self, n_administered: int) -> None:
    """The pick has the largest KL integral over theta_hat +- c / sqrt(max(1, n)) among the remaining items.

    The reference integrates the Bernoulli divergence, written out in the test, with the trapezoid rule.
    It fails if the window ignores the number of administered items, shrinks it by n instead of sqrt(n),
    or integrates over the wrong interval.
    """
    rng = np.random.default_rng(n_administered)
    for _ in range(10):
      item_bank = ItemBank.generate_item_bank(25, itemtype="3PL", seed=int(rng.integers(1_000_000)))
      administered = [int(idx) for idx in rng.choice(25, n_administered, replace=False)]
      theta_hat = float(rng.uniform(-1.5, 1.5))
      half_width = 2.0 / np.sqrt(max(1, n_administered))
      grid = np.linspace(theta_hat - half_width, theta_hat + half_width, 4001)
      a, b, c, d = (item_bank.items[:, [k]] for k in range(4))
      p = c + (d - c) / (1.0 + np.exp(-a * (grid - b)))
      p_hat = c + (d - c) / (1.0 + np.exp(-a * (theta_hat - b)))
      divergence = p_hat * np.log(p_hat / p) + (1.0 - p_hat) * np.log((1.0 - p_hat) / (1.0 - p))
      reference = np.trapezoid(divergence, grid, axis=1)
      reference[administered] = -np.inf

      picked = KLSelector(c=2.0).select(item_bank, administered, theta_hat, rng=np.random.default_rng(0))

      assert reference[picked] == pytest.approx(reference.max(), rel=1e-6)

  @pytest.mark.parametrize(("administered", "expected"), [([2], 1), ([2, 3, 4, 5], 0)])
  def test_window_shrinks_with_the_square_root_of_the_test_length(self, administered: list[int], expected: int) -> None:
    """With c = 2 the window is theta_hat +- 2 after one item and +- 1 after four.

    At theta_hat = 0, item 0 (a = 1, b = 0) and item 1 (a = 4, b = 1.5) trade places as the window narrows:
    their KL integrals are 0.610 and 0.869 over +- 2, 0.225 and 0.146 over +- 1.41, and 0.081 and 0.028
    over +- 1. A window of c / sqrt(n + 1) would already prefer item 0 after one item.
    """
    filler = [1.0, 3.0, 0.0, 1.0]
    item_bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [4.0, 1.5, 0.0, 1.0], filler, filler, filler, filler]))

    assert KLSelector(c=2.0).select(item_bank, administered, 0.0, rng=np.random.default_rng(0)) == expected

  def test_select_skips_items_at_the_exposure_cap(self) -> None:
    """The item with the most KL information is passed over while its exposure rate is at r_max."""
    item_bank = ItemBank(np.array([[1.0, -2.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 1.5, 0.0, 1.0]]))
    selector = KLSelector(c=0.5, r_max=0.5)

    uncapped = selector.select(item_bank, [], 0.0, rng=np.random.default_rng(0))
    capped = selector.select(item_bank, [], 0.0, rng=np.random.default_rng(0), exposure_rates=np.array([0.0, 0.5, 0.0]))

    assert (uncapped, capped) == (1, 2)

  def test_select_raises_when_exhausted(self) -> None:
    """Test that select raises NoItemsAvailableError when all items are administered."""
    item_bank = ItemBank(np.array([[1.0, -2.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]]))

    with pytest.raises(NoItemsAvailableError, match="no more items"):
      KLSelector().select(
        item_bank=item_bank, administered_items=[0, 1, 2], est_theta=0.0, rng=np.random.default_rng(42)
      )


class TestClusterSelector:
  """Tests for ClusterSelector."""

  # All difficulties are 0, so at theta = 0 an item's information is a^2 / 4. Cluster 0 holds the single most
  # informative item (1), cluster 1 the largest information sum (its best item is 7), and cluster 2 the largest
  # average information (its best item is 13).
  CLUSTERS = (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2)
  DISCRIMINATIONS = (0.2, 2.0, 0.2, 0.2, 1.0, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.0, 1.7, 1.8)

  def _select(
    self,
    administered: list[int],
    exposure_rates: np.ndarray | None = None,
    method: str = "item_info",
    r_max: float = 1.0,
    r_control: str = "passive",
  ) -> int | None:
    item_bank = _two_pl_bank(list(self.DISCRIMINATIONS), [0.0] * len(self.DISCRIMINATIONS))
    selector = ClusterSelector(clusters=list(self.CLUSTERS), method=method, r_max=r_max, r_control=r_control)
    return selector.select(
      item_bank=item_bank, administered_items=administered, est_theta=0.0, exposure_rates=exposure_rates
    )

  @pytest.mark.parametrize(("method", "expected"), [("item_info", 1), ("cluster_info", 7), ("weighted_info", 13)])
  def test_select_takes_the_best_item_of_the_cluster_each_method_prefers(self, method: str, expected: int) -> None:
    """Each method ranks clusters differently; the pick is the most informative item of the chosen cluster."""
    assert self._select([], method=method) == expected

  def test_select_moves_on_once_a_cluster_is_fully_administered(self) -> None:
    """With cluster 0 used up, item_info turns to the cluster of the next most informative item."""
    assert self._select([0, 1, 2, 3]) == 13

  def test_select_raises_when_every_cluster_is_exhausted(self) -> None:
    """An exhausted bank raises NoItemsAvailableError, which the engine turns into a stop."""
    with pytest.raises(NoItemsAvailableError):
      self._select(list(range(len(self.CLUSTERS))))

  def test_select_prefers_items_below_the_exposure_cap(self) -> None:
    """Inside the chosen cluster, an item at the cap is passed over for a less informative one below it."""
    exposure_rates = np.zeros(len(self.CLUSTERS))
    exposure_rates[13] = 0.9

    assert self._select([], exposure_rates, method="weighted_info", r_max=0.5) == 12

  @pytest.mark.parametrize(
    ("rates", "r_control", "expected"),
    [
      ((0.8, 0.9), "passive", 13),
      ((0.8, 0.9), "aggressive", 12),
      # item 13 is now the least exposed and still the more informative: not the first or least informative item
      ((0.9, 0.8), "aggressive", 13),
    ],
  )
  def test_r_control_decides_when_the_whole_cluster_is_over_the_cap(
    self, rates: tuple[float, float], r_control: str, expected: int
  ) -> None:
    """Passive control still administers the most informative item; aggressive control the least exposed one."""
    exposure_rates = np.zeros(len(self.CLUSTERS))
    exposure_rates[[12, 13]] = rates

    assert self._select([], exposure_rates, method="weighted_info", r_max=0.5, r_control=r_control) == expected


_N_ITEMS = 30
SELECTOR_FACTORIES = {
  "MaxInfoSelector": MaxInfoSelector,
  "UrrySelector": UrrySelector,
  "KLSelector": KLSelector,
  "IntervalInfoSelector": IntervalInfoSelector,
  "RandomSelector": RandomSelector,
  "RandomesqueSelector": lambda: RandomesqueSelector(bin_size=5),
  "The54321Selector": The54321Selector,
  "LinearSelector": lambda: LinearSelector(list(range(_N_ITEMS))),
  "ClusterSelector": lambda: ClusterSelector(clusters=[i % 3 for i in range(_N_ITEMS)]),
  "AStratSelector": lambda: AStratSelector(test_size=10),
  "AStratBBlockSelector": lambda: AStratBBlockSelector(test_size=10),
  "MaxInfoStratSelector": lambda: MaxInfoStratSelector(test_size=10),
  "MaxInfoBBlockSelector": lambda: MaxInfoBBlockSelector(test_size=10),
  "MEISelector": MEISelector,
  "MLWISelector": MLWISelector,
}


@pytest.mark.parametrize("make_selector", SELECTOR_FACTORIES.values(), ids=SELECTOR_FACTORIES.keys())
def test_select_returns_builtin_int(make_selector) -> None:
  """Called with every input CatEngine passes, select() returns a builtin int.

  A selector that does not accept response_vector fails inside the engine, and numpy integers must
  not leak into session state.
  """
  item_bank = ItemBank.generate_item_bank(_N_ITEMS, seed=3)

  item_id = make_selector().select(
    item_bank=item_bank,
    administered_items=[0, 1],
    est_theta=0.3,
    rng=np.random.default_rng(0),
    response_vector=[True, False],
  )

  assert type(item_id) is int


def test_selector_factories_cover_every_concrete_selector() -> None:
  """Every concrete selector exported by catsim.selection runs through the engine-input test above.

  A selector left out of SELECTOR_FACTORIES could reject the engine's inputs without any test noticing.
  """
  exported = {
    name
    for name, cls in vars(importlib.import_module("catsim.selection")).items()
    if isinstance(cls, type) and issubclass(cls, BaseSelector) and not inspect.isabstract(cls)
  }

  assert exported == set(SELECTOR_FACTORIES)
