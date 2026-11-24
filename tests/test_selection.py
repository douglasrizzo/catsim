"""Tests for catsim.selection module."""

import importlib
import inspect
from collections.abc import Callable
from fractions import Fraction
from itertools import pairwise

import numpy as np
import pytest
from scipy.integrate import quad

from catsim import Estimate, StepContext, irt
from catsim.estimation import BaseEstimator, NumericalSearchEstimator
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
  ProgressiveSelector,
  ProportionalSelector,
  RandomesqueSelector,
  RandomSelector,
  The54321Selector,
  UrrySelector,
  below_cap,
  progressive,
  remaining,
)
from tests.contexts import make_context


class RecordingEstimator(BaseEstimator):
  """Estimator stub that records the history it receives."""

  def __init__(self) -> None:
    self.records: list[tuple[tuple[int, ...], tuple[bool, ...], float]] = []

  def estimate(self, context: StepContext) -> Estimate:
    self.records.append((context.administered_items, context.responses, context.theta))
    return Estimate(theta=float(len(context.responses)), se=1.0)

  def __repr__(self) -> str:
    return "RecordingEstimator()"


class ShiftEstimator(BaseEstimator):
  """Estimator stub that moves the estimate 0.5 toward the last response."""

  def estimate(self, context: StepContext) -> Estimate:
    return Estimate(theta=context.theta + (0.5 if context.responses[-1] else -0.5), se=1.0)


def _two_pl_bank(discriminations: list[float], difficulties: list[float]) -> ItemBank:
  """Build a 2PL bank, whose item information at theta is a^2 P (1 - P)."""
  n_items = len(discriminations)
  return ItemBank(np.column_stack([discriminations, difficulties, np.zeros(n_items), np.ones(n_items)]))


class _DeterministicRNG:
  """Deterministic stand-in for numpy's Generator in selector tests."""

  def __init__(self, uniform_values: list[float] | np.ndarray) -> None:
    self._uniform_values = np.asarray(uniform_values, dtype=float)
    self.choice_calls: list[tuple[np.ndarray, np.ndarray | None]] = []

  def uniform(
    self,
    low: float = 0.0,  # noqa: ARG002
    high: float = 1.0,  # noqa: ARG002
    size: int | tuple[int, ...] | None = None,
  ) -> float | np.ndarray:
    if size is None:
      return float(self._uniform_values[0])
    expected_size = int(np.prod(size)) if isinstance(size, tuple) else int(size)
    if expected_size != self._uniform_values.size:
      msg = f"expected {expected_size} uniform values, got {self._uniform_values.size}"
      raise ValueError(msg)
    return self._uniform_values.copy()

  def choice(self, a: np.ndarray | list[int], p: np.ndarray | None = None) -> int:
    candidate_indices = np.asarray(a, dtype=int)
    probabilities = None if p is None else np.asarray(p, dtype=float)
    self.choice_calls.append((
      candidate_indices.copy(),
      probabilities.copy() if probabilities is not None else None,
    ))
    if probabilities is None:
      return int(candidate_indices[0])
    return int(candidate_indices[int(probabilities.argmax())])


def _progressive_item_bank() -> ItemBank:
  """Create a small bank with a clear information ranking at theta=0."""
  return ItemBank(
    np.array(
      [
        [1.0, -3.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
      ],
      dtype=float,
    )
  )


class TestRemaining:
  """Tests for remaining, the items a selector may still administer."""

  BANK = ItemBank.generate(6, rng=0)

  def test_returns_the_items_not_administered_in_ascending_order(self) -> None:
    """An administered item left in would be administered twice; a dropped one could never be chosen.

    With no items administered, a tuple index of () would select the whole mask and leave no candidate.
    """
    assert remaining(make_context(self.BANK, [4, 1])).tolist() == [0, 2, 3, 5]
    assert remaining(make_context(self.BANK)).tolist() == [0, 1, 2, 3, 4, 5]

  def test_leaves_out_the_excluded_items(self) -> None:
    """An excluded item left in could be handed back to the exposure-control wrapper that had just declined it."""
    assert remaining(make_context(self.BANK, [4], excluded_items=[0, 5])).tolist() == [1, 2, 3]

  @pytest.mark.parametrize(("administered", "excluded"), [([2, 0, 1, 5, 4, 3], []), ([2, 0, 1], [5, 4, 3])])
  def test_raises_when_every_item_is_administered_or_excluded(
    self, administered: list[int], excluded: list[int]
  ) -> None:
    """An empty candidate array would reach the scorer instead of ending the session with no item to give."""
    with pytest.raises(NoItemsAvailableError, match="no more items"):
      remaining(make_context(self.BANK, administered, excluded_items=excluded))


class TestBelowCap:
  """Tests for below_cap, the exposure filter of the capped selectors."""

  RATES = np.array([0.1, 0.9, 0.5, 0.2, 0.0, 0.6])

  def test_keeps_the_candidates_below_the_cap(self) -> None:
    """Item 2 sits exactly at the cap and is passed over; item 4 is below it but not a candidate."""
    assert below_cap(np.array([0, 2, 3, 5]), self.RATES, 0.5).tolist() == [0, 3]

  @pytest.mark.parametrize("rates", [RATES, None], ids=["every_candidate_at_or_over_the_cap", "no_rates"])
  def test_falls_back_to_every_candidate(self, rates: np.ndarray | None) -> None:
    """With no candidate below the cap, or no rates to compare, an empty result would leave no item to administer."""
    assert below_cap(np.array([1, 2, 5]), rates, 0.5).tolist() == [1, 2, 5]


CAPPED_SELECTORS: dict[str, Callable[[float], BaseSelector]] = {
  "MaxInfoSelector": lambda r_max: MaxInfoSelector(r_max=r_max),
  "KLSelector": lambda r_max: KLSelector(r_max=r_max),
  "MEISelector": lambda r_max: MEISelector(r_max=r_max),
  "MLWISelector": lambda r_max: MLWISelector(r_max=r_max),
  "ProgressiveSelector": lambda r_max: ProgressiveSelector(test_size=10, r_max=r_max),
  "ProportionalSelector": lambda r_max: ProportionalSelector(test_size=10, r_max=r_max),
  "ClusterSelector": lambda r_max: ClusterSelector(clusters=[0, 1], r_max=r_max),
}


@pytest.mark.parametrize("r_max", [0.0, -0.1, 1.5])
@pytest.mark.parametrize("make_selector", CAPPED_SELECTORS.values(), ids=CAPPED_SELECTORS.keys())
def test_capped_selectors_reject_r_max_outside_zero_to_one(
  make_selector: Callable[[float], BaseSelector], r_max: float
) -> None:
  """A cap of 0 used to be accepted: every item is at or above it, so the selector fell back to all items.

  It then capped nothing, exactly like r_max = 1.
  """
  with pytest.raises(ValueError, match=r"r_max must be in \(0, 1\]"):
    make_selector(r_max)


def test_capped_selectors_cover_every_selector_with_r_max() -> None:
  """A selector that takes r_max but is left out of CAPPED_SELECTORS could accept r_max = 0 unnoticed."""
  takes_r_max = {
    name
    for name, cls in vars(importlib.import_module("catsim.selection")).items()
    if isinstance(cls, type) and issubclass(cls, BaseSelector) and "r_max" in inspect.signature(cls).parameters
  }

  assert takes_r_max == set(CAPPED_SELECTORS)


@pytest.mark.parametrize("test_size", [0, -5])
@pytest.mark.parametrize(
  "selector_class",
  [
    AStratSelector,
    AStratBBlockSelector,
    MaxInfoStratSelector,
    MaxInfoBBlockSelector,
    ProgressiveSelector,
    ProportionalSelector,
  ],
)
def test_selectors_reject_test_size_below_one(selector_class: Callable[..., BaseSelector], test_size: int) -> None:
  """ProgressiveSelector(test_size=-5) used to construct and select by maximum information from the first item.

  ProportionalSelector(test_size=0) constructed too, and the stratified selectors rejected it only through a message
  about n_strata.
  """
  with pytest.raises(ValueError, match="test_size must be at least 1"):
    selector_class(test_size=test_size)


class TestMaxInfoSelector:
  """Tests for MaxInfoSelector."""

  # all difficulties are 0, so at theta = 0 information is a^2 / 4 and ranks items 1, 2, 0, 3
  BANK_A = (1.0, 2.0, 1.5, 0.5)

  def test_select_returns_the_most_informative_remaining_item(self) -> None:
    """The pick is the remaining item with the largest information at the estimate."""
    item_bank = _two_pl_bank(list(self.BANK_A), [0.0] * 4)
    selector = MaxInfoSelector()

    assert selector.select(make_context(item_bank, [], theta=0.0)) == 1
    assert selector.select(make_context(item_bank, [1], theta=0.0)) == 2

  def test_select_skips_items_at_the_exposure_cap(self) -> None:
    """An item whose exposure rate has reached r_max is passed over for the next most informative one."""
    item_bank = _two_pl_bank(list(self.BANK_A), [0.0] * 4)
    exposure_rates = np.array([0.0, 0.5, 0.0, 0.0])

    selected = MaxInfoSelector(r_max=0.5).select(make_context(item_bank, [], theta=0.0, exposure_rates=exposure_rates))

    assert selected == 2

  def test_the_default_cap_of_1_holds_back_no_item(self) -> None:
    """r_max = 1 applies no cap, as in catR, even to an item every earlier examinee received.

    Such an item has a rate of exactly 1. Fails if the strict ``rate < r_max`` test passes it over, which made the
    default selector give the second examinee none of the first examinee's items.
    """
    item_bank = _two_pl_bank(list(self.BANK_A), [0.0] * 4)

    selected = MaxInfoSelector().select(
      make_context(item_bank, [], theta=0.0, exposure_rates=np.array([0.0, 1.0, 0.0, 0.0]))
    )

    assert selected == 1

  def test_select_ignores_the_cap_when_every_item_is_over_it(self) -> None:
    """With every remaining item at or over r_max, the most informative one is still administered."""
    item_bank = _two_pl_bank(list(self.BANK_A), [0.0] * 4)

    selected = MaxInfoSelector(r_max=0.5).select(make_context(item_bank, [], theta=0.0, exposure_rates=np.ones(4)))

    assert selected == 1


class TestRandomSelector:
  """Tests for RandomSelector."""

  def test_select_draws_every_remaining_item(self) -> None:
    """Over many seeds every remaining item is drawn; a selector that ignored rng or skipped items would miss some."""
    item_bank = ItemBank.generate(50, rng=42)
    administered = [3, 17, 25]

    selections = {
      RandomSelector().select(make_context(item_bank, administered, theta=0.0, rng=np.random.default_rng(seed)))
      for seed in range(1000)
    }

    # each item is missed by all 1000 draws with probability (46/47) ** 1000, about 4e-10
    assert selections == set(range(50)) - set(administered)


class TestLinearSelector:
  """Tests for LinearSelector."""

  def test_select_follows_the_order_of_the_list(self) -> None:
    """A selector that administered its items in bank order would return 5 first instead of 15."""
    selector = LinearSelector([15, 5, 10])
    item_bank = ItemBank.generate(50, rng=42)

    administered: list[int] = []
    for _ in range(3):
      administered.append(selector.select(make_context(item_bank, administered, theta=0.0)))

    assert administered == [15, 5, 10]

  def test_select_rejects_items_outside_the_bank(self) -> None:
    """LinearSelector([0, 99]) on a 20-item bank used to return 99, an index the engine then rejected."""
    selector = LinearSelector([0, 99])
    item_bank = ItemBank.generate(20, rng=1)

    with pytest.raises(ValueError, match=r"items \[99\] are not in the item bank, whose indices run from 0 to 19"):
      selector.select(make_context(item_bank, [0], theta=0.0))


class TestUrrySelector:
  """Tests for UrrySelector."""

  def test_select_matches_difficulty_to_the_estimate_whatever_the_discrimination(self) -> None:
    """The pick is the remaining item whose difficulty is closest to the estimate, not the most informative one."""
    # at theta = 0.8, item 0 is the most informative but item 1's difficulty is the closest
    item_bank = _two_pl_bank([2.5, 0.5, 1.0, 1.0], [0.3, 0.9, -1.0, 2.5])
    selector = UrrySelector()

    assert MaxInfoSelector().select(make_context(item_bank, [], theta=0.8)) == 0
    assert selector.select(make_context(item_bank, [], theta=0.8)) == 1
    assert selector.select(make_context(item_bank, [1], theta=0.8)) == 0


class TestRandomesqueSelector:
  """Tests for RandomesqueSelector."""

  def test_init_rejects_a_bin_smaller_than_one(self) -> None:
    """RandomesqueSelector(bin_size=0) used to construct and fail only at its first selection."""
    with pytest.raises(ValueError, match="bin_size must be at least 1, got 0"):
      RandomesqueSelector(bin_size=0)

  def test_draws_from_the_bin_of_most_informative_remaining_items(self) -> None:
    """Every pick is one of the bin_size most informative remaining items, and each of them gets picked."""
    item_bank = ItemBank.generate(50, rng=42)
    administered = [3, 17, 25]
    by_information = [int(i) for i in np.argsort(-item_bank.information(-0.4)) if i not in administered]
    selector = RandomesqueSelector(4)
    rng = np.random.default_rng(0)

    picks = {selector.select(make_context(item_bank, administered, theta=-0.4, rng=rng)) for _ in range(200)}

    assert picks == set(by_information[:4])


class TestThe54321Selector:
  """The 5-4-3-2-1 rule of McBride & Martin (1983)."""

  def test_init_default_and_validation(self) -> None:
    """McBride and Martin draw the first item from the top 5; an initial bin of 0 would leave nothing to draw from."""
    assert The54321Selector().initial_bin_size == 5
    with pytest.raises(ValueError, match="initial_bin_size must be at least 1, got 0"):
      The54321Selector(0)

  @pytest.mark.parametrize(("n_administered", "bin_size"), [(0, 5), (1, 4), (2, 3), (3, 2), (4, 1), (12, 1)])
  def test_draws_from_a_shrinking_bin_of_most_informative_items(self, n_administered: int, bin_size: int) -> None:
    """Item k is drawn from the 5 - k most informative remaining items, and from the fifth on it is the best one.

    Earlier versions sized the bin as test_size - k, so a 20-item test drew its first item from the top
    20, and past test_size the negative bin drew from almost the whole bank.
    """
    item_bank = ItemBank.generate(50, rng=42)
    administered = list(range(n_administered))
    by_information = [int(i) for i in np.argsort(-item_bank.information(0.3)) if i not in administered]
    selector = The54321Selector()
    rng = np.random.default_rng(0)

    picks = {selector.select(make_context(item_bank, administered, theta=0.3, rng=rng)) for _ in range(200)}

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
    item_bank = ItemBank.generate(200, model="3PL", rng=1)
    _, location = _stratified_keys(item_bank, selector_class)
    tests = {}
    for theta in (-2.0, 2.0):
      selector = selector_class(10)
      administered: list[int] = []
      for position in range(10):
        item = selector.select(make_context(item_bank, administered, theta=theta))
        remaining = [i for i in selector.strata(item_bank)[position] if i not in administered]
        assert item in remaining
        assert abs(location[item] - theta) == pytest.approx(np.abs(location[remaining] - theta).min())
        administered.append(item)
      tests[theta] = administered

    assert set(tests[-2.0]).isdisjoint(tests[2.0])

  @pytest.mark.parametrize("selector_class", [AStratSelector, MaxInfoStratSelector])
  def test_strata_ascend_in_the_stratification_key(self, selector_class: type) -> None:
    """Without blocking, the bank is split into contiguous strata of increasing a (or maximum information)."""
    item_bank = ItemBank.generate(103, model="3PL", rng=2)
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
    item_bank = ItemBank.generate(102, model="3PL", corr=0.5, rng=3)
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
    item_bank = ItemBank.generate(250, model="3PL", rng=4)
    selector = AStratSelector(30, n_strata=5)
    strata = selector.strata(item_bank)
    administered: list[int] = []
    for position in range(30):
      item = selector.select(make_context(item_bank, administered, theta=0.5))
      assert item in strata[position // 6]
      administered.append(item)

  def test_raises_when_run_past_test_size(self) -> None:
    """A test longer than test_size used to stay in the last stratum and, once it emptied, end as an exhausted bank.

    AStratSelector(test_size=4) under a 20-item stopper stopped every session at 13 items of a 40-item bank with
    stop reason "item_bank_exhausted".
    """
    item_bank = ItemBank.generate(60, rng=5)
    selector = MaxInfoStratSelector(5)
    administered: list[int] = []
    for _ in range(5):
      administered.append(selector.select(make_context(item_bank, administered, theta=0.0)))

    with pytest.raises(ValueError, match="MaxInfoStratSelector has test_size=5, but 5 items have already been"):
      selector.select(make_context(item_bank, administered, theta=0.0))

  def test_raises_when_the_current_stratum_is_exhausted(self) -> None:
    """A stage whose stratum has no items left raises NoItemsAvailableError."""
    item_bank = ItemBank.generate(6, rng=6)
    selector = AStratSelector(10, n_strata=2)
    first_stratum = selector.strata(item_bank)[0].tolist()

    with pytest.raises(NoItemsAvailableError, match="stratum 0"):
      selector.select(make_context(item_bank, first_stratum, theta=0.0))

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

  def test_default_interval_is_scaled_by_the_context_standard_error(self) -> None:
    """A selector that ignored context.se, such as one that recomputed the Fisher standard error, picks one item twice.

    Over [-w, w] the 2PL integral is a [sigma(a (w - b)) - sigma(a (-w - b))]. With w = 1.96 * 0.15 = 0.29, item 0
    (a = 2, b = 0) scores 0.572 and item 1 (a = 3, b = 1.5) 0.065; with w = 1.96 * 1.5 = 2.94, 1.989 and 2.961.
    """
    item_bank = ItemBank(np.array([[2.0, 0.0, 0.0, 1.0], [3.0, 1.5, 0.0, 1.0]]))

    narrow = IntervalInfoSelector().select(make_context(item_bank, theta=0.0, se=0.15))
    wide = IntervalInfoSelector().select(make_context(item_bank, theta=0.0, se=1.5))

    assert (narrow, wide) == (0, 1)

  def test_default_interval_with_the_fisher_standard_error_is_equation_11(self) -> None:
    """With the MLE's standard error 1 / sqrt(I(theta_hat)), the interval must be Veerkamp & Berger's (1994) Eq. 11.

    The quadrature over theta_hat +/- 1.96 SE picks item 51; the whole scale, an interval of the same width centred at 0
    and one 2.576 SE wide each pick item 24.
    """
    item_bank = ItemBank.generate(60, model="3PL", rng=7)
    administered = [3, 11, 25, 40, 52]
    theta = 0.7
    se = 1 / np.sqrt(irt.total_information(theta, item_bank.items[administered]))
    half_width = 1.959963984540054 * se
    remaining = [i for i in range(item_bank.n_items) if i not in administered]
    integrals = [
      quad(irt.information, theta - half_width, theta + half_width, args=(item_bank.items[i],))[0] for i in remaining
    ]

    selected = IntervalInfoSelector().select(make_context(item_bank, administered, theta=theta, se=se))

    assert selected == remaining[int(np.argmax(integrals))]

  def test_default_interval_is_the_whole_scale_while_the_standard_error_is_infinite(self) -> None:
    """An infinite standard error must integrate over the whole ability scale (Equation 16), where a 2PL item scores a.

    Item 1 (a = 2, b = 5) then scores 2 against item 0's (a = 1, b = 0) 1. Over any interval narrower than
    theta_hat +/- 4.98, item 0 scores more.
    """
    item_bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [2.0, 5.0, 0.0, 1.0]]))

    assert IntervalInfoSelector().select(make_context(item_bank, theta=0.0, se=np.inf)) == 1

  @pytest.mark.parametrize("se", [0.0, 1e-300, 1e-16])
  def test_a_vanishing_interval_selects_the_most_informative_item_at_the_estimate(self, se: float) -> None:
    """A zero or near-zero standard error selects by the items' information at theta_hat, the interval's limit.

    At theta = 0.3 the 2PL items (a, b) = (2, 0), (1, 3) and (3, -0.2) give a^2 P (1 - P) = 0.915, 0.059 and 1.34,
    so item 2. Fails if the selector integrates over the collapsed interval: every integral is 0 (a random tie) or
    rounding noise, which picks another item for some seeds.
    """
    item_bank = ItemBank(np.array([[2.0, 0.0, 0.0, 1.0], [1.0, 3.0, 0.0, 1.0], [3.0, -0.2, 0.0, 1.0]]))

    selected = {
      IntervalInfoSelector().select(make_context(item_bank, theta=0.3, se=se, rng=np.random.default_rng(seed)))
      for seed in range(20)
    }

    assert selected == {2}

  def test_fixed_interval_matches_numerical_integration(self) -> None:
    """A fixed half-width integrates information over theta_hat +/- interval."""
    item_bank = ItemBank.generate(40, rng=8)
    integrals = [quad(irt.information, -1.3, -0.3, args=(item,))[0] for item in item_bank.items]

    selected = IntervalInfoSelector(interval=0.5).select(make_context(item_bank, [], theta=-0.8))

    assert selected == int(np.argmax(integrals))

  def test_select_excludes_administered_and_raises_when_exhausted(self) -> None:
    """Administered items are never returned; an exhausted bank raises NoItemsAvailableError."""
    item_bank = ItemBank.generate(3, rng=9)
    selector = IntervalInfoSelector(interval=1.0)

    assert selector.select(make_context(item_bank, [0, 2], theta=0.0)) == 1
    with pytest.raises(NoItemsAvailableError):
      selector.select(make_context(item_bank, [0, 1, 2], theta=0.0))


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
    responses = [True]
    theta = 0.4
    estimator = NumericalSearchEstimator()
    selector = MEISelector(estimator=estimator)

    candidate_mask = np.ones(item_bank.n_items, dtype=bool)
    candidate_mask[administered_items] = False
    candidates = np.flatnonzero(candidate_mask)
    manual_scores = np.array(
      [
        float(
          (
            irt.icc(theta, item_bank.items[int(idx)])
            * item_bank.total_information(
              estimator.estimate(
                make_context(item_bank, [*administered_items, int(idx)], [*responses, True], theta=theta)
              ).theta,
              [*administered_items, int(idx)],
            )
          )
          + (
            (1.0 - irt.icc(theta, item_bank.items[int(idx)]))
            * item_bank.total_information(
              estimator.estimate(
                make_context(item_bank, [*administered_items, int(idx)], [*responses, False], theta=theta)
              ).theta,
              [*administered_items, int(idx)],
            )
          )
        )
        for idx in candidates
      ],
      dtype=float,
    )

    selected = selector.select(
      make_context(item_bank, administered_items, responses, theta=theta, rng=np.random.default_rng(0))
    )

    assert selected == 1
    assert selected == int(candidates[int(manual_scores.argmax())])

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


class TestMLWISelector:
  """Tests for MLWISelector."""

  def test_select_matches_manual_weighted_information(self) -> None:
    """Likelihood-weighted scores should match the manual integration."""
    item_bank = ItemBank.generate(8, rng=42)
    selector = MLWISelector(n_nodes=11)
    administered_items = [0, 1]
    responses = [True, False]
    candidates = np.array([idx for idx in range(item_bank.n_items) if idx not in administered_items])

    selected = selector.select(
      make_context(item_bank, administered_items, responses, theta=0.0, rng=np.random.default_rng(42))
    )

    nodes = np.linspace(-6.0, 6.0, 11)
    delta = (6.0 - (-6.0)) / (11 - 1)
    log_likelihood = np.array(
      [irt.log_likelihood(float(theta), responses, item_bank.get_items(administered_items)) for theta in nodes],
      dtype=float,
    )
    log_likelihood -= float(log_likelihood.max())
    weights = np.exp(log_likelihood)
    manual_scores = np.array(
      [float(np.sum(irt.information(nodes, item_bank.items[int(idx)]) * weights) * delta) for idx in candidates],
      dtype=float,
    )
    expected = int(candidates[int(manual_scores.argmax())])

    assert selected == expected

  def test_select_falls_back_when_exposure_cap_blocks_all_candidates(self) -> None:
    """With every remaining item over r_max, the cap is ignored and the best remaining item is still chosen."""
    item_bank = ItemBank.generate(8, rng=42)
    selector = MLWISelector(n_nodes=11, r_max=0.5)

    selected = selector.select(
      make_context(
        item_bank,
        [0, 1],
        [True, False],
        theta=0.0,
        rng=np.random.default_rng(42),
        exposure_rates=np.ones(item_bank.n_items, dtype=float),
      )
    )

    # the pick of the manual integration in test_select_matches_manual_weighted_information
    assert selected == 6

  def test_select_with_no_responses_reduces_to_information_integral(self) -> None:
    """With no responses, MLWI should reduce to the integral of item information.

    This verifies the no-data degeneracy case where likelihood weights become
    uniform; if that fallback changed, MLWI would stop matching its mathematical definition.
    """
    item_bank = ItemBank.generate(8, rng=0)
    n_nodes = 21
    selector = MLWISelector(n_nodes=n_nodes)
    nodes = np.linspace(-6.0, 6.0, n_nodes)
    delta = (6.0 - (-6.0)) / (n_nodes - 1)

    manual_scores = np.array(
      [float(np.sum(irt.information(nodes, item_bank.items[idx])) * delta) for idx in range(item_bank.n_items)],
      dtype=float,
    )

    selected = selector.select(make_context(item_bank, [], [], theta=0.0))

    assert selected == int(manual_scores.argmax())

  def test_select_with_no_data_prefers_the_most_discriminating_item(self) -> None:
    """With flat weights a 2PL item's integrated information is its discrimination a, whatever its difficulty."""
    item_bank = ItemBank(
      np.array(
        [[0.8, 2.0, 0.0, 1.0], [1.9, -2.5, 0.0, 1.0], [1.2, 0.0, 0.0, 1.0], [1.5, 1.0, 0.0, 1.0]],
        dtype=float,
      )
    )

    selected = MLWISelector().select(make_context(item_bank, [], [], theta=0.0))

    assert selected == 1

  def test_select_stays_finite_on_a_long_test(self) -> None:
    """Normalizing the log-likelihood keeps 300-item likelihood weights from underflowing."""
    item_bank = ItemBank.generate(320, rng=7)
    rng = np.random.default_rng(7)
    administered_items = list(range(300))
    responses = [bool(rng.random() < irt.icc(0.4, item_bank.items[idx])) for idx in administered_items]
    selector = MLWISelector()

    weights = selector._likelihood_weights(responses, item_bank.get_items(administered_items))
    selected = selector.select(make_context(item_bank, administered_items, responses, theta=0.0))

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
    item_bank = ItemBank.generate(30, rng=1)
    theta_true = 0.7
    rng = np.random.default_rng(5)
    administered_items = list(range(20))
    responses = [bool(rng.random() < irt.icc(theta_true, item_bank.items[idx])) for idx in administered_items]
    mle_theta = (
      NumericalSearchEstimator()
      .estimate(make_context(item_bank, administered_items, responses, theta=theta_true))
      .theta
    )

    mlwi = MLWISelector(n_nodes=81)
    max_info = MaxInfoSelector()

    mlwi_choice = mlwi.select(make_context(item_bank, administered_items, responses, theta=mle_theta))
    max_info_choice = max_info.select(make_context(item_bank, administered_items, theta=mle_theta))

    assert mlwi_choice == max_info_choice


class TestKLSelector:
  """Tests for KLSelector."""

  @pytest.mark.parametrize(
    ("kwargs", "match"),
    [
      pytest.param({"window": 0.0}, "window must be positive", id="zero_window"),
      pytest.param({"window": -1.0}, "window must be positive", id="negative_window"),
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
    p_hat, p = irt.icc(theta_hat, np.array(item)), irt.icc(theta, np.array(item))
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
      item_bank = ItemBank.generate(25, model="3PL", rng=int(rng.integers(1_000_000)))
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

      picked = KLSelector(window=2.0).select(
        make_context(item_bank, administered, theta=theta_hat, rng=np.random.default_rng(0))
      )

      assert reference[picked] == pytest.approx(reference.max(), rel=1e-6)

  @pytest.mark.parametrize(("administered", "expected"), [([2], 1), ([2, 3, 4, 5], 0)])
  def test_window_shrinks_with_the_square_root_of_the_test_length(self, administered: list[int], expected: int) -> None:
    """With window c = 2 the window is theta_hat +- 2 after one item and +- 1 after four.

    At theta_hat = 0, item 0 (a = 1, b = 0) and item 1 (a = 4, b = 1.5) trade places as the window narrows:
    their KL integrals are 0.610 and 0.869 over +- 2, 0.225 and 0.146 over +- 1.41, and 0.081 and 0.028
    over +- 1. A window of c / sqrt(n + 1) would already prefer item 0 after one item.
    """
    filler = [1.0, 3.0, 0.0, 1.0]
    item_bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [4.0, 1.5, 0.0, 1.0], filler, filler, filler, filler]))

    assert (
      KLSelector(window=2.0).select(make_context(item_bank, administered, theta=0.0, rng=np.random.default_rng(0)))
      == expected
    )

  def test_select_skips_items_at_the_exposure_cap(self) -> None:
    """The item with the most KL information is passed over while its exposure rate is at r_max."""
    item_bank = ItemBank(np.array([[1.0, -2.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 1.5, 0.0, 1.0]]))
    selector = KLSelector(window=0.5, r_max=0.5)

    uncapped = selector.select(make_context(item_bank, [], theta=0.0, rng=np.random.default_rng(0)))
    capped = selector.select(
      make_context(item_bank, [], theta=0.0, rng=np.random.default_rng(0), exposure_rates=np.array([0.0, 0.5, 0.0]))
    )

    assert (uncapped, capped) == (1, 2)

  def test_select_raises_when_exhausted(self) -> None:
    """An exhausted bank must raise NoItemsAvailableError, which ends the session, not score an empty array."""
    item_bank = ItemBank(np.array([[1.0, -2.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]]))

    with pytest.raises(NoItemsAvailableError, match="no more items"):
      KLSelector().select(make_context(item_bank, [0, 1, 2], theta=0.0, rng=np.random.default_rng(42)))


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
    clusters: tuple[int, ...] = CLUSTERS,
  ) -> int | None:
    item_bank = _two_pl_bank(list(self.DISCRIMINATIONS), [0.0] * len(self.DISCRIMINATIONS))
    selector = ClusterSelector(clusters=list(clusters), method=method, r_max=r_max, r_control=r_control)
    return selector.select(make_context(item_bank, administered, theta=0.0, exposure_rates=exposure_rates))

  @pytest.mark.parametrize(("method", "expected"), [("item_info", 1), ("cluster_info", 7), ("weighted_info", 13)])
  @pytest.mark.parametrize(
    "clusters", [CLUSTERS, tuple({0: 7, 1: 2**40, 2: 2**62}[label] for label in CLUSTERS)], ids=["small", "huge"]
  )
  def test_select_takes_the_best_item_of_the_cluster_each_method_prefers(
    self, method: str, expected: int, clusters: tuple[int, ...]
  ) -> None:
    """Each method ranks clusters differently; the pick is the most informative item of the chosen cluster.

    Labels need not be small: hashed or database IDs must pick the same items. Scoring arrays indexed by label value
    allocated one slot per value up to the largest label, which fails outright for 2**62.
    """
    assert self._select([], method=method, clusters=clusters) == expected

  def test_select_moves_on_once_a_cluster_is_fully_administered(self) -> None:
    """With cluster 0 used up, item_info turns to the cluster of the next most informative item."""
    assert self._select([0, 1, 2, 3]) == 13

  def test_item_info_scores_a_cluster_by_its_administered_items_too(self) -> None:
    """Item 0 is given, yet its cluster keeps its score, so item 1 (0.037) is picked over item 2 (1.0).

    This is the published method (see the technique page). Scoring only the remaining items would pick item 2.
    """
    item_bank = ItemBank(np.array([[2.5, 0.0, 0.0, 1.0], [0.5, 3.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))

    assert ClusterSelector(clusters=[0, 0, 1]).select(make_context(item_bank, [0], theta=0.0)) == 1

  def test_select_raises_when_every_cluster_is_exhausted(self) -> None:
    """An exhausted bank raises NoItemsAvailableError, which the engine turns into a stop."""
    with pytest.raises(NoItemsAvailableError, match="no more items"):
      self._select(list(range(len(self.CLUSTERS))))

  @pytest.mark.parametrize("method", ["item_info", "cluster_info", "weighted_info"])
  @pytest.mark.parametrize("n_labels", [len(CLUSTERS) - 1, len(CLUSTERS) + 1])
  def test_select_rejects_labels_that_do_not_match_the_bank(self, method: str, n_labels: int) -> None:
    """Labels for another bank size raised IndexError, or with cluster_info silently skipped the unlabelled items."""
    item_bank = _two_pl_bank(list(self.DISCRIMINATIONS), [0.0] * len(self.DISCRIMINATIONS))
    selector = ClusterSelector(clusters=[*self.CLUSTERS, 2][:n_labels], method=method)

    with pytest.raises(ValueError, match=f"clusters holds {n_labels} labels, but the item bank has 14 items"):
      selector.select(make_context(item_bank, [], theta=0.0))

  def test_cluster_sums_and_means_match_the_hand_computed_values(self) -> None:
    """Each cluster's information sum and mean must be a^2 / 4 summed or averaged over its items, gaps in labels too.

    The three clusters are relabelled 0, 3 and 5, so labels 1, 2 and 4 hold no item and must not appear.
    """
    item_bank = _two_pl_bank(list(self.DISCRIMINATIONS), [0.0] * len(self.DISCRIMINATIONS))
    clusters = [{0: 0, 1: 3, 2: 5}[label] for label in self.CLUSTERS]
    sums = {0: (3 * 0.2**2 + 2.0**2) / 4, 3: (7 * 1.0**2 + 1.05**2) / 4, 5: (1.7**2 + 1.8**2) / 4}

    assert ClusterSelector.sum_cluster_infos(0.0, item_bank, clusters) == pytest.approx(sums, rel=1e-12)
    assert ClusterSelector.weighted_cluster_infos(0.0, item_bank, clusters) == pytest.approx(
      {0: sums[0] / 4, 3: sums[3] / 8, 5: sums[5] / 2}, rel=1e-12
    )

  @pytest.mark.parametrize(
    ("kwargs", "match"),
    [
      ({"clusters": []}, "cannot be empty"),
      ({"clusters": [0, -1]}, "non-negative integers"),
      ({"clusters": [0.0, 1.5]}, "non-negative integers"),
      ({"clusters": [0, 1], "method": "max_info"}, "not a valid cluster selection method"),
      ({"clusters": [0, 1], "r_control": "strict"}, "not a valid cap fallback"),
    ],
  )
  def test_constructor_rejects_invalid_arguments(self, kwargs: dict[str, object], match: str) -> None:
    """A bad label, method or fallback must fail at construction; float labels used to be accepted."""
    with pytest.raises(ValueError, match=match):
      ClusterSelector(**kwargs)

  def test_select_prefers_items_below_the_exposure_cap(self) -> None:
    """Inside the chosen cluster, an item at the cap is passed over for a less informative one below it."""
    exposure_rates = np.zeros(len(self.CLUSTERS))
    exposure_rates[13] = 0.9

    assert self._select([], exposure_rates, method="weighted_info", r_max=0.5) == 12

  def test_the_default_cap_of_1_holds_back_no_item(self) -> None:
    """r_max = 1 applies no cap inside the cluster, even to an item every earlier examinee received (rate 1).

    Fails if the strict ``rate < r_max`` test passes item 13 over for the less informative item 12.
    """
    exposure_rates = np.zeros(len(self.CLUSTERS))
    exposure_rates[13] = 1.0

    assert self._select([], exposure_rates, method="weighted_info") == 13

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

  def test_aggressive_control_breaks_exposure_ties_at_random(self) -> None:
    """Items tied for the smallest exposure rate are drawn uniformly (ADR-0008)."""
    item_bank = ItemBank.generate(12, rng=4)
    rates = np.full(12, 0.5)
    rates[[3, 7, 9]] = 0.2
    selector = ClusterSelector(clusters=[0] * 12, r_max=0.1, r_control="aggressive")

    picks = {
      selector.select(make_context(item_bank, [], theta=0.0, rng=np.random.default_rng(seed), exposure_rates=rates))
      for seed in range(40)
    }

    assert picks == {3, 7, 9}


class TestPositionWeight:
  """Tests for _position_weight, the schedule ProgressiveSelector and ProportionalSelector share."""

  @pytest.mark.parametrize(
    ("test_size", "acceleration", "position", "expected"),
    [
      pytest.param(5, 1.0, 1, 0.0, id="first_position"),
      pytest.param(5, 0.0, 3, 2 / 4, id="zero_is_linear"),
      pytest.param(5, 1.0, 3, (1 + 2) / (1 + 2 + 3 + 4), id="positive_delays"),
      pytest.param(5, 2.0, 4, (1 + 4 + 9) / (1 + 4 + 9 + 16), id="larger_delays_more"),
      pytest.param(5, -1.0, 2, 1 / (1 + 1 / 2 + 1 / 3 + 1 / 4), id="negative_hastens"),
      pytest.param(5, 1.0, 5, 1.0, id="last_position"),
      pytest.param(5, 1.0, 7, 1.0, id="clamped_past_test_size"),
      pytest.param(1, 1.0, 1, 1.0, id="one_item_test"),
    ],
  )
  def test_follows_barrada_et_al_equation_12(
    self, test_size: int, acceleration: float, position: int, expected: float
  ) -> None:
    """At position n of N the weight is the sum of k ** t for k < n over the same sum for k < N.

    The expected values are those sums worked by hand from Barrada et al.'s (2008) Equation 12, which catR's nextItem
    computes too. A power schedule, ((n - 1) / (N - 1)) ** t, would give 0.5, 0.5625 and 4 in the t = 1, 2 and -1
    rows. Past N the weight stays 1: the stopper, not the selector, ends the test, and a weight above 1 would give the
    random component a negative coefficient. A one-item test is at its last position from the start.
    """
    assert progressive._position_weight(position, test_size, acceleration) == pytest.approx(expected)

  @pytest.mark.parametrize("acceleration", [600, 1000])
  def test_large_acceleration_does_not_overflow(self, acceleration: int) -> None:
    """4.0 ** 600 overflows to inf, which would make the weight inf / inf, NaN, for an accepted finite acceleration.

    The expected value is the ratio of the sums in exact integer arithmetic.
    """
    expected = Fraction(sum(m**acceleration for m in range(1, 4)), sum(m**acceleration for m in range(1, 5)))

    assert progressive._position_weight(4, 5, float(acceleration)) == pytest.approx(float(expected), rel=1e-12)


class TestProgressiveSelector:
  """Tests for ProgressiveSelector."""

  @pytest.mark.parametrize(
    ("administered", "draws", "expected"),
    [
      pytest.param([], [0.1, 0.9, 0.2, 0.3], 1, id="first_position_follows_the_draws"),
      pytest.param([0, 2], [0.1, 0.99, 0.2, 0.0], 3, id="last_position_follows_the_information"),
    ],
  )
  def test_select_moves_from_the_draws_to_the_information(
    self, administered: list[int], draws: list[float], expected: int
  ) -> None:
    """The largest random draw wins at the first position, the most informative item at the last.

    Item 3 (a = 2, b = 0) is the most informative at theta = 0, with scaled information 1, against
    0.197 for item 1. A weight of 0.5 at either end would flip the pick: 0.548 for item 1 against 0.65
    for item 3 with the first draws, and 0.594 against 0.5 with the last ones.
    """
    selector = ProgressiveSelector(test_size=3)

    selected = selector.select(
      make_context(_progressive_item_bank(), administered, theta=0.0, rng=_DeterministicRNG(draws))
    )

    assert selected == expected

  def test_select_respects_exposure_cap(self) -> None:
    """Item 3, the most informative, is at the cap, so the last position, scored by information only, passes it over."""
    item_bank = _progressive_item_bank()
    selector = ProgressiveSelector(test_size=3, r_max=0.5)
    exposure_rates = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    selected = selector.select(
      make_context(
        item_bank, [0, 1], theta=0.0, rng=_DeterministicRNG([0.1, 0.2, 0.3, 0.4]), exposure_rates=exposure_rates
      )
    )

    assert selected == 2

  def test_random_component_is_scaled_to_available_items(self) -> None:
    """Both score components share the scale of the most informative available item.

    Item 0 dominates the bank's information at theta=0 but is already administered. Normalizing
    by the bank-wide maximum would shrink the remaining items' information toward 0 and let the
    random draw decide (item 2); normalizing by the available maximum keeps the halfway blend
    balanced, so the far more informative item 1 wins.
    """
    item_bank = ItemBank(
      np.array(
        [
          [3.0, 0.0, 0.0, 1.0],
          [1.0, 0.0, 0.0, 1.0],
          [0.5, 0.0, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    selector = ProgressiveSelector(test_size=3)

    selected = selector.select(make_context(item_bank, [0], theta=0.0, rng=_DeterministicRNG([0.9, 0.1, 0.4])))

    assert selected == 1


class TestProportionalSelector:
  """Tests for ProportionalSelector."""

  # Information of _progressive_item_bank's items at theta = 0, in closed form: L (1 - L) with L = 1 / (1 + e^b) for
  # the items with a = 1, and a^2 / 4 for item 3.
  INFO_AT_ZERO = np.array([np.e**3 / (1 + np.e**3) ** 2, np.e / (1 + np.e) ** 2, np.e / (1 + np.e) ** 2, 1.0])

  @pytest.mark.parametrize(
    ("test_size", "acceleration", "administered", "exponent"),
    [
      pytest.param(3, 0.0, [], 0.0, id="first_position_is_uniform"),
      pytest.param(4, 1.0, [0, 1], 4 * (1 + 2) / (1 + 2 + 3), id="positive_acceleration"),
      pytest.param(3, -1.0, [0], 3 * 1 / (1 + 1 / 2), id="negative_acceleration"),
      pytest.param(3, 0.0, [0, 1], 3.0, id="last_position_reaches_the_test_length"),
    ],
  )
  def test_select_draws_by_information_raised_to_the_scheduled_exponent(
    self, test_size: int, acceleration: float, administered: list[int], exponent: float
  ) -> None:
    """Each remaining item is drawn with probability proportional to its information raised to N times the weight.

    That exponent is Barrada et al.'s (2008) Equation 13, worked by hand here from the weight's sums: 0 at the first
    position and the test length N at the last. An exponent of 6 ((n - 1) / (N - 1)) ** t would raise the information
    to 4 in the positive_acceleration row, and requiring a positive acceleration would reject the other rows.
    """
    rng = _DeterministicRNG([0.0, 0.0, 0.0, 0.0])

    ProportionalSelector(test_size=test_size, acceleration=acceleration).select(
      make_context(_progressive_item_bank(), administered, theta=0.0, rng=rng)
    )

    ((candidates, probs),) = rng.choice_calls
    remaining_items = [item for item in range(4) if item not in administered]
    assert candidates.tolist() == remaining_items
    weights = self.INFO_AT_ZERO[remaining_items] ** exponent
    np.testing.assert_allclose(probs, weights / weights.sum())

  def test_large_exponent_does_not_underflow_to_uniform_sampling(self) -> None:
    """Far from every item, the information raised to the exponent underflows to 0 for every item.

    At theta = -400 the information of an item with a = 1 is about e^(-400 - b), and its square, the exponent at the
    last position of a two-item test, is below the smallest float. Unscaled weights would all be 0, and the draw would
    fall back to uniform sampling, or divide 0 by 0, where it should favor the most informative item.
    """
    rng = _DeterministicRNG([0.0, 0.0, 0.0, 0.0])

    ProportionalSelector(test_size=2).select(
      make_context(_two_pl_bank([1.0] * 4, [0.0, 1.0, 2.0, 3.0]), [3], theta=-400.0, rng=rng)
    )

    ((_, probs),) = rng.choice_calls
    # This far out the information ratios are e^-b to double precision, and the exponent is the test length, 2.
    weights = np.exp(-2.0 * np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(probs, weights / weights.sum())


@pytest.mark.parametrize("acceleration", [float("nan"), float("inf")])
@pytest.mark.parametrize("selector_class", [ProgressiveSelector, ProportionalSelector])
def test_scheduled_selectors_reject_non_finite_acceleration(
  selector_class: Callable[..., BaseSelector], acceleration: float
) -> None:
  """A NaN or infinite acceleration makes the weight NaN; an infinite one makes both of its sums infinite.

  Without the check the selector would construct, then fail at its first select() on NaN scores or probabilities.
  """
  with pytest.raises(ValueError, match="acceleration must be finite"):
    selector_class(test_size=10, acceleration=acceleration)


@pytest.mark.parametrize(
  "make_selector",
  [
    pytest.param(lambda: ProgressiveSelector(test_size=8), id="progressive"),
    pytest.param(lambda: ProportionalSelector(test_size=8), id="proportional"),
  ],
)
def test_randomized_selectors_repeat_with_the_same_seed(make_selector: Callable[[], BaseSelector]) -> None:
  """Two runs seeded alike choose the same items at every position, as contract 5 promises."""
  item_bank = ItemBank.generate(40, rng=5)

  def run(seed: int) -> list[int | None]:
    selector, rng, administered = make_selector(), np.random.default_rng(seed), []
    for _ in range(8):
      administered.append(selector.select(make_context(item_bank, administered, theta=0.3, rng=rng)))
    return administered

  assert run(11) == run(11)
  assert run(11) != run(12)


_N_ITEMS = 30
# Each factory builds its selector for a bank of n_items items.
SELECTOR_FACTORIES: dict[str, Callable[[int], BaseSelector]] = {
  "MaxInfoSelector": lambda _: MaxInfoSelector(),
  "UrrySelector": lambda _: UrrySelector(),
  "KLSelector": lambda _: KLSelector(),
  "IntervalInfoSelector": lambda _: IntervalInfoSelector(),
  "RandomSelector": lambda _: RandomSelector(),
  "RandomesqueSelector": lambda _: RandomesqueSelector(bin_size=5),
  "The54321Selector": lambda _: The54321Selector(),
  "LinearSelector": lambda n_items: LinearSelector(list(range(n_items))),
  "ProgressiveSelector": lambda _: ProgressiveSelector(test_size=10),
  "ProportionalSelector": lambda _: ProportionalSelector(test_size=10),
  "ClusterSelector": lambda n_items: ClusterSelector(clusters=[i % 3 for i in range(n_items)]),
  "AStratSelector": lambda _: AStratSelector(test_size=10),
  "AStratBBlockSelector": lambda _: AStratBBlockSelector(test_size=10),
  "MaxInfoStratSelector": lambda _: MaxInfoStratSelector(test_size=10),
  "MaxInfoBBlockSelector": lambda _: MaxInfoBBlockSelector(test_size=10),
  "MEISelector": lambda _: MEISelector(),
  "MLWISelector": lambda _: MLWISelector(),
}


@pytest.mark.parametrize("make_selector", SELECTOR_FACTORIES.values(), ids=SELECTOR_FACTORIES.keys())
def test_select_takes_a_minimal_context(make_selector: Callable[[int], BaseSelector]) -> None:
  """Given only an item bank and a generator, every selector returns the bank's one item as a builtin int.

  A selector that still took the old arguments, or that needs a field the minimal context leaves at its default (no
  responses, no exposure rates), fails here, and numpy integers must not leak into session state.
  """
  item_bank = ItemBank.generate(1, rng=3)

  item_id = make_selector(1).select(StepContext(item_bank=item_bank, rng=np.random.default_rng(0)))

  assert type(item_id) is int
  assert item_id == 0


@pytest.mark.parametrize("make_selector", SELECTOR_FACTORIES.values(), ids=SELECTOR_FACTORIES.keys())
def test_select_never_returns_an_excluded_item(make_selector: Callable[[int], BaseSelector]) -> None:
  """A selector that drew its candidates without leaving out excluded_items would pick its first pick again.

  Each round excludes the earlier picks and selects with the same seed, so a selector that ignored them would repeat
  the first one.
  """
  item_bank = ItemBank.generate(_N_ITEMS, rng=3)
  selector = make_selector(_N_ITEMS)
  excluded: list[int] = []

  for _ in range(3):
    context = make_context(item_bank, theta=0.3, excluded_items=excluded, rng=np.random.default_rng(0))
    excluded.append(selector.select(context))

  assert len(set(excluded)) == 3


def test_excluded_items_do_not_move_the_test_position() -> None:
  """Excluded items are not administered, so a selector that counted them would move on to a later stage of the test.

  The a-stratified selector's first stratum of a three-item test holds the three least discriminating of nine items;
  with two of them excluded, the third is the only item it may give first.
  """
  item_bank = ItemBank.generate(9, rng=0)
  first_stratum = np.argsort(item_bank.discrimination, kind="stable")[:3].tolist()

  item_id = AStratSelector(test_size=3).select(make_context(item_bank, excluded_items=first_stratum[:2]))

  assert item_id == first_stratum[2]


SELECTOR_REPRS = {
  "MaxInfoSelector": "MaxInfoSelector(r_max=0.5)",
  "UrrySelector": "UrrySelector()",
  "KLSelector": "KLSelector(window=2.0, r_max=0.5)",
  "IntervalInfoSelector": "IntervalInfoSelector(interval=0.5, confidence=0.9)",
  "RandomSelector": "RandomSelector()",
  "RandomesqueSelector": "RandomesqueSelector(bin_size=4)",
  "The54321Selector": "The54321Selector(initial_bin_size=3)",
  "LinearSelector": "LinearSelector(items=[3, 1, 2])",
  "ProgressiveSelector": "ProgressiveSelector(test_size=12, acceleration=2.0, r_max=0.5)",
  "ProportionalSelector": "ProportionalSelector(test_size=12, acceleration=-1.0, r_max=0.5)",
  "ClusterSelector": (
    "ClusterSelector(clusters=[0, 1, 0], method=ClusterMethod.CLUSTER_INFO, r_max=0.5, "
    "r_control=CapFallback.AGGRESSIVE)"
  ),
  "AStratSelector": "AStratSelector(test_size=12, n_strata=4)",
  "AStratBBlockSelector": "AStratBBlockSelector(test_size=12, n_strata=4)",
  "MaxInfoStratSelector": "MaxInfoStratSelector(test_size=12, n_strata=4)",
  "MaxInfoBBlockSelector": "MaxInfoBBlockSelector(test_size=12, n_strata=4)",
  "MEISelector": "MEISelector(estimator=RecordingEstimator(), r_max=0.5)",
  "MEISelector-default-estimator": (
    "MEISelector(estimator=NumericalSearchEstimator(tol=1e-06, bounds=(-6.0, 6.0)), r_max=1.0)"
  ),
  "MLWISelector": "MLWISelector(n_nodes=21, bounds=(-4.0, 4.0), r_max=0.5)",
}


@pytest.mark.parametrize("source", SELECTOR_REPRS.values(), ids=SELECTOR_REPRS.keys())
def test_repr_is_the_constructor_call(source: str) -> None:
  """A selector's repr names every constructor argument with its value, so evaluating it rebuilds the selector.

  Selectors used to print display names, some without their arguments (MaxInfo, Randomesque, the stratified
  selectors) and others under names their constructors do not take (s=, k=, Q=).
  """
  namespace = {
    **vars(importlib.import_module("catsim.selection")),
    "NumericalSearchEstimator": NumericalSearchEstimator,
    "RecordingEstimator": RecordingEstimator,
  }

  assert repr(eval(source, namespace)) == source


@pytest.mark.parametrize(
  ("numpy_source", "source"),
  [
    ("RandomesqueSelector(bin_size=np.int64(4))", "RandomesqueSelector(bin_size=4)"),
    ("The54321Selector(initial_bin_size=np.int64(3))", "The54321Selector(initial_bin_size=3)"),
    ("AStratSelector(test_size=np.int64(12), n_strata=np.int64(4))", "AStratSelector(test_size=12, n_strata=4)"),
    (
      "ProgressiveSelector(test_size=np.int64(12), acceleration=2.0, r_max=0.5)",
      "ProgressiveSelector(test_size=12, acceleration=2.0, r_max=0.5)",
    ),
  ],
)
def test_numpy_integer_arguments_are_stored_as_python_ints(numpy_source: str, source: str) -> None:
  """Integer arguments taken from a NumPy array are stored as Python ints, so the repr still evaluates.

  Fails if np.int64(4) is kept as given: the repr then reads bin_size=np.int64(4), naming a module that the
  catsim.selection namespace does not have.
  """
  namespace = {**vars(importlib.import_module("catsim.selection")), "np": np}

  assert repr(eval(numpy_source, namespace)) == source


@pytest.mark.parametrize(
  "build",
  [
    lambda: RandomesqueSelector(bin_size=2.5),
    lambda: AStratSelector(test_size=12.5),
    lambda: AStratSelector(test_size=12, n_strata=4.5),
  ],
  ids=["bin_size", "test_size", "n_strata"],
)
def test_integer_arguments_reject_fractions(build: Callable[[], BaseSelector]) -> None:
  """A fractional size is rejected at construction instead of being stored and used to slice or count items."""
  with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
    build()


def test_selector_tables_cover_every_concrete_selector() -> None:
  """Every concrete selector exported by catsim.selection runs through the context and repr tests above.

  A selector left out of SELECTOR_FACTORIES could reject a minimal context or select an excluded item, and one left
  out of SELECTOR_REPRS could hide an argument from its repr, without any test noticing.
  """
  exported = {
    name
    for name, cls in vars(importlib.import_module("catsim.selection")).items()
    if isinstance(cls, type) and issubclass(cls, BaseSelector) and not inspect.isabstract(cls)
  }

  assert exported == set(SELECTOR_FACTORIES)
  assert exported == {source.partition("(")[0] for source in SELECTOR_REPRS.values()}
