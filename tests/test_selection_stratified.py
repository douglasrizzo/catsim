"""Tests for the stratified selectors."""

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import example, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from catsim import CatEngine, StopReason, simulate
from catsim.estimation import NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import AStratBBlockSelector, AStratSelector, MaxInfoBBlockSelector, MaxInfoStratSelector
from catsim.stopping import MaxLengthStopper
from tests.contexts import make_context

STRATIFIED_SELECTORS = [AStratSelector, AStratBBlockSelector, MaxInfoStratSelector, MaxInfoBBlockSelector]


@composite
def _sizes(draw: DrawFn) -> tuple[int, int, int]:
  """Draw (n_items, n_strata, test_size) with 1 <= n_strata <= test_size <= n_items <= 23.

  Half the draws put the bank and the test in the same multiple of n_strata, each with a remainder: a stage then
  needs one item more than the stratum it reads only if the two sizing rules disagree, the case a uniform draw rarely
  reaches.
  """
  if draw(st.booleans()):
    n_strata = draw(st.integers(min_value=2, max_value=6))
    quotient = draw(st.integers(min_value=1, max_value=3))
    test_remainder = draw(st.integers(min_value=0, max_value=n_strata - 1))
    bank_remainder = draw(st.integers(min_value=test_remainder, max_value=n_strata - 1))
    return quotient * n_strata + bank_remainder, n_strata, quotient * n_strata + test_remainder
  n_items = draw(st.integers(min_value=2, max_value=20))
  n_strata = draw(st.integers(min_value=1, max_value=n_items))
  return n_items, n_strata, draw(st.integers(min_value=n_strata, max_value=n_items))


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
    """Without blocking, the bank is split into contiguous strata of increasing a (or maximum information).

    The first n_items % n_strata strata hold one item more, as the first stages hold one position more: 103 items in
    5 strata give [21, 21, 21, 20, 20]. A split that spread the larger strata out, [21, 21, 20, 21, 20], fails.
    """
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

    The trailing block of 102 % 4 = 2 items fills strata 0 and 1, so the strata are 26, 26, 25 and 25 items long like
    the unblocked ones; spreading it over strata 0 and 2 would leave the second stage of a 102-item test, 26 positions
    long, one item short. Making each location block a stratum instead would move every test from the easiest block to
    the hardest, whatever the estimate.
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
      assert stratum_of[by_key].tolist() == list(range(len(block)))

  @pytest.mark.parametrize(("test_size", "n_strata", "stage_sizes"), [(30, 5, [6, 6, 6, 6, 6]), (10, 4, [3, 3, 2, 2])])
  def test_each_stage_administers_consecutive_items_from_its_stratum(
    self, test_size: int, n_strata: int, stage_sizes: list[int]
  ) -> None:
    """The stages take the test's positions in order, the first test_size % n_strata stages one position more.

    With 5 strata in a 30-item test (Barrada et al., 2006) each stage administers 6 items; 4 strata in a 10-item test
    give [3, 3, 2, 2]. Stages that spread the longer ones out, [3, 2, 3, 2], fail; on a 10-item bank, split
    [3, 3, 2, 2], they need a third item from a 2-item stratum and end a test of the whole bank at its eighth item.
    """
    item_bank = ItemBank.generate(250, model="3PL", rng=4)
    selector = AStratSelector(test_size, n_strata=n_strata)
    stratum_of = np.empty(item_bank.n_items, dtype=int)
    for index, stratum in enumerate(selector.strata(item_bank)):
      stratum_of[stratum] = index
    administered: list[int] = []
    for _ in range(test_size):
      administered.append(selector.select(make_context(item_bank, administered, theta=0.5)))

    assert stratum_of[administered].tolist() == np.repeat(np.arange(n_strata), stage_sizes).tolist()

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


def test_excluded_items_do_not_move_the_test_position() -> None:
  """Excluded items are not administered, so a selector that counted them would move on to a later stage of the test.

  The a-stratified selector's first stratum of a three-item test holds the three least discriminating of nine items;
  with two of them excluded, the third is the only item it may give first.
  """
  item_bank = ItemBank.generate(9, rng=0)
  first_stratum = np.argsort(item_bank.discrimination, kind="stable")[:3].tolist()

  item_id = AStratSelector(test_size=3).select(make_context(item_bank, excluded_items=first_stratum[:2]))

  assert item_id == first_stratum[2]


@pytest.mark.parametrize("selector_class", STRATIFIED_SELECTORS)
@given(
  sizes=_sizes(),
  true_theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@example(sizes=(10, 4, 10), true_theta=0.0, seed=0)
@example(sizes=(8, 5, 7), true_theta=0.0, seed=0)
@settings(max_examples=100, deadline=None)
def test_a_session_administers_test_size_items(
  selector_class: type, sizes: tuple[int, int, int], true_theta: float, seed: int
) -> None:
  """An engine session with n_strata <= test_size <= n_items administers test_size items, and the stopper ends it.

  A stage that needs more items than its stratum holds ends the session early with NO_ITEMS. Stages sized
  [3, 2, 3, 2] for a 10-item test, against a 10-item bank split [3, 3, 2, 2], did so at the eighth item; strata sized
  by that same spread-out rule, [2, 2, 1, 2, 1] for 8 items, did so at the fifth item of a 7-item test in 5 stages,
  [2, 1, 2, 1, 1].
  """
  n_items, n_strata, test_size = sizes
  engine = CatEngine(
    ItemBank.generate(n_items, rng=seed),
    FixedPointInitializer(0.0),
    selector_class(test_size=test_size, n_strata=n_strata),
    NumericalSearchEstimator(),
    MaxLengthStopper(test_size),
  )
  session = simulate(engine, true_thetas=[true_theta], rng=seed).sessions[0]

  assert session.stop_reason == StopReason.STOPPER
  assert len(session.administered_items) == test_size
