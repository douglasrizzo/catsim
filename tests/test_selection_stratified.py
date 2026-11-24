"""Tests for the stratified selectors."""

from itertools import pairwise

import numpy as np
import pytest

from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import AStratBBlockSelector, AStratSelector, MaxInfoBBlockSelector, MaxInfoStratSelector
from tests.contexts import make_context

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


def test_excluded_items_do_not_move_the_test_position() -> None:
  """Excluded items are not administered, so a selector that counted them would move on to a later stage of the test.

  The a-stratified selector's first stratum of a three-item test holds the three least discriminating of nine items;
  with two of them excluded, the third is the only item it may give first.
  """
  item_bank = ItemBank.generate(9, rng=0)
  first_stratum = np.argsort(item_bank.discrimination, kind="stable")[:3].tolist()

  item_id = AStratSelector(test_size=3).select(make_context(item_bank, excluded_items=first_stratum[:2]))

  assert item_id == first_stratum[2]
