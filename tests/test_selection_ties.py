"""Tests for the package-wide tie-breaking rule of item selectors (ADR-0008)."""

from collections import Counter
from collections.abc import Callable

import numpy as np
import pytest

from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratSelector,
  BaseSelector,
  ClusterSelector,
  IntervalInfoSelector,
  KLSelector,
  MaxInfoSelector,
  MEISelector,
  MLWISelector,
  ProgressiveSelector,
  RandomesqueSelector,
  The54321Selector,
  UrrySelector,
)

_N_COPIES = 6
_ITEM = [1.4, 0.3, 0.15, 0.98]


def _identical_bank(n_items: int = _N_COPIES) -> ItemBank:
  """Return a bank in which every item is the same, so every candidate ties for every selector."""
  return ItemBank(np.tile(_ITEM, (n_items, 1)))


class TestPickBest:
  """Tests for BaseSelector._pick_best."""

  def test_single_best_item_does_not_consume_random_numbers(self) -> None:
    """An untied pick must leave the generator untouched, so untied runs keep their random stream."""
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state

    picked = BaseSelector._pick_best(np.array([3, 7, 9]), np.array([0.2, 0.9, 0.4]), rng)

    assert picked == 7
    assert type(picked) is int
    assert rng.bit_generator.state == state

  def test_exact_ties_are_broken_uniformly(self) -> None:
    """Each of four tied items is picked about a quarter of the time."""
    rng = np.random.default_rng(2)
    candidates = np.array([2, 5, 8, 11, 14])
    scores = np.array([0.7, 0.7, 0.1, 0.7, 0.7])

    counts = Counter(BaseSelector._pick_best(candidates, scores, rng) for _ in range(4000))

    assert set(counts) == {2, 5, 11, 14}
    assert all(abs(count / 4000 - 0.25) < 0.03 for count in counts.values())

  def test_same_seed_gives_the_same_pick(self) -> None:
    """Random tie-breaking stays reproducible under a seeded generator."""
    candidates, scores = np.arange(5), np.ones(5)

    picks = [BaseSelector._pick_best(candidates, scores, np.random.default_rng(seed)) for seed in range(20)]
    again = [BaseSelector._pick_best(candidates, scores, np.random.default_rng(seed)) for seed in range(20)]

    assert picks == again

  def test_tie_without_rng_raises(self) -> None:
    """A tie cannot be broken without a generator."""
    with pytest.raises(ValueError, match="rng parameter cannot be None"):
      BaseSelector._pick_best(np.arange(3), np.array([1.0, 1.0, 0.5]), None)

  def test_untied_pick_works_without_rng(self) -> None:
    """Deterministic selectors keep working without a generator when nothing ties."""
    assert BaseSelector._pick_best(np.arange(3), np.array([1.0, 2.0, 0.5]), None) == 1

  def test_nan_scores_raise(self) -> None:
    """A NaN score would make every comparison false; it should raise instead of picking silently."""
    with pytest.raises(ValueError, match="NaN"):
      BaseSelector._pick_best(np.arange(3), np.array([1.0, np.nan, 0.5]), np.random.default_rng(0))


class TestBestPool:
  """Tests for BaseSelector._best_pool."""

  def test_items_tied_at_the_cutoff_join_the_pool(self) -> None:
    """As in catR, a bin of two grows to hold every item tied with its second-best item."""
    candidates = np.array([10, 11, 12, 13, 14])
    scores = np.array([0.5, 0.9, 0.5, 0.2, 0.5])

    pool = BaseSelector._best_pool(candidates, scores, 2)

    assert pool.tolist() == [11, 10, 12, 14]

  def test_pool_without_ties_is_the_top_items_best_first(self) -> None:
    """Without ties at the cutoff the pool is exactly the requested number of best items."""
    pool = BaseSelector._best_pool(np.arange(5), np.array([0.1, 0.4, 0.3, 0.9, 0.2]), 3)

    assert pool.tolist() == [3, 1, 2]

  def test_pool_larger_than_the_candidates_holds_them_all(self) -> None:
    """Asking for more items than there are candidates returns every candidate."""
    pool = BaseSelector._best_pool(np.array([4, 6]), np.array([0.3, 0.8]), 5)

    assert pool.tolist() == [6, 4]

  def test_pool_rejects_a_size_below_one(self) -> None:
    """A bin of zero items would otherwise silently become the whole bank."""
    with pytest.raises(ValueError, match="at least 1"):
      BaseSelector._best_pool(np.arange(3), np.array([0.1, 0.2, 0.3]), 0)

  def test_pool_rejects_nan_scores(self) -> None:
    """A NaN score would empty the pool; it should raise instead."""
    with pytest.raises(ValueError, match="NaN"):
      BaseSelector._best_pool(np.arange(3), np.array([0.1, np.nan, 0.3]), 3)


SELECTOR_FACTORIES: dict[str, Callable[[], BaseSelector]] = {
  "MaxInfoSelector": MaxInfoSelector,
  "UrrySelector": UrrySelector,
  "IntervalInfoSelector": lambda: IntervalInfoSelector(interval=1.0),
  "AStratSelector": lambda: AStratSelector(test_size=_N_COPIES, n_strata=1),
  "ClusterSelector": lambda: ClusterSelector(clusters=[0] * _N_COPIES),
  "MEISelector": MEISelector,
  "MLWISelector": MLWISelector,
  "KLSelector": KLSelector,
  "RandomesqueSelector": lambda: RandomesqueSelector(bin_size=1),
}


@pytest.mark.parametrize("name", sorted(SELECTOR_FACTORIES))
def test_selectors_break_ties_at_random(name: str) -> None:
  """On a bank of identical items, the pick is a non-administered item and varies with the seed."""
  item_bank = _identical_bank()
  administered_items, response_vector = [0], [True]

  picks = {
    SELECTOR_FACTORIES[name]().select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=0.3,
      rng=np.random.default_rng(seed),
      response_vector=response_vector,
    )
    for seed in range(60)
  }

  assert picks == set(range(1, _N_COPIES))


def test_progressive_breaks_ties_at_random_once_fully_information_based() -> None:
  """At the last position the progressive weight is 1, so tied information is broken at random."""
  item_bank = _identical_bank(_N_COPIES + 3)
  administered_items = list(range(_N_COPIES - 1))

  picks = {
    ProgressiveSelector(test_size=_N_COPIES).select(
      item_bank=item_bank, administered_items=administered_items, est_theta=0.3, rng=np.random.default_rng(seed)
    )
    for seed in range(60)
  }

  assert picks == set(range(_N_COPIES - 1, _N_COPIES + 3))


def test_54321_breaks_ties_at_random_once_the_bin_has_one_item() -> None:
  """From the fifth item on the bin holds the single best item, plus every item tied with it."""
  item_bank = _identical_bank(9)
  administered_items = [0, 1, 2, 3, 4]

  picks = {
    The54321Selector().select(
      item_bank=item_bank, administered_items=administered_items, est_theta=0.3, rng=np.random.default_rng(seed)
    )
    for seed in range(60)
  }

  assert picks == {5, 6, 7, 8}


def test_tie_without_rng_raises_in_a_deterministic_selector() -> None:
  """A deterministic selector called without a generator raises only when items actually tie."""
  with pytest.raises(ValueError, match="rng parameter cannot be None"):
    MaxInfoSelector().select(item_bank=_identical_bank(), administered_items=[], est_theta=0.3)


def _old_max_info(item_bank: ItemBank, administered_items: list[int], est_theta: float, rates, r_max: float) -> int:
  """Return MaxInfoSelector's pick as implemented before ADR-0008."""
  if item_bank.model == 1:
    ordered = list(np.abs(item_bank.difficulty - est_theta).argsort())
  else:
    ordered = list((-item_bank.information(est_theta)).argsort())
  valid = [idx for idx in ordered if idx not in administered_items]
  low = [idx for idx in valid if rates[idx] < r_max]
  return int(low[0] if low else valid[0])


def _old_bin(item_bank: ItemBank, administered_items: list[int], est_theta: float, size: int) -> list[int]:
  """Return the Randomesque / 5-4-3-2-1 bin, best first, as built before ADR-0008."""
  if item_bank.model == 1:
    ordered = list(np.abs(item_bank.difficulty - est_theta).argsort())
  else:
    ordered = list((-item_bank.information(est_theta)).argsort())
  return [idx for idx in ordered if idx not in administered_items][:size]


@pytest.mark.parametrize("itemtype", ["1PL", "2PL", "3PL", "4PL"])
def test_untied_banks_give_the_same_picks_as_before(itemtype: str) -> None:
  """On banks without ties, the rewritten selectors choose exactly what they chose before ADR-0008."""
  rng = np.random.default_rng(11)
  for _ in range(40):
    item_bank = ItemBank.generate_item_bank(60, itemtype=itemtype, seed=int(rng.integers(1_000_000)))
    administered_items = [int(idx) for idx in rng.choice(60, int(rng.integers(0, 20)), replace=False)]
    est_theta = float(rng.normal())
    rates = rng.uniform(0.0, 0.6, 60)
    seed = int(rng.integers(1_000_000))

    assert MaxInfoSelector(r_max=0.3).select(
      item_bank=item_bank, administered_items=administered_items, est_theta=est_theta, exposure_rates=rates
    ) == _old_max_info(item_bank, administered_items, est_theta, rates, 0.3)

    closest = next(idx for idx in np.abs(item_bank.difficulty - est_theta).argsort() if idx not in administered_items)
    assert UrrySelector().select(item_bank=item_bank, administered_items=administered_items, est_theta=est_theta) == (
      closest
    )

    old_randomesque = int(np.random.default_rng(seed).choice(_old_bin(item_bank, administered_items, est_theta, 4)))
    assert (
      RandomesqueSelector(bin_size=4).select(
        item_bank=item_bank, administered_items=administered_items, est_theta=est_theta, rng=np.random.default_rng(seed)
      )
      == old_randomesque
    )

    bin_size = max(1, 5 - len(administered_items))
    old_54321 = int(np.random.default_rng(seed).choice(_old_bin(item_bank, administered_items, est_theta, bin_size)))
    assert (
      The54321Selector().select(
        item_bank=item_bank, administered_items=administered_items, est_theta=est_theta, rng=np.random.default_rng(seed)
      )
      == old_54321
    )
