"""Tests for the package-wide tie-breaking rule of item selectors (ADR-0008)."""

from collections import Counter
from collections.abc import Callable

import numpy as np
import pytest

from catsim.item_bank import IrtModel, ItemBank
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
from catsim.selection.base import best_pool, pick_best
from tests.contexts import make_context

_N_COPIES = 6
_ITEM = [1.4, 0.3, 0.15, 0.98]


def _identical_bank(n_items: int = _N_COPIES) -> ItemBank:
  """Return a bank in which every item is the same, so every candidate ties for every selector."""
  return ItemBank(np.tile(_ITEM, (n_items, 1)))


class TestPickBest:
  """Tests for pick_best."""

  def test_single_best_item_does_not_consume_random_numbers(self) -> None:
    """An untied pick must leave the generator untouched, so untied runs keep their random stream."""
    rng = np.random.default_rng(1)
    state = rng.bit_generator.state

    picked = pick_best(np.array([3, 7, 9]), np.array([0.2, 0.9, 0.4]), rng)

    assert picked == 7
    assert type(picked) is int
    assert rng.bit_generator.state == state

  def test_exact_ties_are_broken_uniformly(self) -> None:
    """Each of four tied items is picked about a quarter of the time."""
    rng = np.random.default_rng(2)
    candidates = np.array([2, 5, 8, 11, 14])
    scores = np.array([0.7, 0.7, 0.1, 0.7, 0.7])

    counts = Counter(pick_best(candidates, scores, rng) for _ in range(4000))

    assert set(counts) == {2, 5, 11, 14}
    assert all(abs(count / 4000 - 0.25) < 0.03 for count in counts.values())

  def test_same_seed_gives_the_same_pick(self) -> None:
    """Random tie-breaking stays reproducible under a seeded generator."""
    candidates, scores = np.arange(5), np.ones(5)

    picks = [pick_best(candidates, scores, np.random.default_rng(seed)) for seed in range(20)]
    again = [pick_best(candidates, scores, np.random.default_rng(seed)) for seed in range(20)]

    assert picks == again

  def test_untied_pick_draws_no_random_number(self) -> None:
    """Breaking ties with a draw on every pick, such as jittering the scores, would change seeded runs without ties."""
    rng = np.random.default_rng(0)
    state = rng.bit_generator.state

    assert pick_best(np.arange(3), np.array([1.0, 2.0, 0.5]), rng) == 1
    assert rng.bit_generator.state == state

  def test_nan_scores_raise(self) -> None:
    """A NaN score would make every comparison false; it should raise instead of picking silently."""
    with pytest.raises(ValueError, match="NaN"):
      pick_best(np.arange(3), np.array([1.0, np.nan, 0.5]), np.random.default_rng(0))


class TestBestPool:
  """Tests for best_pool."""

  def test_items_tied_at_the_cutoff_join_the_pool(self) -> None:
    """As in catR, a bin of two grows to hold every item tied with its second-best item."""
    candidates = np.array([10, 11, 12, 13, 14])
    scores = np.array([0.5, 0.9, 0.5, 0.2, 0.5])

    pool = best_pool(candidates, scores, 2)

    assert pool.tolist() == [11, 10, 12, 14]

  def test_pool_without_ties_is_the_top_items_best_first(self) -> None:
    """Without ties at the cutoff the pool is exactly the requested number of best items."""
    pool = best_pool(np.arange(5), np.array([0.1, 0.4, 0.3, 0.9, 0.2]), 3)

    assert pool.tolist() == [3, 1, 2]

  def test_pool_larger_than_the_candidates_holds_them_all(self) -> None:
    """Asking for more items than there are candidates returns every candidate."""
    pool = best_pool(np.array([4, 6]), np.array([0.3, 0.8]), 5)

    assert pool.tolist() == [6, 4]

  def test_pool_rejects_a_size_below_one(self) -> None:
    """A bin of zero items would otherwise silently become the whole bank."""
    with pytest.raises(ValueError, match="at least 1"):
      best_pool(np.arange(3), np.array([0.1, 0.2, 0.3]), 0)

  def test_pool_rejects_nan_scores(self) -> None:
    """A NaN score would empty the pool; it should raise instead."""
    with pytest.raises(ValueError, match="NaN"):
      best_pool(np.arange(3), np.array([0.1, np.nan, 0.3]), 3)


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
      make_context(item_bank, administered_items, response_vector, theta=0.3, rng=np.random.default_rng(seed))
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
      make_context(item_bank, administered_items, theta=0.3, rng=np.random.default_rng(seed))
    )
    for seed in range(60)
  }

  assert picks == set(range(_N_COPIES - 1, _N_COPIES + 3))


def test_54321_breaks_ties_at_random_once_the_bin_has_one_item() -> None:
  """From the fifth item on the bin holds the single best item, plus every item tied with it."""
  item_bank = _identical_bank(9)
  administered_items = [0, 1, 2, 3, 4]

  picks = {
    The54321Selector().select(make_context(item_bank, administered_items, theta=0.3, rng=np.random.default_rng(seed)))
    for seed in range(60)
  }

  assert picks == {5, 6, 7, 8}


@pytest.mark.parametrize("method", ["item_info", "cluster_info", "weighted_info"])
def test_cluster_selector_draws_tied_clusters_uniformly(method: str) -> None:
  """Three clusters holding the same two items tie under every method, so each is chosen about a third of the time.

  Before ADR-0008 reached the choice of cluster, the first tied cluster in sort order won every draw. Each share is
  binomial with p = 1/3 over 3000 draws, a standard deviation of 0.0086, so the 0.03 tolerance is 3.5 of them; a
  selector that favoured one cluster, or split the draws between two, would miss by at least 1/6.
  """
  item_bank = ItemBank(np.tile([_ITEM, [0.8, -1.0, 0.1, 1.0]], (3, 1)))
  selector = ClusterSelector(clusters=[0, 0, 1, 1, 2, 2], method=method)
  rng = np.random.default_rng(3)

  # inside each cluster the first item is the more informative, so the pick names the chosen cluster
  counts = Counter(selector.select(make_context(item_bank, [], theta=0.3, rng=rng)) for _ in range(3000))

  assert set(counts) == {0, 2, 4}
  assert all(abs(count / 3000 - 1 / 3) < 0.03 for count in counts.values())


def _old_max_info(item_bank: ItemBank, administered_items: list[int], est_theta: float, rates, r_max: float) -> int:
  """Return MaxInfoSelector's pick as implemented before ADR-0008."""
  if item_bank.model == IrtModel.PL1:
    ordered = list(np.abs(item_bank.difficulty - est_theta).argsort())
  else:
    ordered = list((-item_bank.information(est_theta)).argsort())
  valid = [idx for idx in ordered if idx not in administered_items]
  low = [idx for idx in valid if rates[idx] < r_max]
  return int(low[0] if low else valid[0])


def _old_bin(item_bank: ItemBank, administered_items: list[int], est_theta: float, size: int) -> list[int]:
  """Return the Randomesque / 5-4-3-2-1 bin, best first, as built before ADR-0008."""
  if item_bank.model == IrtModel.PL1:
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
      make_context(item_bank, administered_items, theta=est_theta, exposure_rates=rates)
    ) == _old_max_info(item_bank, administered_items, est_theta, rates, 0.3)

    closest = next(idx for idx in np.abs(item_bank.difficulty - est_theta).argsort() if idx not in administered_items)
    assert UrrySelector().select(make_context(item_bank, administered_items, theta=est_theta)) == closest

    old_randomesque = int(np.random.default_rng(seed).choice(_old_bin(item_bank, administered_items, est_theta, 4)))
    assert (
      RandomesqueSelector(bin_size=4).select(
        make_context(item_bank, administered_items, theta=est_theta, rng=np.random.default_rng(seed))
      )
      == old_randomesque
    )

    bin_size = max(1, 5 - len(administered_items))
    old_54321 = int(np.random.default_rng(seed).choice(_old_bin(item_bank, administered_items, est_theta, bin_size)))
    assert (
      The54321Selector().select(
        make_context(item_bank, administered_items, theta=est_theta, rng=np.random.default_rng(seed))
      )
      == old_54321
    )
