"""Tests for RandomSelector, RandomesqueSelector and The54321Selector."""

import numpy as np
import pytest

from catsim import ItemBank
from catsim.selection import RandomesqueSelector, RandomSelector, The54321Selector
from tests.contexts import make_context


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
