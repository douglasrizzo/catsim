"""Tests for LinearSelector."""

import pytest

from catsim.item_bank import ItemBank
from catsim.selection import LinearSelector
from tests.contexts import make_context


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
