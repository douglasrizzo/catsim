"""Property-based tests for stopping rules."""

from __future__ import annotations

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from catsim.item_bank import ItemBank
from catsim.stopping import TestLengthStopper as LengthStopper

STOP_SETTINGS = settings(max_examples=40, deadline=None)


@given(
  max_items=st.integers(min_value=1, max_value=20),
  bank_size=st.integers(min_value=8, max_value=40),
  administered_count=st.integers(min_value=0, max_value=35),
)
@STOP_SETTINGS
def test_length_stopper_max_items_rule(
  max_items: int,
  bank_size: int,
  administered_count: int,
) -> None:
  assume(bank_size > max_items)
  assume(administered_count <= bank_size)
  bank = ItemBank.generate_item_bank(bank_size, seed=123)
  stopper = LengthStopper(max_items=max_items)
  administered = list(range(administered_count))
  stopped = stopper.stop(item_bank=bank, administered_items=administered, theta=0.0)
  expected = administered_count >= max_items or administered_count >= bank_size
  assert stopped is expected


@given(
  min_items=st.integers(min_value=2, max_value=8),
  max_items=st.integers(min_value=10, max_value=20),
  administered_count=st.integers(min_value=0, max_value=15),
)
@STOP_SETTINGS
def test_length_stopper_respects_min_items_before_other_rules(
  min_items: int,
  max_items: int,
  administered_count: int,
) -> None:
  assume(min_items < max_items)
  assume(administered_count < max_items)
  bank = ItemBank.generate_item_bank(30, seed=7)
  stopper = LengthStopper(min_items=min_items, max_items=max_items)
  administered = list(range(administered_count))
  stopped = stopper.stop(item_bank=bank, administered_items=administered, theta=0.0)
  if administered_count < min_items:
    assert stopped is False
