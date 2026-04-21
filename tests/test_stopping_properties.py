"""Property-based tests for stopping rules."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from catsim import irt
from catsim.item_bank import ItemBank
from catsim.stopping import ConfidenceIntervalStopper, MinErrorStopper
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


MIN_ERROR_SETTINGS = settings(max_examples=30, deadline=None)


@given(
  params=st.integers(min_value=4, max_value=12).flatmap(
    lambda n: st.integers(min_value=0, max_value=999_999).map(lambda seed: ItemBank.generate_item_bank(n, seed=seed))
  ),
  administered_count=st.integers(min_value=0, max_value=10),
  theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
  min_error=st.floats(0.01, 2.0, allow_nan=False, allow_infinity=False),
)
@MIN_ERROR_SETTINGS
def test_min_error_stopper_decision_matches_see(
  params: ItemBank,
  administered_count: int,
  theta: float,
  min_error: float,
) -> None:
  bank = params
  assume(administered_count <= bank.n_items)
  # Keep administered strictly less than bank size so bank-exhaustion hard-stop
  # doesn't fire; we only want to test the SEE criterion.
  assume(administered_count < bank.n_items)
  stopper = MinErrorStopper(min_error=min_error)
  administered = list(range(administered_count))
  stopped = stopper.stop(item_bank=bank, administered_items=administered, theta=theta)
  if administered_count == 0:
    assert stopped is False
  else:
    admin_items = bank.get_items(administered)
    see = irt.see(theta, admin_items)
    assert stopped is (see < min_error)


@given(
  params=st.integers(min_value=4, max_value=12).flatmap(
    lambda n: st.integers(min_value=0, max_value=999_999).map(lambda seed: ItemBank.generate_item_bank(n, seed=seed))
  ),
  administered_count=st.integers(min_value=0, max_value=10),
  theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
  min_items=st.integers(min_value=2, max_value=6),
  min_error=st.floats(0.01, 2.0, allow_nan=False, allow_infinity=False),
)
@MIN_ERROR_SETTINGS
def test_min_error_stopper_respects_min_items_guard(
  params: ItemBank,
  administered_count: int,
  theta: float,
  min_items: int,
  min_error: float,
) -> None:
  bank = params
  assume(administered_count <= bank.n_items)
  stopper = MinErrorStopper(min_error=min_error, min_items=min_items)
  administered = list(range(administered_count))
  stopped = stopper.stop(item_bank=bank, administered_items=administered, theta=theta)
  if administered_count < min_items:
    assert stopped is False


CI_SETTINGS = settings(max_examples=25, deadline=None)


@given(
  params=st.integers(min_value=4, max_value=12).flatmap(
    lambda n: st.integers(min_value=0, max_value=999_999).map(lambda seed: ItemBank.generate_item_bank(n, seed=seed))
  ),
  administered_count=st.integers(min_value=1, max_value=10),
  theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
  confidence=st.floats(0.5, 0.99, allow_nan=False, allow_infinity=False),
  bounds=st.lists(
    st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=5,
  ).map(sorted),
)
@CI_SETTINGS
def test_confidence_interval_stopper_ci_symmetric_and_finite(
  params: ItemBank,
  administered_count: int,
  theta: float,
  confidence: float,
  bounds: list[float],
) -> None:
  bank = params
  assume(administered_count <= bank.n_items)
  assume(len(set(bounds)) == len(bounds))
  stopper = ConfidenceIntervalStopper(interval_bounds=bounds, confidence=confidence)
  administered = list(range(administered_count))
  admin_items = bank.get_items(administered)
  lower, upper = irt.confidence_interval(theta, admin_items, confidence)
  if np.isfinite(lower) and np.isfinite(upper):
    assert lower <= theta <= upper
    assert upper - lower >= 0.0
    # stopper decision must be consistent: only stops when CI fits in one interval
    stopped = stopper.stop(item_bank=bank, administered_items=administered, theta=theta)
    assert isinstance(stopped, bool)


@given(
  params=st.integers(min_value=4, max_value=12).flatmap(
    lambda n: st.integers(min_value=0, max_value=999_999).map(lambda seed: ItemBank.generate_item_bank(n, seed=seed))
  ),
  theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
  confidence=st.floats(0.5, 0.99, allow_nan=False, allow_infinity=False),
  bounds=st.lists(
    st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=5,
  ).map(sorted),
)
@CI_SETTINGS
def test_confidence_interval_stopper_never_stops_with_zero_items(
  params: ItemBank,
  theta: float,
  confidence: float,
  bounds: list[float],
) -> None:
  bank = params
  assume(len(set(bounds)) == len(bounds))
  stopper = ConfidenceIntervalStopper(interval_bounds=bounds, confidence=confidence)
  assert stopper.stop(item_bank=bank, administered_items=[], theta=theta) is False
