"""Property-based tests for :mod:`catsim.cat`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from catsim import cat
from catsim.item_bank import ItemBank
from tests.hypothesis_strategies import item_params_matrix

CAT_SETTINGS = settings(max_examples=35, deadline=None)


@given(
  pairs=st.lists(
    st.tuples(
      st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
      st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=35,
  )
)
@CAT_SETTINGS
def test_bias_and_mse_well_defined_for_equal_length_pairs(pairs: list[tuple[float, float]]) -> None:
  actual, predicted = zip(*pairs, strict=True)
  assert isinstance(cat.bias(actual, predicted), float)
  m = cat.mse(actual, predicted)
  assert m >= 0.0
  assert cat.rmse(actual, predicted) == pytest.approx(np.sqrt(m))


@given(
  actual=st.lists(st.floats(-2.0, 2.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=20),
)
@CAT_SETTINGS
def test_mse_zero_for_identical_vectors(actual: list[float]) -> None:
  assert cat.mse(actual, list(actual)) == pytest.approx(0.0, abs=1e-12)


@given(
  theta=st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
  params=item_params_matrix(min_items=2, max_items=12),
  correct=st.booleans(),
)
@CAT_SETTINGS
def test_dodd_is_finite_for_nonempty_bank(theta: float, params: np.ndarray, correct: bool) -> None:
  bank = ItemBank(params, validate=True)
  out = cat.dodd(theta, bank, correct)
  assert np.isfinite(out)


@given(
  rates_and_test_size=st.lists(
    st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=25
  ).flatmap(lambda rates: st.tuples(st.just(rates), st.integers(min_value=1, max_value=len(rates)))),
)
@CAT_SETTINGS
def test_overlap_rate_finite_for_valid_inputs(rates_and_test_size: tuple[list[float], int]) -> None:
  # Draw test_size from the bank size instead of filtering with assume(), which discarded most
  # examples and tripped Hypothesis's filter_too_much health check.
  rates, test_size = rates_and_test_size
  arr = np.asarray(rates, dtype=np.float64)
  val = cat.overlap_rate(arr, test_size)
  assert np.isfinite(val)


@given(size=st.integers(min_value=0, max_value=40))
@CAT_SETTINGS
def test_random_response_vector_has_expected_length(size: int) -> None:
  vec = cat.random_response_vector(size)
  assert len(vec) == size
  assert all(isinstance(x, bool) for x in vec)
