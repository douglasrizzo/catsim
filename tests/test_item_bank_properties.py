"""Property-based tests for :class:`catsim.item_bank.ItemBank`."""

from __future__ import annotations

from itertools import starmap

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from catsim import irt
from catsim.item_bank import ItemBank
from tests.hypothesis_strategies import item_params_matrix

BANK_SETTINGS = settings(max_examples=25, deadline=None)


@given(params=item_params_matrix())
@BANK_SETTINGS
def test_item_bank_model_matches_detect_model(params: np.ndarray) -> None:
  bank = ItemBank(params, validate=True)
  assert bank.model == irt.detect_model(params)


@given(params=item_params_matrix())
@BANK_SETTINGS
def test_max_info_values_are_each_items_information_at_its_own_peak(params: np.ndarray) -> None:
  """Item i's cached value is its information at max_info_thetas[i].

  Passing every peak to irt.information evaluates each peak against every item, an n x n matrix.
  """
  bank = ItemBank(params, validate=True)
  expected = list(starmap(irt.information, zip(irt.max_info_theta(params), params, strict=True)))

  np.testing.assert_allclose(bank.max_info_thetas, irt.max_info_theta(params), rtol=1e-12)
  np.testing.assert_allclose(bank.max_info_values, expected, rtol=1e-12)


@given(params=item_params_matrix())
@BANK_SETTINGS
def test_item_bank_stores_the_four_parameter_columns(params: np.ndarray) -> None:
  bank = ItemBank(params, validate=True)
  assert np.array_equal(bank.items, params)


@given(
  n=st.integers(min_value=12, max_value=80),
  seed=st.integers(min_value=0, max_value=999_999),
  itemtype=st.sampled_from(["1PL", "2PL", "3PL", "4PL"]),
)
@BANK_SETTINGS
def test_generate_item_bank_respects_count_and_validates(n: int, seed: int, itemtype: str) -> None:
  """Every model and seed gives n items that pass validation; an invalid draw (a <= 0, c >= d) would raise here."""
  bank = ItemBank.generate_item_bank(n, itemtype=itemtype, seed=seed, validate=True)
  assert bank.n_items == n


@given(
  params=item_params_matrix(min_items=2, max_items=12),
  theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
)
@BANK_SETTINGS
def test_test_information_matches_irt_helper(params: np.ndarray, theta: float) -> None:
  bank = ItemBank(params, validate=True)
  assert bank.test_information(theta) == pytest.approx(irt.total_information(theta, params), rel=1e-9, abs=1e-9)


@given(params=item_params_matrix(min_items=3, max_items=12), data=st.data())
@BANK_SETTINGS
def test_get_items_preserves_row_order(params: np.ndarray, data) -> None:
  n = params.shape[0]
  k = data.draw(st.integers(min_value=1, max_value=n))
  indices = list(range(k))
  bank = ItemBank(params, validate=True)
  subset = bank.get_items(indices)
  assert subset.shape == (len(indices), 4)
  for row, idx in enumerate(indices):
    assert np.allclose(subset[row], params[idx])
