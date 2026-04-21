"""Property-based tests for :class:`catsim.item_bank.ItemBank`."""

from __future__ import annotations

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
def test_max_info_cache_matches_direct_irt(params: np.ndarray) -> None:
  bank = ItemBank(params, validate=True)
  expected_thetas = irt.max_info_hpc(params)
  expected_values = irt.inf_hpc(expected_thetas, params)
  assert np.allclose(bank.max_info_thetas, expected_thetas, rtol=1e-9, atol=1e-9)
  assert np.allclose(bank.max_info_values, expected_values, rtol=1e-9, atol=1e-9)


@given(params=item_params_matrix())
@BANK_SETTINGS
def test_item_bank_shape_and_exposure_column(params: np.ndarray) -> None:
  bank = ItemBank(params, validate=True)
  assert bank.items.shape == (params.shape[0], 5)
  assert np.allclose(bank.items[:, :4], params)
  assert np.allclose(bank.exposure_rates, 0.0)


@given(
  n=st.integers(min_value=12, max_value=80),
  seed=st.integers(min_value=0, max_value=999_999),
  itemtype=st.sampled_from(["1PL", "2PL", "3PL", "4PL"]),
)
@BANK_SETTINGS
def test_generate_item_bank_respects_count_and_validates(n: int, seed: int, itemtype: str) -> None:
  bank = ItemBank.generate_item_bank(n, itemtype=itemtype, seed=seed, validate=True)
  assert bank.n_items == n
  irt.validate_item_bank(bank.items[:, :4], raise_err=True)


@given(
  params=item_params_matrix(min_items=2, max_items=12),
  theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
)
@BANK_SETTINGS
def test_test_information_matches_irt_helper(params: np.ndarray, theta: float) -> None:
  bank = ItemBank(params, validate=True)
  assert bank.test_information(theta) == pytest.approx(irt.test_info(theta, params), rel=1e-9, abs=1e-9)


@given(params=item_params_matrix(min_items=3, max_items=12), data=st.data())
@BANK_SETTINGS
def test_get_items_preserves_row_order(params: np.ndarray, data) -> None:
  n = params.shape[0]
  k = data.draw(st.integers(min_value=1, max_value=n))
  indices = list(range(k))
  bank = ItemBank(params, validate=True)
  subset = bank.get_items(indices)
  assert subset.shape == (len(indices), 5)
  for row, idx in enumerate(indices):
    assert np.allclose(subset[row, :4], params[idx])
