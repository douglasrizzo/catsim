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

BANK_SETTINGS = settings(max_examples=25)


@given(params=item_params_matrix())
@BANK_SETTINGS
def test_max_info_values_are_each_items_information_at_its_own_peak(params: np.ndarray) -> None:
  """Item i's cached value is its information at max_info_thetas[i].

  Passing every peak to irt.information evaluates each peak against every item, an n x n matrix.
  """
  bank = ItemBank(params)
  expected = list(starmap(irt.information, zip(irt.max_info_theta(params), params, strict=True)))

  np.testing.assert_allclose(bank.max_info_thetas, irt.max_info_theta(params), rtol=1e-12)
  np.testing.assert_allclose(bank.max_info_values, expected, rtol=1e-12)


@given(params=item_params_matrix())
@BANK_SETTINGS
def test_item_bank_stores_the_four_parameter_columns(params: np.ndarray) -> None:
  """A four-column matrix is stored unchanged; completing or reordering columns it already has would change it."""
  np.testing.assert_array_equal(ItemBank(params).items, params)


@given(
  st.lists(st.floats(-2.5, 2.5, allow_nan=False, allow_infinity=False), min_size=1, max_size=14).map(
    lambda xs: np.asarray(xs, dtype=np.float64).reshape(-1, 1)
  )
)
@BANK_SETTINGS
def test_one_column_is_read_as_difficulties(difficulties: np.ndarray) -> None:
  """One column holds b, completed with a = 1, c = 0 and d = 1; reading it as the discrimination would not."""
  items = ItemBank(difficulties).items

  np.testing.assert_array_equal(items[:, 1], difficulties[:, 0])
  np.testing.assert_array_equal(items[:, [0, 2, 3]], np.tile([1.0, 0.0, 1.0], (len(difficulties), 1)))


@given(
  st.lists(
    st.tuples(
      st.floats(0.3, 2.5, allow_nan=False, allow_infinity=False),
      st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=12,
  ).map(lambda rows: np.asarray(rows, dtype=np.float64))
)
@BANK_SETTINGS
def test_two_columns_are_read_as_discrimination_and_difficulty(two_pl: np.ndarray) -> None:
  """Two columns hold a and b, completed with c = 0 and d = 1; filling the wrong columns would move them."""
  items = ItemBank(two_pl).items

  np.testing.assert_array_equal(items[:, :2], two_pl)
  np.testing.assert_array_equal(items[:, 2:], np.tile([0.0, 1.0], (len(two_pl), 1)))


@given(
  st.lists(
    st.tuples(
      st.floats(0.3, 2.5, allow_nan=False, allow_infinity=False),
      st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
      st.floats(0.0, 0.25, allow_nan=False, allow_infinity=False),
    ),
    min_size=1,
    max_size=12,
  ).map(lambda rows: np.asarray(rows, dtype=np.float64))
)
@BANK_SETTINGS
def test_three_columns_are_read_as_a_b_and_c(three_pl: np.ndarray) -> None:
  """Three columns hold a, b and c, completed with d = 1; completing them with d = 0 would make every item invalid."""
  items = ItemBank(three_pl).items

  np.testing.assert_array_equal(items[:, :3], three_pl)
  np.testing.assert_array_equal(items[:, 3], 1.0)


@given(
  n=st.integers(min_value=12, max_value=80),
  seed=st.integers(min_value=0, max_value=999_999),
  model=st.sampled_from(["1PL", "2PL", "3PL", "4PL"]),
)
@BANK_SETTINGS
def test_generate_gives_n_valid_items(n: int, seed: int, model: str) -> None:
  """Every model and seed gives n items that pass validation; an invalid draw (a <= 0, c >= d) would raise here."""
  bank = ItemBank.generate(n, model=model, rng=seed)
  assert bank.n_items == n


@given(
  params=item_params_matrix(min_items=2, max_items=12),
  theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
)
@BANK_SETTINGS
def test_total_information_sums_every_item_by_default(params: np.ndarray, theta: float) -> None:
  """Without item indices the test information sums the information of every item; a default subset leaves some out."""
  expected = sum(float(irt.information(theta, item)) for item in params)

  assert ItemBank(params).total_information(theta) == pytest.approx(expected, rel=1e-12)


@given(params=item_params_matrix(min_items=3, max_items=12), data=st.data())
@BANK_SETTINGS
def test_get_items_gives_one_row_per_index_in_order(params: np.ndarray, data: st.DataObject) -> None:
  """A tuple of indices, in any order, repeated or empty, gives row k = the parameters of item indices[k].

  Indexing the matrix with the tuple itself reads (1, 2) as one parameter and () as every item, and sorting or
  deduplicating the indices reorders or drops rows.
  """
  indices = tuple(data.draw(st.lists(st.integers(0, params.shape[0] - 1), max_size=8)))

  subset = ItemBank(params).get_items(indices)

  assert subset.shape == (len(indices), 4)
  for row, index in zip(subset, indices, strict=True):
    np.testing.assert_array_equal(row, params[index])
