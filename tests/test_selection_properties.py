"""Structural property-based tests for item selectors."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector, RandomSelector
from tests.hypothesis_strategies import item_params_matrix

SEL_SETTINGS = settings(max_examples=25, deadline=None)


@given(
  params=item_params_matrix(min_items=4, max_items=18),
  administered_count=st.integers(min_value=0, max_value=15),
  est_theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@SEL_SETTINGS
def test_max_info_selector_returns_valid_unseen_item(
  params: np.ndarray,
  administered_count: int,
  est_theta: float,
  seed: int,
) -> None:
  n = params.shape[0]
  assume(administered_count < n)
  administered = list(range(administered_count))
  bank = ItemBank(params, validate=True)
  rng = np.random.default_rng(seed)
  selector = MaxInfoSelector()
  item_id = selector.select(
    item_bank=bank,
    administered_items=administered,
    est_theta=est_theta,
    rng=rng,
  )
  assert isinstance(item_id, (int, np.integer))
  assert 0 <= int(item_id) < n
  assert item_id not in administered


@given(
  params=item_params_matrix(min_items=2, max_items=12),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@SEL_SETTINGS
def test_max_info_selector_raises_when_bank_exhausted(params: np.ndarray, seed: int) -> None:
  n = params.shape[0]
  bank = ItemBank(params, validate=True)
  rng = np.random.default_rng(seed)
  selector = MaxInfoSelector()
  with pytest.raises(NoItemsAvailableError):
    selector.select(
      item_bank=bank,
      administered_items=list(range(n)),
      est_theta=0.0,
      rng=rng,
    )


@given(
  params=item_params_matrix(min_items=4, max_items=18),
  administered_count=st.integers(min_value=0, max_value=15),
  est_theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@SEL_SETTINGS
def test_random_selector_returns_valid_unseen_item(
  params: np.ndarray,
  administered_count: int,
  est_theta: float,
  seed: int,
) -> None:
  n = params.shape[0]
  assume(administered_count < n)
  administered = list(range(administered_count))
  bank = ItemBank(params, validate=True)
  rng = np.random.default_rng(seed)
  selector = RandomSelector()
  item_id = selector.select(
    item_bank=bank,
    administered_items=administered,
    est_theta=est_theta,
    rng=rng,
  )
  assert isinstance(item_id, (int, np.integer))
  assert 0 <= int(item_id) < n
  assert item_id not in administered
