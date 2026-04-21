"""Property-based tests for LinearSelector, ClusterSelector, and stratified selectors."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite

from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  ClusterSelector,
  LinearSelector,
  MaxInfoBBlockSelector,
  MaxInfoStratSelector,
)
from tests.hypothesis_strategies import item_params_matrix

LINEAR_SETTINGS = settings(max_examples=40, deadline=None)
CLUSTER_SETTINGS = settings(max_examples=25, deadline=None)
STRAT_SETTINGS = settings(max_examples=20, deadline=None)


# ---------------------------------------------------------------------------
# Shared composite strategies
# ---------------------------------------------------------------------------


@composite
def bank_with_clusters(draw: DrawFn, min_items: int = 4, max_items: int = 16) -> tuple[ItemBank, list[int]]:
  """ItemBank paired with a valid cluster assignment covering all items."""
  params = draw(item_params_matrix(min_items=min_items, max_items=max_items))
  n = params.shape[0]
  n_clusters = draw(st.integers(min_value=1, max_value=max(1, n // 2)))
  clusters = draw(st.lists(st.integers(min_value=0, max_value=n_clusters - 1), min_size=n, max_size=n))
  return ItemBank(params, validate=True), clusters


@composite
def bank_with_test_size(
  draw: DrawFn,
  min_items: int = 6,
  max_items: int = 20,
) -> tuple[ItemBank, int]:
  """ItemBank paired with a valid test_size (≤ n_items)."""
  params = draw(item_params_matrix(min_items=min_items, max_items=max_items))
  n = params.shape[0]
  test_size = draw(st.integers(min_value=1, max_value=n))
  return ItemBank(params, validate=True), test_size


# ---------------------------------------------------------------------------
# LinearSelector
# ---------------------------------------------------------------------------


@given(
  params=item_params_matrix(min_items=4, max_items=18),
  administered_count=st.integers(min_value=0, max_value=15),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@LINEAR_SETTINGS
def test_linear_selector_returns_first_non_administered_in_order(
  params: np.ndarray,
  administered_count: int,
  seed: int,
) -> None:
  n = params.shape[0]
  assume(administered_count < n)
  indexes = list(range(n))
  bank = ItemBank(params, validate=True)
  rng = np.random.default_rng(seed)
  selector = LinearSelector(indexes)
  administered = indexes[:administered_count]
  item_id = selector.select(item_bank=bank, administered_items=administered, est_theta=0.0, rng=rng)
  assert item_id is not None
  assert isinstance(item_id, (int, np.integer))
  assert int(item_id) not in administered
  assert int(item_id) == administered_count


@given(
  params=item_params_matrix(min_items=2, max_items=12),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@LINEAR_SETTINGS
def test_linear_selector_raises_when_all_indexes_administered(
  params: np.ndarray,
  seed: int,
) -> None:
  n = params.shape[0]
  indexes = list(range(n))
  bank = ItemBank(params, validate=True)
  rng = np.random.default_rng(seed)
  selector = LinearSelector(indexes)
  import pytest

  with pytest.raises(NoItemsAvailableError):
    selector.select(item_bank=bank, administered_items=indexes, est_theta=0.0, rng=rng)


# ---------------------------------------------------------------------------
# ClusterSelector
# ---------------------------------------------------------------------------


@given(
  bank_clusters=bank_with_clusters(),
  method=st.sampled_from(["item_info", "cluster_info", "weighted_info"]),
  est_theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False),
  administered_count=st.integers(min_value=0, max_value=10),
)
@CLUSTER_SETTINGS
def test_cluster_selector_returns_valid_unseen_item(
  bank_clusters: tuple[ItemBank, list[int]],
  method: str,
  est_theta: float,
  administered_count: int,
) -> None:
  bank, clusters = bank_clusters
  n = bank.n_items
  assume(administered_count < n)
  administered = list(range(administered_count))
  selector = ClusterSelector(clusters=clusters, method=method)
  item_id = selector.select(
    item_bank=bank,
    administered_items=administered,
    est_theta=est_theta,
  )
  assert isinstance(item_id, (int, np.integer))
  assert 0 <= int(item_id) < n
  assert int(item_id) not in administered


@given(
  theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False),
  bank_clusters=bank_with_clusters(),
)
@CLUSTER_SETTINGS
def test_cluster_selector_sum_cluster_infos_keys_match_unique_clusters(
  theta: float,
  bank_clusters: tuple[ItemBank, list[int]],
) -> None:
  bank, clusters = bank_clusters
  infos = ClusterSelector.sum_cluster_infos(theta, bank, clusters)
  assert set(infos.keys()) == set(clusters)
  for v in infos.values():
    assert np.isfinite(v)
    assert v >= 0.0


@given(
  theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False),
  bank_clusters=bank_with_clusters(),
)
@CLUSTER_SETTINGS
def test_cluster_selector_weighted_le_sum_cluster_infos(
  theta: float,
  bank_clusters: tuple[ItemBank, list[int]],
) -> None:
  bank, clusters = bank_clusters
  sums = ClusterSelector.sum_cluster_infos(theta, bank, clusters)
  weighted = ClusterSelector.weighted_cluster_infos(theta, bank, clusters)
  assert set(sums.keys()) == set(weighted.keys())
  for k in sums:
    assert weighted[k] <= sums[k] + 1e-12


# ---------------------------------------------------------------------------
# Stratified selectors
# ---------------------------------------------------------------------------


def _strat_simulate_steps(selector_cls, bank: ItemBank, test_size: int, steps: int, theta: float) -> None:
  """Run `steps` sequential selections and assert each result is valid and unseen.

  Stratified selectors pair each call with the item returned by the previous call,
  so the only correct way to build `administered` is to accumulate results from the
  selector itself — passing arbitrary indices breaks stratum assumptions.
  """
  n = bank.n_items
  selector = selector_cls(test_size=test_size)
  administered: list[int] = []
  for _ in range(steps):
    item_id = selector.select(item_bank=bank, administered_items=administered, est_theta=theta)
    assert item_id is not None
    assert isinstance(item_id, (int, np.integer))
    assert 0 <= int(item_id) < n
    assert int(item_id) not in administered
    administered.append(int(item_id))


@given(
  bank_ts=bank_with_test_size(),
  steps=st.integers(min_value=1, max_value=8),
  theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False),
)
@STRAT_SETTINGS
def test_a_strat_selector_sequential_steps_are_valid(
  bank_ts: tuple[ItemBank, int],
  steps: int,
  theta: float,
) -> None:
  bank, test_size = bank_ts
  assume(steps <= test_size)
  _strat_simulate_steps(AStratSelector, bank, test_size, steps, theta)


@given(
  bank_ts=bank_with_test_size(),
  steps=st.integers(min_value=1, max_value=8),
  theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False),
)
@STRAT_SETTINGS
def test_a_strat_b_block_selector_sequential_steps_are_valid(
  bank_ts: tuple[ItemBank, int],
  steps: int,
  theta: float,
) -> None:
  bank, test_size = bank_ts
  assume(steps <= test_size)
  _strat_simulate_steps(AStratBBlockSelector, bank, test_size, steps, theta)


@given(
  bank_ts=bank_with_test_size(),
  steps=st.integers(min_value=1, max_value=8),
  theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False),
)
@STRAT_SETTINGS
def test_max_info_strat_selector_sequential_steps_are_valid(
  bank_ts: tuple[ItemBank, int],
  steps: int,
  theta: float,
) -> None:
  bank, test_size = bank_ts
  assume(steps <= test_size)
  _strat_simulate_steps(MaxInfoStratSelector, bank, test_size, steps, theta)


@given(
  bank_ts=bank_with_test_size(),
  steps=st.integers(min_value=1, max_value=8),
  theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False),
)
@STRAT_SETTINGS
def test_max_info_b_block_selector_sequential_steps_are_valid(
  bank_ts: tuple[ItemBank, int],
  steps: int,
  theta: float,
) -> None:
  bank, test_size = bank_ts
  assume(steps <= test_size)
  _strat_simulate_steps(MaxInfoBBlockSelector, bank, test_size, steps, theta)
