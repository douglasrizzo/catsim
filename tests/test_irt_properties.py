"""Property-based tests for :mod:`catsim.irt`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from catsim import irt
from tests.hypothesis_strategies import item_params_matrix, pl4_item_row, scale_mapping

# Keep CI fast; properties are meant to stress invariants rather than sample size.
IRT_SETTINGS = settings(max_examples=30, deadline=None)


@given(theta=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False), row=pl4_item_row())
@IRT_SETTINGS
def test_icc_bounded_by_asymptotes(theta: float, row: tuple[float, float, float, float]) -> None:
  a, b, c, d = row
  p = irt.icc(theta, a, b, c, d)
  assert c <= p <= d


@given(
  theta1=st.floats(-4.0, 3.5, allow_nan=False, allow_infinity=False),
  delta=st.floats(0.05, 2.0, allow_nan=False, allow_infinity=False),
  row=pl4_item_row(),
)
@IRT_SETTINGS
def test_icc_non_decreasing_in_theta(theta1: float, delta: float, row: tuple[float, float, float, float]) -> None:
  a, b, c, d = row
  theta2 = theta1 + delta
  p1 = irt.icc(theta1, a, b, c, d)
  p2 = irt.icc(theta2, a, b, c, d)
  assert p1 <= p2 + 1e-12


@given(theta=st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False), items=item_params_matrix(max_items=10))
@IRT_SETTINGS
def test_icc_hpc_matches_scalar_icc(theta: float, items: np.ndarray) -> None:
  vec = irt.icc_hpc(theta, items)
  for i in range(items.shape[0]):
    assert vec[i] == pytest.approx(float(irt.icc(theta, *items[i])), rel=1e-9, abs=1e-9)


@given(items=item_params_matrix(max_items=10))
@IRT_SETTINGS
def test_inf_non_negative_at_difficulty(items: np.ndarray) -> None:
  for i in range(items.shape[0]):
    b = float(items[i, 1])
    infv = irt.inf(b, *items[i])
    assert np.isfinite(infv)
    assert infv >= 0.0


@given(theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False), items=item_params_matrix(max_items=10))
@IRT_SETTINGS
def test_inf_hpc_non_negative(theta: float, items: np.ndarray) -> None:
  infos = irt.inf_hpc(theta, items)
  assert np.all(np.isfinite(infos))
  assert bool(np.all(infos >= 0.0))


@given(theta=st.floats(-3.5, 3.5, allow_nan=False, allow_infinity=False), items=item_params_matrix(max_items=10))
@IRT_SETTINGS
def test_test_info_matches_sum_of_item_information(theta: float, items: np.ndarray) -> None:
  expected = float(np.sum(irt.inf_hpc(theta, items)))
  assert irt.test_info(theta, items) == pytest.approx(expected, rel=1e-9, abs=1e-9)


@given(
  est_theta=st.floats(-2.5, 2.5, allow_nan=False, allow_infinity=False),
  items=item_params_matrix(min_items=1, max_items=6),
  responses=st.lists(st.booleans(), min_size=1, max_size=6),
)
@IRT_SETTINGS
def test_log_likelihood_finite_when_probabilities_interior(
  est_theta: float,
  items: np.ndarray,
  responses: list[bool],
) -> None:
  k = min(len(responses), items.shape[0])
  assume(k >= 1)
  administered = items[:k, :]
  response_vector = responses[:k]
  ps = irt.icc_hpc(est_theta, administered)
  assume(float(np.min(ps)) > 1e-8)
  assume(float(np.max(ps)) < 1.0 - 1e-8)
  ll = irt.log_likelihood(est_theta, response_vector, administered)
  assert np.isfinite(ll)
  assert irt.negative_log_likelihood(est_theta, response_vector, administered) == pytest.approx(-ll)


@given(m=scale_mapping())
@IRT_SETTINGS
def test_theta_scale_round_trip_identity(m: tuple[float, float, float, float]) -> None:
  theta_min, theta_max, scale_min, scale_max = m
  theta = np.linspace(theta_min, theta_max, num=5)
  scores = irt.theta_to_scale(theta, scale_min, scale_max, theta_min, theta_max)
  back = irt.scale_to_theta(scores, scale_min, scale_max, theta_min, theta_max)
  assert np.allclose(back, theta, rtol=1e-9, atol=1e-9)


@given(items=item_params_matrix())
@IRT_SETTINGS
def test_normalize_item_bank_idempotent_for_four_columns(items: np.ndarray) -> None:
  once = irt.normalize_item_bank(items)
  twice = irt.normalize_item_bank(once)
  assert np.array_equal(once, twice)


@given(
  st.lists(st.floats(-2.5, 2.5, allow_nan=False, allow_infinity=False), min_size=1, max_size=14).map(
    lambda xs: np.asarray(xs, dtype=np.float64).reshape(-1, 1)
  )
)
@IRT_SETTINGS
def test_normalize_one_column_is_rasch_expansion(difficulties: np.ndarray) -> None:
  out = irt.normalize_item_bank(difficulties)
  assert out.shape == (difficulties.shape[0], 4)
  assert np.allclose(out[:, 0], 1.0)
  assert np.allclose(out[:, 2], 0.0)
  assert np.allclose(out[:, 3], 1.0)
  assert np.allclose(out[:, 1].ravel(), difficulties.ravel())


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
@IRT_SETTINGS
def test_normalize_two_column_is_two_pl_expansion(two_pl: np.ndarray) -> None:
  out = irt.normalize_item_bank(two_pl)
  assert out.shape == (two_pl.shape[0], 4)
  assert np.allclose(out[:, :2], two_pl)
  assert np.allclose(out[:, 2], 0.0)
  assert np.allclose(out[:, 3], 1.0)


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
@IRT_SETTINGS
def test_normalize_three_column_is_three_pl_expansion(three_pl: np.ndarray) -> None:
  out = irt.normalize_item_bank(three_pl)
  assert out.shape == (three_pl.shape[0], 4)
  assert np.allclose(out[:, :3], three_pl)
  assert np.allclose(out[:, 3], 1.0)


@given(items=item_params_matrix())
@IRT_SETTINGS
def test_detect_model_returns_plausible_code(items: np.ndarray) -> None:
  model = irt.detect_model(items)
  assert model in {1, 2, 3, 4}


@given(items=item_params_matrix(max_items=8))
@IRT_SETTINGS
def test_max_info_hpc_row_matches_max_information_value(items: np.ndarray) -> None:
  peak_thetas = irt.max_info_hpc(items)
  peak_infos = irt.inf_hpc(peak_thetas, items)
  for i in range(items.shape[0]):
    theta_i = float(peak_thetas[i])
    grid = np.linspace(theta_i - 1.2, theta_i + 1.2, num=25)
    infos = irt.inf_hpc(grid, items[i : i + 1, :])
    assert float(peak_infos[i]) >= float(np.max(infos)) - 1e-6
