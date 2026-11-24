"""Property-based tests for :mod:`catsim.irt`."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from catsim import irt
from tests.hypothesis_strategies import item_params_matrix, pl4_item_row, scale_mapping

# Keep CI fast; properties are meant to stress invariants rather than sample size.
IRT_SETTINGS = settings(max_examples=30, deadline=None)


_ABILITY = st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False)

_ITEM_FUNCTIONS = [
  pytest.param(irt.icc, id="icc"),
  pytest.param(irt.information, id="information"),
  pytest.param(lambda _theta, items: irt.max_info_theta(items), id="max_info_theta"),
  pytest.param(lambda theta, items: irt.info_integral(theta - 1.0, theta + 2.0, items), id="info_integral"),
]


@pytest.mark.parametrize("function", _ITEM_FUNCTIONS)
@given(theta=_ABILITY, items=item_params_matrix(max_items=10))
@IRT_SETTINGS
def test_a_matrix_gives_each_item_its_value_as_a_row(
  function: Callable[[float, np.ndarray], np.ndarray], theta: float, items: np.ndarray
) -> None:
  """Each item's value from a matrix equals its value as a row.

  Splitting the parameters with items[:, k] fails on a row, and items[k] reads a matrix's rows as parameters.
  """
  np.testing.assert_allclose(function(theta, items), [function(theta, row) for row in items], rtol=1e-12, atol=1e-15)


@pytest.mark.parametrize("function", [irt.icc, irt.information])
@given(thetas=st.lists(_ABILITY, min_size=1, max_size=6), items=item_params_matrix(max_items=10))
@IRT_SETTINGS
def test_k_abilities_give_one_row_of_item_values_per_ability(
  function: Callable[[float | np.ndarray, np.ndarray], np.ndarray], thetas: list[float], items: np.ndarray
) -> None:
  """k abilities and n items give a (k, n) array whose row j holds the values at thetas[j].

  Pairing ability j with item j element-wise gives another shape, and raises when k != n.
  """
  values = function(np.array(thetas), items)

  assert values.shape == (len(thetas), len(items))
  np.testing.assert_allclose(values, [function(theta, items) for theta in thetas], rtol=1e-12, atol=1e-15)


@given(
  theta1=st.floats(-4.0, 3.5, allow_nan=False, allow_infinity=False),
  delta=st.floats(0.05, 2.0, allow_nan=False, allow_infinity=False),
  row=pl4_item_row(),
)
@IRT_SETTINGS
def test_icc_non_decreasing_in_theta(theta1: float, delta: float, row: tuple[float, float, float, float]) -> None:
  """A higher ability never lowers the probability of a correct response; a sign error in theta - b reverses it."""
  item = np.array(row)

  assert irt.icc(theta1, item) <= irt.icc(theta1 + delta, item) + 1e-12


def _probability_formula(theta: float, responses: list[bool], items: np.ndarray) -> tuple[float, float]:
  """Return the sum(log P or log(1 - P)) log-likelihood and a bound on its rounding error."""
  correct = np.asarray(responses, dtype=bool)
  ps = irt.icc(theta, items)
  reference = float(np.sum(np.where(correct, np.log(ps), np.log(1 - ps))))
  # rounding P by a few eps moves the reference's log P or log(1 - P) by about eps / P or eps / (1 - P)
  tolerance = 1e-12 + 8 * np.finfo(float).eps * float(np.sum(1 / np.where(correct, ps, 1 - ps)))
  return reference, tolerance


@given(
  est_theta=st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
  items=item_params_matrix(min_items=1, max_items=10),
  responses=st.lists(st.booleans(), min_size=10, max_size=10),
)
@IRT_SETTINGS
def test_log_likelihood_matches_the_probability_formula(
  est_theta: float,
  items: np.ndarray,
  responses: list[bool],
) -> None:
  """The logit-based log-likelihood equals sum(log P or log(1 - P)) wherever that formula is accurate."""
  responses = responses[: items.shape[0]]
  reference, tolerance = _probability_formula(est_theta, responses, items)

  assert irt.log_likelihood(est_theta, responses, items) == pytest.approx(reference, abs=tolerance)


@given(
  thetas=st.lists(st.floats(-6.0, 6.0, allow_nan=False, allow_infinity=False), min_size=1, max_size=8),
  items=item_params_matrix(min_items=1, max_items=10),
  responses=st.lists(st.booleans(), min_size=10, max_size=10),
)
@IRT_SETTINGS
def test_log_likelihood_matches_the_probability_formula_at_each_theta(
  thetas: list[float],
  items: np.ndarray,
  responses: list[bool],
) -> None:
  """Each value returned for an array of abilities matches the reference formula at that ability."""
  responses = responses[: items.shape[0]]
  values = irt.log_likelihood(np.array(thetas), responses, items)

  assert values.shape == (len(thetas),)
  for theta, value in zip(thetas, values, strict=True):
    reference, tolerance = _probability_formula(theta, responses, items)
    assert value == pytest.approx(reference, abs=tolerance)


@given(m=scale_mapping())
@IRT_SETTINGS
def test_theta_scale_round_trip_identity(m: tuple[float, float, float, float]) -> None:
  theta_min, theta_max, scale_min, scale_max = m
  theta = np.linspace(theta_min, theta_max, num=5)
  scores = irt.theta_to_scale(theta, scale_min, scale_max, theta_min, theta_max)
  back = irt.scale_to_theta(scores, scale_min, scale_max, theta_min, theta_max)
  assert np.allclose(back, theta, rtol=1e-9, atol=1e-9)


@given(items=item_params_matrix(max_items=8))
@IRT_SETTINGS
def test_max_info_theta_is_the_peak_of_the_information(items: np.ndarray) -> None:
  """No ability within 1.2 of an item's max_info_theta gives it more information; a wrong closed form moves the peak."""
  for item, peak in zip(items, irt.max_info_theta(items), strict=True):
    grid = np.linspace(peak - 1.2, peak + 1.2, num=25)
    assert irt.information(peak, item) >= np.max(irt.information(grid, item)) - 1e-6
