"""Property-based tests for ability estimation."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from catsim.estimation import NumericalSearchEstimator
from catsim.irt import THETA_MAX_EXTENDED, THETA_MIN_EXTENDED
from catsim.item_bank import ItemBank
from tests.hypothesis_strategies import item_params_matrix

EST_SETTINGS = settings(max_examples=15, deadline=None)


@given(
  params=item_params_matrix(min_items=8, max_items=18),
  k=st.integers(min_value=4, max_value=10),
  data=st.data(),
)
@EST_SETTINGS
def test_bounded_estimator_returns_finite_theta_in_extended_bounds(
  params: np.ndarray,
  k: int,
  data,
) -> None:
  n = params.shape[0]
  assume(k <= n)
  administered_items = list(range(k))
  response_vector = data.draw(st.lists(st.booleans(), min_size=k, max_size=k))
  assume(len(set(response_vector)) > 1)
  bank = ItemBank(params, validate=True)
  estimator = NumericalSearchEstimator(method="bounded", dodd=True)
  theta = estimator.estimate(
    item_bank=bank,
    administered_items=administered_items,
    response_vector=response_vector,
    est_theta=0.0,
  )
  assert isinstance(theta, float)
  assert np.isfinite(theta)
  assert THETA_MIN_EXTENDED <= theta <= THETA_MAX_EXTENDED
