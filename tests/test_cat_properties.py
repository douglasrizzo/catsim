"""Property-based tests for :mod:`catsim.cat`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from catsim import cat

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
def test_mse_is_squared_bias_plus_error_variance(pairs: list[tuple[float, float]]) -> None:
  """The mean squared error splits into the squared bias and the variance of the errors."""
  actual, predicted = zip(*pairs, strict=True)
  errors = np.subtract(predicted, actual)

  assert cat.mse(actual, predicted) == pytest.approx(cat.bias(actual, predicted) ** 2 + np.var(errors), abs=1e-9)
  assert cat.rmse(actual, predicted) == pytest.approx(np.sqrt(cat.mse(actual, predicted)), abs=1e-9)
