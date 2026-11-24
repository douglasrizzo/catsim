"""Property-based tests for :mod:`catsim.metrics`."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from catsim import metrics

METRIC_SETTINGS = settings(max_examples=35)


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
@METRIC_SETTINGS
def test_mse_is_squared_bias_plus_error_variance(pairs: list[tuple[float, float]]) -> None:
  """The mean squared error splits into the squared bias and the variance of the errors.

  Fails if the mse is not the squared bias plus the error variance, or the rmse not the mse's root,
  on inputs the hand-worked cases in test_metrics.py do not cover. The bias enters squared, so its
  sign is checked there, not here.
  """
  actual, predicted = zip(*pairs, strict=True)
  errors = np.subtract(predicted, actual)

  assert metrics.mse(actual, predicted) == pytest.approx(
    metrics.bias(actual, predicted) ** 2 + np.var(errors), abs=1e-9
  )
  assert metrics.rmse(actual, predicted) == pytest.approx(np.sqrt(metrics.mse(actual, predicted)), abs=1e-9)
