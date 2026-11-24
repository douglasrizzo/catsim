"""Tests for catsim.metrics module."""

from collections.abc import Callable

import numpy as np
import pytest
from numpy.typing import ArrayLike

from catsim import metrics

# Errors (predicted minus actual) of -1 and 7: bias (-1 + 7) / 2 = 3, MSE (1 + 49) / 2 = 25, RMSE 5.
ACTUAL = [1.0, -1.0]
PREDICTED = [0.0, 6.0]


class TestBias:
  """Tests for the bias() function."""

  def test_bias_is_mean_signed_error(self) -> None:
    """Errors -1 and 7 give a bias of (-1 + 7) / 2 = 3, a builtin float.

    Fails if bias subtracts the estimates from the true abilities (-3), averages the absolute errors
    (4) or sums them (6), or returns a numpy.float64 where the signature says float.
    """
    value = metrics.bias(ACTUAL, PREDICTED)
    assert value == 3.0
    assert type(value) is float


class TestMse:
  """Tests for the mse() function."""

  def test_mse_is_mean_squared_error(self) -> None:
    """Errors -1 and 7 give an MSE of (1 + 49) / 2 = 25, a builtin float.

    Fails if mse averages the signed errors (3) instead of their squares or takes the root (5), or
    returns a numpy.float64 where the signature says float.
    """
    value = metrics.mse(ACTUAL, PREDICTED)
    assert value == 25.0
    assert type(value) is float


class TestRmse:
  """Tests for the rmse() function."""

  def test_rmse_is_root_of_mean_squared_error(self) -> None:
    """Errors -1 and 7 give an RMSE of the root of 25, 5, a builtin float.

    Fails if rmse returns the mean squared error (25) or the mean absolute error (4), or a
    numpy.float64 where the signature says float.
    """
    value = metrics.rmse(ACTUAL, PREDICTED)
    assert value == 5.0
    assert type(value) is float


@pytest.mark.parametrize("metric", [metrics.bias, metrics.mse, metrics.rmse], ids=["bias", "mse", "rmse"])
def test_error_metrics_reject_vectors_of_different_sizes(metric: Callable[[ArrayLike, ArrayLike], float]) -> None:
  """Fails if a metric broadcasts a one-element vector against a longer one instead of raising."""
  with pytest.raises(ValueError, match="same size"):
    metric([1.0, 2.0, 3.0], [1.0])


class TestOverlapRate:
  """Tests for the overlap_rate() function."""

  @pytest.mark.parametrize(
    ("exposure_rates", "test_size", "expected"),
    [
      # Every item seen by 3 / 5 of the examinees: S_r^2 = 0, so T = n / M = 3 / 5.
      pytest.param([0.6] * 5, 3, 0.6, id="uniform"),
      # Every examinee takes items 0 and 1: mean 0.5, S_r^2 = 0.25, T = (4 / 2) * 0.25 + 2 / 4 = 1.
      pytest.param([1.0, 1.0, 0.0, 0.0], 2, 1.0, id="one-test-for-all"),
      # Half the examinees take items {0, 1}, half {0, 2}: two examinees share both items with
      # probability 1/2 and one item otherwise, so T = 1/2 * 1 + 1/2 * 1/2 = 0.75. From the rates:
      # mean 0.5, S_r^2 = (0.25 + 0 + 0 + 0.25) / 4 = 0.125, T = (4 / 2) * 0.125 + 2 / 4 = 0.75.
      pytest.param([1.0, 0.5, 0.5, 0.0], 2, 0.75, id="two-tests"),
    ],
  )
  def test_overlap_rate_is_expected_share_of_items(
    self, exposure_rates: list[float], test_size: int, expected: float
  ) -> None:
    """The overlap rate is the expected share of items two examinees' tests have in common.

    Fails if overlap_rate takes the sample variance of the exposure rates (M - 1 denominator) or
    swaps the bank size M and the test length n, or returns a numpy.float64 where the signature says
    float.
    """
    value = metrics.overlap_rate(np.array(exposure_rates), test_size)
    assert value == pytest.approx(expected)
    assert type(value) is float

  @pytest.mark.parametrize("exposure_rates", [[-0.1, 0.5, 0.5], [0.5, 1.5, 0.5]], ids=["negative", "above-one"])
  def test_overlap_rate_rejects_rates_outside_unit_interval(self, exposure_rates: list[float]) -> None:
    """Fails if overlap_rate accepts an exposure rate below 0 or above 1, which no proportion can be."""
    with pytest.raises(ValueError, match="between 0 and 1"):
      metrics.overlap_rate(np.array(exposure_rates), 2)

  @pytest.mark.parametrize("test_size", [0, -1])
  def test_overlap_rate_rejects_non_positive_test_size(self, test_size: int) -> None:
    """Fails if overlap_rate divides by a test size of zero or accepts a negative one."""
    with pytest.raises(ValueError, match="Test size must be positive"):
      metrics.overlap_rate(np.array([0.5, 0.5, 0.5]), test_size)

  def test_overlap_rate_rejects_test_longer_than_bank(self) -> None:
    """Fails if overlap_rate accepts a test longer than the bank, which no examinee can take."""
    with pytest.raises(ValueError, match="cannot be larger than bank size"):
      metrics.overlap_rate(np.array([0.5, 0.5, 0.5]), 10)
