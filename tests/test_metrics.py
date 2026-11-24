"""Tests for catsim.metrics module."""

import numpy as np
import pytest

from catsim import metrics


class TestBias:
  """Tests for the bias() function."""

  def test_bias_zero(self) -> None:
    """Test that identical arrays produce zero bias."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 3.0]
    assert metrics.bias(actual, predicted) == pytest.approx(0.0)

  def test_bias_positive(self) -> None:
    """Test positive bias (overestimation)."""
    actual = [0.0, 0.0, 0.0]
    predicted = [1.0, 1.0, 1.0]
    assert metrics.bias(actual, predicted) == pytest.approx(1.0)

  def test_bias_negative(self) -> None:
    """Test negative bias (underestimation)."""
    actual = [1.0, 1.0, 1.0]
    predicted = [0.0, 0.0, 0.0]
    assert metrics.bias(actual, predicted) == pytest.approx(-1.0)

  def test_bias_mixed(self) -> None:
    """Test bias with mixed over and underestimation."""
    actual = [0.0, 0.0, 0.0, 0.0]
    predicted = [1.0, -1.0, 2.0, -2.0]
    # Mean of differences: (1 + (-1) + 2 + (-2)) / 4 = 0
    assert metrics.bias(actual, predicted) == pytest.approx(0.0)

  def test_bias_with_numpy_arrays(self) -> None:
    """Test bias with numpy arrays."""
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.5, 2.5, 3.5])
    assert metrics.bias(actual, predicted) == pytest.approx(0.5)

  def test_bias_different_sizes_raises(self) -> None:
    """Test that different sized arrays raise ValueError."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0]
    with pytest.raises(ValueError, match="same size"):
      metrics.bias(actual, predicted)


class TestMse:
  """Tests for the mse() function."""

  def test_mse_zero(self) -> None:
    """Test that identical arrays produce zero MSE."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 3.0]
    assert metrics.mse(actual, predicted) == pytest.approx(0.0)

  def test_mse_positive(self) -> None:
    """Test MSE calculation with differences."""
    actual = [0.0, 0.0, 0.0]
    predicted = [1.0, 2.0, 3.0]
    # MSE = (1^2 + 2^2 + 3^2) / 3 = (1 + 4 + 9) / 3 = 14/3
    assert metrics.mse(actual, predicted) == pytest.approx(14.0 / 3.0)

  def test_mse_negative_differences(self) -> None:
    """Test that negative differences are squared (always positive)."""
    actual = [1.0, 2.0, 3.0]
    predicted = [0.0, 0.0, 0.0]
    # Same MSE as test_mse_positive due to squaring
    assert metrics.mse(actual, predicted) == pytest.approx(14.0 / 3.0)

  def test_mse_with_numpy_arrays(self) -> None:
    """Test MSE with numpy arrays."""
    actual = np.array([0.0, 0.0])
    predicted = np.array([1.0, 1.0])
    assert metrics.mse(actual, predicted) == pytest.approx(1.0)

  def test_mse_different_sizes_raises(self) -> None:
    """Test that different sized arrays raise ValueError."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0]
    with pytest.raises(ValueError, match="same size"):
      metrics.mse(actual, predicted)


class TestRmse:
  """Tests for the rmse() function."""

  def test_rmse_zero(self) -> None:
    """Test that identical arrays produce zero RMSE."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 3.0]
    assert metrics.rmse(actual, predicted) == pytest.approx(0.0)

  def test_rmse_is_sqrt_of_mse(self) -> None:
    """Test that RMSE equals sqrt of MSE."""
    actual = [0.0, 0.0, 0.0]
    predicted = [1.0, 2.0, 3.0]
    mse_value = metrics.mse(actual, predicted)
    rmse_value = metrics.rmse(actual, predicted)
    assert rmse_value == pytest.approx(np.sqrt(mse_value))

  def test_rmse_unit_difference(self) -> None:
    """Test RMSE with unit differences."""
    actual = [0.0, 0.0]
    predicted = [1.0, 1.0]
    # RMSE = sqrt((1^2 + 1^2) / 2) = sqrt(1) = 1
    assert metrics.rmse(actual, predicted) == pytest.approx(1.0)

  def test_rmse_different_sizes_raises(self) -> None:
    """Test that different sized arrays raise ValueError."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0]
    with pytest.raises(ValueError, match="same size"):
      metrics.rmse(actual, predicted)


class TestOverlapRate:
  """Tests for the overlap_rate() function."""

  def test_overlap_rate_uniform_exposure(self) -> None:
    """Test overlap rate with uniform exposure rates."""
    # All items exposed equally: variance = 0
    exposure_rates = np.array([0.3, 0.3, 0.3, 0.3, 0.3])
    test_size = 3
    bank_size = 5
    # T = (N/Q) * 0 + (Q/N) = Q/N = 3/5 = 0.6
    expected = test_size / bank_size
    assert metrics.overlap_rate(exposure_rates, test_size) == pytest.approx(expected)

  def test_overlap_rate_with_variance(self) -> None:
    """Test overlap rate with non-uniform exposure."""
    exposure_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    test_size = 2
    bank_size = 5
    var_r = np.var(exposure_rates)
    expected = (bank_size / test_size) * var_r + (test_size / bank_size)
    assert metrics.overlap_rate(exposure_rates, test_size) == pytest.approx(expected)

  def test_overlap_rate_minimum(self) -> None:
    """Test that minimum overlap occurs with uniform exposure."""
    # Minimum overlap is Q/N when variance = 0
    exposure_rates = np.array([0.5] * 10)
    test_size = 5
    result = metrics.overlap_rate(exposure_rates, test_size)
    assert result == pytest.approx(0.5)  # 5/10

  def test_overlap_rate_invalid_exposure_negative(self) -> None:
    """Test that negative exposure rates raise ValueError."""
    exposure_rates = np.array([-0.1, 0.5, 0.5])
    with pytest.raises(ValueError, match="between 0 and 1"):
      metrics.overlap_rate(exposure_rates, 2)

  def test_overlap_rate_invalid_exposure_greater_than_one(self) -> None:
    """Test that exposure rates > 1 raise ValueError."""
    exposure_rates = np.array([0.5, 1.5, 0.5])
    with pytest.raises(ValueError, match="between 0 and 1"):
      metrics.overlap_rate(exposure_rates, 2)

  def test_overlap_rate_invalid_test_size_zero(self) -> None:
    """Test that test_size=0 raises ValueError."""
    exposure_rates = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="Test size must be positive"):
      metrics.overlap_rate(exposure_rates, 0)

  def test_overlap_rate_invalid_test_size_negative(self) -> None:
    """Test that negative test_size raises ValueError."""
    exposure_rates = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="Test size must be positive"):
      metrics.overlap_rate(exposure_rates, -1)

  def test_overlap_rate_test_size_larger_than_bank(self) -> None:
    """Test that test_size > bank_size raises ValueError."""
    exposure_rates = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="cannot be larger than bank size"):
      metrics.overlap_rate(exposure_rates, 10)
