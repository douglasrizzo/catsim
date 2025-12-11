"""Tests for catsim.cat module."""

import numpy as np
import pytest

from catsim import cat
from catsim.item_bank import ItemBank


class TestDodd:
  """Tests for the dodd() function."""

  def test_dodd_correct_response(self) -> None:
    """Test dodd heuristic when examinee answers correctly."""
    item_bank = ItemBank.generate_item_bank(10)
    theta = 0.0
    new_theta = cat.dodd(theta, item_bank, correct=True)
    # When correct, theta should increase toward max difficulty
    assert new_theta > theta

  def test_dodd_incorrect_response(self) -> None:
    """Test dodd heuristic when examinee answers incorrectly."""
    item_bank = ItemBank.generate_item_bank(10)
    theta = 0.0
    new_theta = cat.dodd(theta, item_bank, correct=False)
    # When incorrect, theta should decrease toward min difficulty
    assert new_theta < theta

  def test_dodd_correct_formula(self) -> None:
    """Test that dodd follows the correct formula."""
    item_bank = ItemBank.generate_item_bank(10)
    theta = 0.5
    b = item_bank.difficulty
    b_max = max(b)
    b_min = min(b)

    # Correct response
    expected_correct = theta + ((b_max - theta) / 2)
    assert cat.dodd(theta, item_bank, correct=True) == pytest.approx(expected_correct)

    # Incorrect response
    expected_incorrect = theta - ((theta - b_min) / 2)
    assert cat.dodd(theta, item_bank, correct=False) == pytest.approx(expected_incorrect)

  def test_dodd_at_max_difficulty(self) -> None:
    """Test dodd when theta equals max difficulty."""
    item_bank = ItemBank.generate_item_bank(10)
    b_max = max(item_bank.difficulty)
    # When theta = b_max and correct, result should equal theta (no movement)
    new_theta = cat.dodd(b_max, item_bank, correct=True)
    assert new_theta == pytest.approx(b_max)

  def test_dodd_at_min_difficulty(self) -> None:
    """Test dodd when theta equals min difficulty."""
    item_bank = ItemBank.generate_item_bank(10)
    b_min = min(item_bank.difficulty)
    # When theta = b_min and incorrect, result should equal theta (no movement)
    new_theta = cat.dodd(b_min, item_bank, correct=False)
    assert new_theta == pytest.approx(b_min)


class TestBias:
  """Tests for the bias() function."""

  def test_bias_zero(self) -> None:
    """Test that identical arrays produce zero bias."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 3.0]
    assert cat.bias(actual, predicted) == pytest.approx(0.0)

  def test_bias_positive(self) -> None:
    """Test positive bias (overestimation)."""
    actual = [0.0, 0.0, 0.0]
    predicted = [1.0, 1.0, 1.0]
    assert cat.bias(actual, predicted) == pytest.approx(1.0)

  def test_bias_negative(self) -> None:
    """Test negative bias (underestimation)."""
    actual = [1.0, 1.0, 1.0]
    predicted = [0.0, 0.0, 0.0]
    assert cat.bias(actual, predicted) == pytest.approx(-1.0)

  def test_bias_mixed(self) -> None:
    """Test bias with mixed over and underestimation."""
    actual = [0.0, 0.0, 0.0, 0.0]
    predicted = [1.0, -1.0, 2.0, -2.0]
    # Mean of differences: (1 + (-1) + 2 + (-2)) / 4 = 0
    assert cat.bias(actual, predicted) == pytest.approx(0.0)

  def test_bias_with_numpy_arrays(self) -> None:
    """Test bias with numpy arrays."""
    actual = np.array([1.0, 2.0, 3.0])
    predicted = np.array([1.5, 2.5, 3.5])
    assert cat.bias(actual, predicted) == pytest.approx(0.5)

  def test_bias_different_sizes_raises(self) -> None:
    """Test that different sized arrays raise ValueError."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0]
    with pytest.raises(ValueError, match="same size"):
      cat.bias(actual, predicted)


class TestMse:
  """Tests for the mse() function."""

  def test_mse_zero(self) -> None:
    """Test that identical arrays produce zero MSE."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 3.0]
    assert cat.mse(actual, predicted) == pytest.approx(0.0)

  def test_mse_positive(self) -> None:
    """Test MSE calculation with differences."""
    actual = [0.0, 0.0, 0.0]
    predicted = [1.0, 2.0, 3.0]
    # MSE = (1^2 + 2^2 + 3^2) / 3 = (1 + 4 + 9) / 3 = 14/3
    assert cat.mse(actual, predicted) == pytest.approx(14.0 / 3.0)

  def test_mse_negative_differences(self) -> None:
    """Test that negative differences are squared (always positive)."""
    actual = [1.0, 2.0, 3.0]
    predicted = [0.0, 0.0, 0.0]
    # Same MSE as test_mse_positive due to squaring
    assert cat.mse(actual, predicted) == pytest.approx(14.0 / 3.0)

  def test_mse_with_numpy_arrays(self) -> None:
    """Test MSE with numpy arrays."""
    actual = np.array([0.0, 0.0])
    predicted = np.array([1.0, 1.0])
    assert cat.mse(actual, predicted) == pytest.approx(1.0)

  def test_mse_different_sizes_raises(self) -> None:
    """Test that different sized arrays raise ValueError."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0]
    with pytest.raises(ValueError, match="same size"):
      cat.mse(actual, predicted)


class TestRmse:
  """Tests for the rmse() function."""

  def test_rmse_zero(self) -> None:
    """Test that identical arrays produce zero RMSE."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0, 3.0]
    assert cat.rmse(actual, predicted) == pytest.approx(0.0)

  def test_rmse_is_sqrt_of_mse(self) -> None:
    """Test that RMSE equals sqrt of MSE."""
    actual = [0.0, 0.0, 0.0]
    predicted = [1.0, 2.0, 3.0]
    mse_value = cat.mse(actual, predicted)
    rmse_value = cat.rmse(actual, predicted)
    assert rmse_value == pytest.approx(np.sqrt(mse_value))

  def test_rmse_unit_difference(self) -> None:
    """Test RMSE with unit differences."""
    actual = [0.0, 0.0]
    predicted = [1.0, 1.0]
    # RMSE = sqrt((1^2 + 1^2) / 2) = sqrt(1) = 1
    assert cat.rmse(actual, predicted) == pytest.approx(1.0)

  def test_rmse_different_sizes_raises(self) -> None:
    """Test that different sized arrays raise ValueError."""
    actual = [1.0, 2.0, 3.0]
    predicted = [1.0, 2.0]
    with pytest.raises(ValueError, match="same size"):
      cat.rmse(actual, predicted)


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
    assert cat.overlap_rate(exposure_rates, test_size) == pytest.approx(expected)

  def test_overlap_rate_with_variance(self) -> None:
    """Test overlap rate with non-uniform exposure."""
    exposure_rates = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    test_size = 2
    bank_size = 5
    var_r = np.var(exposure_rates)
    expected = (bank_size / test_size) * var_r + (test_size / bank_size)
    assert cat.overlap_rate(exposure_rates, test_size) == pytest.approx(expected)

  def test_overlap_rate_minimum(self) -> None:
    """Test that minimum overlap occurs with uniform exposure."""
    # Minimum overlap is Q/N when variance = 0
    exposure_rates = np.array([0.5] * 10)
    test_size = 5
    result = cat.overlap_rate(exposure_rates, test_size)
    assert result == pytest.approx(0.5)  # 5/10

  def test_overlap_rate_invalid_exposure_negative(self) -> None:
    """Test that negative exposure rates raise ValueError."""
    exposure_rates = np.array([-0.1, 0.5, 0.5])
    with pytest.raises(ValueError, match="between 0 and 1"):
      cat.overlap_rate(exposure_rates, 2)

  def test_overlap_rate_invalid_exposure_greater_than_one(self) -> None:
    """Test that exposure rates > 1 raise ValueError."""
    exposure_rates = np.array([0.5, 1.5, 0.5])
    with pytest.raises(ValueError, match="between 0 and 1"):
      cat.overlap_rate(exposure_rates, 2)

  def test_overlap_rate_invalid_test_size_zero(self) -> None:
    """Test that test_size=0 raises ValueError."""
    exposure_rates = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="Test size must be positive"):
      cat.overlap_rate(exposure_rates, 0)

  def test_overlap_rate_invalid_test_size_negative(self) -> None:
    """Test that negative test_size raises ValueError."""
    exposure_rates = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="Test size must be positive"):
      cat.overlap_rate(exposure_rates, -1)

  def test_overlap_rate_test_size_larger_than_bank(self) -> None:
    """Test that test_size > bank_size raises ValueError."""
    exposure_rates = np.array([0.5, 0.5, 0.5])
    with pytest.raises(ValueError, match="cannot be larger than bank size"):
      cat.overlap_rate(exposure_rates, 10)


class TestRandomResponseVector:
  """Tests for the random_response_vector() function."""

  def test_random_response_vector_size(self) -> None:
    """Test that the vector has the correct size."""
    size = 10
    result = cat.random_response_vector(size)
    assert len(result) == size

  def test_random_response_vector_empty(self) -> None:
    """Test that size=0 returns empty list."""
    result = cat.random_response_vector(0)
    assert result == []

  def test_random_response_vector_contains_booleans(self) -> None:
    """Test that all elements are booleans."""
    result = cat.random_response_vector(20)
    assert all(isinstance(x, bool) for x in result)

  def test_random_response_vector_randomness(self) -> None:
    """Test that the function produces random results (statistically)."""
    # Generate a large sample and check that both True and False appear
    result = cat.random_response_vector(100)
    assert True in result
    assert False in result

  def test_random_response_vector_large_size(self) -> None:
    """Test with a larger size."""
    size = 1000
    result = cat.random_response_vector(size)
    assert len(result) == size
    # With 1000 samples, we should have roughly 50% True
    true_count = sum(result)
    # Allow for statistical variation (between 40% and 60%)
    assert 400 <= true_count <= 600
