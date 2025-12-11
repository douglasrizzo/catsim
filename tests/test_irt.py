"""Tests for catsim.irt module."""

import math

import numpy as np
import pytest

from catsim import irt


class TestNumParams:
  """Tests for the NumParams enum."""

  def test_num_params_values(self) -> None:
    """Test that NumParams enum has correct values."""
    assert irt.NumParams.PL1.value == 1
    assert irt.NumParams.PL2.value == 2
    assert irt.NumParams.PL3.value == 3
    assert irt.NumParams.PL4.value == 4


class TestIcc:
  """Tests for the icc() function (Item Characteristic Curve)."""

  def test_icc_at_difficulty(self) -> None:
    """Test ICC probability at theta = b (difficulty)."""
    # At theta = b, probability should be (c + d) / 2 for 4PL model
    # For 2PL (c=0, d=1), probability should be 0.5
    prob = irt.icc(theta=0.0, a=1.0, b=0.0, c=0.0, d=1.0)
    assert prob == pytest.approx(0.5)

  def test_icc_high_ability(self) -> None:
    """Test ICC probability for high ability approaches upper asymptote."""
    prob = irt.icc(theta=6.0, a=1.5, b=0.0, c=0.0, d=1.0)
    assert prob > 0.99

  def test_icc_low_ability(self) -> None:
    """Test ICC probability for low ability approaches lower asymptote."""
    prob = irt.icc(theta=-6.0, a=1.5, b=0.0, c=0.0, d=1.0)
    assert prob < 0.01

  def test_icc_with_guessing(self) -> None:
    """Test ICC with pseudo-guessing parameter."""
    # At very low ability, probability should approach c
    prob = irt.icc(theta=-10.0, a=1.0, b=0.0, c=0.2, d=1.0)
    assert prob == pytest.approx(0.2, abs=0.01)

  def test_icc_with_upper_asymptote(self) -> None:
    """Test ICC with upper asymptote parameter."""
    # At very high ability, probability should approach d
    prob = irt.icc(theta=10.0, a=1.0, b=0.0, c=0.0, d=0.9)
    assert prob == pytest.approx(0.9, abs=0.01)

  def test_icc_discrimination_effect(self) -> None:
    """Test that higher discrimination produces steeper curve."""
    # Higher discrimination should result in faster probability change
    prob_low_a = irt.icc(theta=0.5, a=0.5, b=0.0)
    prob_high_a = irt.icc(theta=0.5, a=2.0, b=0.0)
    # Both should be > 0.5, but high_a should be higher
    assert prob_high_a > prob_low_a

  def test_icc_bounds(self) -> None:
    """Test that ICC is always between c and d."""
    for theta in np.linspace(-4, 4, 20):
      prob = irt.icc(theta, a=1.0, b=0.0, c=0.1, d=0.9)
      assert 0.1 <= prob <= 0.9


class TestDetectModel:
  """Tests for the detect_model() function."""

  def test_detect_1pl_model(self) -> None:
    """Test detection of 1PL model (only b varies)."""
    items = np.array([
      [1.0, -1.0, 0.0, 1.0],
      [1.0, 0.0, 0.0, 1.0],
      [1.0, 1.0, 0.0, 1.0],
    ])
    assert irt.detect_model(items) == 1

  def test_detect_2pl_model(self) -> None:
    """Test detection of 2PL model (a and b vary)."""
    items = np.array([
      [0.8, -1.0, 0.0, 1.0],
      [1.2, 0.0, 0.0, 1.0],
      [1.5, 1.0, 0.0, 1.0],
    ])
    assert irt.detect_model(items) == 2

  def test_detect_3pl_model(self) -> None:
    """Test detection of 3PL model (a, b, c vary)."""
    items = np.array([
      [0.8, -1.0, 0.1, 1.0],
      [1.2, 0.0, 0.15, 1.0],
      [1.5, 1.0, 0.2, 1.0],
    ])
    assert irt.detect_model(items) == 3

  def test_detect_4pl_model(self) -> None:
    """Test detection of 4PL model (all parameters vary)."""
    items = np.array([
      [0.8, -1.0, 0.1, 0.95],
      [1.2, 0.0, 0.15, 0.98],
      [1.5, 1.0, 0.2, 0.99],
    ])
    assert irt.detect_model(items) == 4


class TestInf:
  """Tests for the inf() function (Item Information)."""

  def test_inf_at_difficulty(self) -> None:
    """Test item information is maximized near difficulty for 2PL."""
    # For 2PL, info is maximized at theta = b
    info_at_b = irt.inf(theta=0.0, a=1.0, b=0.0)
    info_away = irt.inf(theta=2.0, a=1.0, b=0.0)
    assert info_at_b > info_away

  def test_inf_discrimination_effect(self) -> None:
    """Test that higher discrimination produces more information."""
    info_low_a = irt.inf(theta=0.0, a=0.5, b=0.0)
    info_high_a = irt.inf(theta=0.0, a=2.0, b=0.0)
    assert info_high_a > info_low_a

  def test_inf_always_positive(self) -> None:
    """Test that information is always non-negative."""
    for theta in np.linspace(-4, 4, 20):
      info = irt.inf(theta, a=1.0, b=0.0)
      assert info >= 0


class TestTestInfo:
  """Tests for the test_info() function."""

  def test_test_info_sum_of_items(self) -> None:
    """Test that test info is sum of item information."""
    items = np.array([
      [1.0, 0.0, 0.0, 1.0],
      [1.2, 0.5, 0.0, 1.0],
    ])
    theta = 0.0
    total_info = irt.test_info(theta, items)
    individual_sum = sum(irt.inf(theta, *item) for item in items)
    assert total_info == pytest.approx(individual_sum)

  def test_test_info_increases_with_items(self) -> None:
    """Test that test info increases with more items."""
    item1 = np.array([[1.0, 0.0, 0.0, 1.0]])
    items2 = np.array([
      [1.0, 0.0, 0.0, 1.0],
      [1.0, 0.0, 0.0, 1.0],
    ])
    theta = 0.0
    assert irt.test_info(theta, items2) > irt.test_info(theta, item1)


class TestVar:
  """Tests for the var() function."""

  def test_var_inverse_of_info(self) -> None:
    """Test that variance is inverse of test information."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    theta = 0.0
    test_inf = irt.test_info(theta, items)
    variance = irt.var(theta, items)
    assert variance == pytest.approx(1 / test_inf)

  def test_var_with_test_inf_param(self) -> None:
    """Test variance when test_inf is provided directly."""
    test_inf = 2.5
    variance = irt.var(test_inf=test_inf)
    assert variance == pytest.approx(1 / test_inf)

  def test_var_missing_params_raises(self) -> None:
    """Test that missing parameters raise ValueError."""
    with pytest.raises(ValueError, match="Either theta and items or test_inf"):
      irt.var()

  def test_var_zero_info_returns_neg_inf(self) -> None:
    """Test that zero information returns negative infinity."""
    variance = irt.var(test_inf=0)
    assert variance == float("-inf")


class TestSee:
  """Tests for the see() function (Standard Error of Estimation)."""

  def test_see_sqrt_of_var(self) -> None:
    """Test that SEE is square root of variance."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    theta = 0.0
    variance = irt.var(theta, items)
    see = irt.see(theta, items)
    assert see == pytest.approx(math.sqrt(variance))

  def test_see_decreases_with_items(self) -> None:
    """Test that SEE decreases with more items."""
    items1 = np.array([[1.0, 0.0, 0.0, 1.0]])
    items5 = np.array([[1.0, 0.0, 0.0, 1.0]] * 5)
    theta = 0.0
    assert irt.see(theta, items5) < irt.see(theta, items1)


class TestConfidenceInterval:
  """Tests for the confidence_interval() function."""

  def test_confidence_interval_symmetric(self) -> None:
    """Test that CI is symmetric around theta."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0]])
    theta = 0.0
    lower, upper = irt.confidence_interval(theta, items, confidence=0.95)
    assert lower < theta < upper
    assert abs(theta - lower) == pytest.approx(abs(upper - theta))

  def test_confidence_interval_wider_with_higher_confidence(self) -> None:
    """Test that higher confidence produces wider intervals."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.5, 0.0, 1.0]])
    theta = 0.0
    lower_90, upper_90 = irt.confidence_interval(theta, items, confidence=0.90)
    lower_99, upper_99 = irt.confidence_interval(theta, items, confidence=0.99)
    width_90 = upper_90 - lower_90
    width_99 = upper_99 - lower_99
    assert width_99 > width_90

  def test_confidence_interval_invalid_confidence_raises(self) -> None:
    """Test that invalid confidence raises ValueError."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="between 0 and 1"):
      irt.confidence_interval(0.0, items, confidence=1.5)

  def test_confidence_interval_boundary_confidence_raises(self) -> None:
    """Test that confidence=0 or confidence=1 raises ValueError."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="between 0 and 1"):
      irt.confidence_interval(0.0, items, confidence=0.0)
    with pytest.raises(ValueError, match="between 0 and 1"):
      irt.confidence_interval(0.0, items, confidence=1.0)


class TestReliability:
  """Tests for the reliability() function."""

  def test_reliability_bounded(self) -> None:
    """Test that reliability is between 0 and 1."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]] * 20)
    theta = 0.0
    rel = irt.reliability(theta, items)
    assert 0 <= rel <= 1

  def test_reliability_increases_with_items(self) -> None:
    """Test that reliability increases with more items."""
    items1 = np.array([[1.0, 0.0, 0.0, 1.0]])
    items10 = np.array([[1.0, 0.0, 0.0, 1.0]] * 10)
    theta = 0.0
    assert irt.reliability(theta, items10) > irt.reliability(theta, items1)


class TestMaxInfo:
  """Tests for the max_info() function."""

  def test_max_info_at_difficulty(self) -> None:
    """Test that max info returns difficulty for 2PL."""
    # For 2PL model (c=0, d=1), max info is at theta = b
    theta_max = irt.max_info(a=1.0, b=0.5, c=0.0, d=1.0)
    assert theta_max == pytest.approx(0.5)

  def test_max_info_shifts_with_guessing(self) -> None:
    """Test that pseudo-guessing shifts max info location."""
    # With guessing parameter, max info shifts above b
    theta_no_guess = irt.max_info(a=1.0, b=0.0, c=0.0, d=1.0)
    theta_with_guess = irt.max_info(a=1.0, b=0.0, c=0.2, d=1.0)
    assert theta_with_guess > theta_no_guess


class TestNormalizeItemBank:
  """Tests for the normalize_item_bank() function."""

  def test_normalize_1pl(self) -> None:
    """Test normalization of 1PL (difficulty only)."""
    items = np.array([[0.0], [1.0]])  # Just difficulty
    normalized = irt.normalize_item_bank(items)
    assert normalized.shape == (2, 4)
    # Check a=1, b=original, c=0, d=1
    assert np.all(normalized[:, 0] == 1.0)  # a
    assert np.all(normalized[:, 2] == 0.0)  # c
    assert np.all(normalized[:, 3] == 1.0)  # d

  def test_normalize_2pl(self) -> None:
    """Test normalization of 2PL (a, b)."""
    items = np.array([[1.5, 0.0], [1.0, 1.0]])
    normalized = irt.normalize_item_bank(items)
    assert normalized.shape == (2, 4)
    assert np.all(normalized[:, 2] == 0.0)  # c
    assert np.all(normalized[:, 3] == 1.0)  # d

  def test_normalize_3pl(self) -> None:
    """Test normalization of 3PL (a, b, c)."""
    items = np.array([[1.5, 0.0, 0.2], [1.0, 1.0, 0.1]])
    normalized = irt.normalize_item_bank(items)
    assert normalized.shape == (2, 4)
    assert np.all(normalized[:, 3] == 1.0)  # d

  def test_normalize_4pl_unchanged(self) -> None:
    """Test that 4PL items are returned unchanged."""
    items = np.array([[1.5, 0.0, 0.2, 0.95], [1.0, 1.0, 0.1, 0.98]])
    normalized = irt.normalize_item_bank(items)
    assert np.array_equal(items, normalized)

  def test_normalize_single_item(self) -> None:
    """Test normalization of a single item (1D array)."""
    item = np.array([0.5])  # Single difficulty value
    normalized = irt.normalize_item_bank(item)
    assert normalized.shape == (1, 4)


class TestValidateItemBank:
  """Tests for the validate_item_bank() function."""

  def test_validate_valid_bank(self) -> None:
    """Test that valid item bank passes validation."""
    items = np.array([[1.0, 0.0, 0.1, 0.95], [1.2, 0.5, 0.15, 0.98]])
    # Should not raise
    irt.validate_item_bank(items, raise_err=True)

  def test_validate_not_numpy_array_raises(self) -> None:
    """Test that non-numpy array raises TypeError."""
    items = [[1.0, 0.0, 0.1, 0.95]]
    with pytest.raises(TypeError, match=r"not of type numpy\.ndarray"):
      irt.validate_item_bank(items, raise_err=True)  # type: ignore[arg-type]

  def test_validate_negative_discrimination_raises(self) -> None:
    """Test that negative discrimination raises ValueError."""
    items = np.array([[-0.5, 0.0, 0.1, 0.95]])
    with pytest.raises(ValueError, match="discrimination < 0"):
      irt.validate_item_bank(items, raise_err=True)

  def test_validate_negative_guessing_raises(self) -> None:
    """Test that negative guessing parameter raises ValueError."""
    items = np.array([[1.0, 0.0, -0.1, 0.95]])
    with pytest.raises(ValueError, match="pseudo-guessing < 0"):
      irt.validate_item_bank(items, raise_err=True)

  def test_validate_guessing_greater_than_one_raises(self) -> None:
    """Test that guessing > 1 raises ValueError."""
    items = np.array([[1.0, 0.0, 1.5, 0.95]])
    with pytest.raises(ValueError, match="pseudo-guessing > 1"):
      irt.validate_item_bank(items, raise_err=True)

  def test_validate_asymptote_greater_than_one_raises(self) -> None:
    """Test that upper asymptote > 1 raises ValueError."""
    items = np.array([[1.0, 0.0, 0.1, 1.5]])
    with pytest.raises(ValueError, match="upper asymptote > 1"):
      irt.validate_item_bank(items, raise_err=True)

  def test_validate_negative_asymptote_raises(self) -> None:
    """Test that negative upper asymptote raises ValueError."""
    items = np.array([[1.0, 0.0, 0.1, -0.1]])
    with pytest.raises(ValueError, match="upper asymptote < 0"):
      irt.validate_item_bank(items, raise_err=True)


class TestLogLikelihood:
  """Tests for the log_likelihood() function."""

  def test_log_likelihood_correct_responses(self) -> None:
    """Test log likelihood with all correct responses."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    responses = [True]
    theta = 2.0  # High ability
    ll = irt.log_likelihood(theta, responses, items)
    # High theta with correct response should give high log-likelihood
    assert ll < 0  # Log likelihood is always negative
    assert ll > -0.5  # Should be reasonably close to 0 for high probability

  def test_log_likelihood_incorrect_responses(self) -> None:
    """Test log likelihood with all incorrect responses."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    responses = [False]
    theta = 2.0  # High ability
    ll = irt.log_likelihood(theta, responses, items)
    # High theta with incorrect response should give low log-likelihood
    assert ll < -1  # Should be more negative

  def test_log_likelihood_increases_toward_optimal_theta(self) -> None:
    """Test that log likelihood peaks near true ability."""
    items = np.array([[1.0, -1.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    # Responses consistent with theta = 0
    responses = [True, False]  # Easy correct, hard incorrect
    ll_far_low = irt.log_likelihood(-3.0, responses, items)
    ll_near = irt.log_likelihood(0.0, responses, items)
    ll_far_high = irt.log_likelihood(3.0, responses, items)
    # Log likelihood should be highest near the middle
    assert ll_near > ll_far_low
    assert ll_near > ll_far_high


class TestThetaToScale:
  """Tests for the theta_to_scale() function."""

  def test_theta_to_scale_at_midpoint(self) -> None:
    """Test that theta=0 maps to midpoint of scale."""
    # Default: theta [-4, 4] -> scale [0, 100]
    # theta=0 should map to 50
    scaled = irt.theta_to_scale(0.0)
    assert scaled == pytest.approx(50.0)

  def test_theta_to_scale_at_boundaries(self) -> None:
    """Test transformation at theta boundaries."""
    # theta=-4 should map to 0, theta=4 should map to 100
    assert irt.theta_to_scale(-4.0) == pytest.approx(0.0)
    assert irt.theta_to_scale(4.0) == pytest.approx(100.0)

  def test_theta_to_scale_custom_scale(self) -> None:
    """Test transformation with custom scale range."""
    # SAT-like scale: 200-800
    scaled = irt.theta_to_scale(0.0, scale_min=200, scale_max=800)
    assert scaled == pytest.approx(500.0)

  def test_theta_to_scale_linear(self) -> None:
    """Test that transformation is linear."""
    # theta=2 should be 3/4 of the way from min to max
    scaled = irt.theta_to_scale(2.0)
    assert scaled == pytest.approx(75.0)


class TestScaleToTheta:
  """Tests for the scale_to_theta() function."""

  def test_scale_to_theta_at_midpoint(self) -> None:
    """Test that midpoint of scale maps to theta=0."""
    theta = irt.scale_to_theta(50.0)
    assert theta == pytest.approx(0.0)

  def test_scale_to_theta_at_boundaries(self) -> None:
    """Test transformation at scale boundaries."""
    assert irt.scale_to_theta(0.0) == pytest.approx(-4.0)
    assert irt.scale_to_theta(100.0) == pytest.approx(4.0)

  def test_scale_to_theta_roundtrip(self) -> None:
    """Test roundtrip conversion."""
    original_theta = 1.5
    scaled = irt.theta_to_scale(original_theta)
    recovered_theta = irt.scale_to_theta(scaled)
    assert recovered_theta == pytest.approx(original_theta)

  def test_scale_to_theta_custom_scale(self) -> None:
    """Test transformation with custom scale range."""
    # SAT-like scale: 200-800
    theta = irt.scale_to_theta(500.0, scale_min=200, scale_max=800)
    assert theta == pytest.approx(0.0)


class TestConstants:
  """Tests for module constants."""

  def test_theta_bounds(self) -> None:
    """Test that theta bounds are symmetric and sensible."""
    assert irt.THETA_MIN_TYPICAL == -irt.THETA_MAX_TYPICAL
    assert irt.THETA_MIN_EXTENDED == -irt.THETA_MAX_EXTENDED
    assert irt.THETA_MIN_EXTENDED < irt.THETA_MIN_TYPICAL
