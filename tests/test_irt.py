"""Tests for catsim.irt module."""

import math
import re

import numpy as np
import pytest
from scipy.integrate import quad

from catsim import irt


class TestIcc:
  """Tests for icc()."""

  @pytest.mark.parametrize(
    ("theta", "item", "expected"),
    [
      pytest.param(0.4, [1.3, 0.4, 0.2, 0.9], 0.55, id="theta = b gives (c + d) / 2"),
      pytest.param(0.5, [2.0, 0.0, 0.0, 1.0], 1 / (1 + math.exp(-1.0)), id="a scales the logit"),
      pytest.param(1.0, [1.0, 2.0, 0.25, 1.0], 0.25 + 0.75 / (1 + math.exp(1.0)), id="guessing floor"),
    ],
  )
  def test_icc_matches_the_4pl_formula(self, theta: float, item: list[float], expected: float) -> None:
    """P = c + (d - c) / (1 + exp(-a (theta - b))) at hand-worked points.

    Scaling the logistic term by d instead of d - c, dropping a, or flipping the sign of theta - b changes one of them.
    """
    value = irt.icc(theta, np.array(item))

    assert np.shape(value) == ()
    assert value == pytest.approx(expected, rel=1e-12)

  def test_icc_reaches_its_asymptotes_without_overflowing(self) -> None:
    """Far from b the ICC returns its asymptotes; evaluating e^x with Python floats would raise OverflowError."""
    item = np.array([1.0, 0.0, 0.2, 0.9])

    np.testing.assert_allclose(irt.icc(np.array([-1000.0, 1000.0]), item), [0.2, 0.9], rtol=1e-12)


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


class TestInformation:
  """Tests for information()."""

  @pytest.mark.parametrize("a", [0.5, 1.0, 2.3])
  def test_2pl_information_at_the_difficulty_is_a_squared_over_four(self, a: float) -> None:
    """A 2PL item gives a^2 L (1 - L) = a^2 / 4 at theta = b; dropping the square of a or the 1 - L changes it."""
    assert irt.information(-0.7, np.array([a, -0.7, 0.0, 1.0])) == pytest.approx(a**2 / 4, rel=1e-12)

  @pytest.mark.parametrize("theta", [-2.0, 0.3, 1.5])
  def test_1pl_information_is_the_variance_of_the_response(self, theta: float) -> None:
    """A 1PL item's information is P (1 - P); 4PL ratios that do not cancel to 1 when c = 0 and d = 1 change it."""
    p = 1 / (1 + math.exp(-(theta - 0.4)))

    assert irt.information(theta, np.array([1.0, 0.4, 0.0, 1.0])) == pytest.approx(p * (1 - p), rel=1e-12)

  @pytest.mark.parametrize("x", [40.0, -40.0, 700.0, -1000.0])
  def test_information_is_accurate_far_from_difficulty(self, x: float) -> None:
    """Far from b the 2PL information a^2 L (1 - L) is tiny but finite.

    The textbook formula computes 1 - P after P has rounded to 1, so it divides 0 by 0 and returns NaN.
    """
    expected = math.exp(-abs(x)) / (1 + math.exp(-abs(x))) ** 2

    assert irt.information(x, np.array([1.0, 0.0, 0.0, 1.0])) == pytest.approx(expected, rel=1e-12, abs=0.0)

  def test_information_matches_the_textbook_formula(self) -> None:
    """Where the textbook 4PL formula is well conditioned, it agrees at every ability and item."""
    items = np.array([[1.4, -0.5, 0.2, 0.95], [0.8, 1.0, 0.0, 1.0], [2.0, 0.3, 0.1, 1.0]])
    thetas = np.linspace(-4.0, 4.0, 17)
    a, b, c, d = items.T
    p = c + (d - c) / (1 + np.exp(-a * (thetas[:, np.newaxis] - b)))
    expected = a**2 * (p - c) ** 2 * (d - p) ** 2 / ((d - c) ** 2 * p * (1 - p))

    np.testing.assert_allclose(irt.information(thetas, items), expected, rtol=1e-10)


class TestInfoIntegral:
  """Tests for the closed-form integral of item information."""

  def test_matches_numerical_integration(self) -> None:
    """The closed form agrees with quadrature for 4PL items over finite intervals."""
    rng = np.random.default_rng(0)
    items = np.column_stack([
      rng.uniform(0.5, 2.5, 30),
      rng.uniform(-3, 3, 30),
      rng.uniform(0, 0.35, 30),
      rng.uniform(0.85, 1.0, 30),
    ])
    items[::3, 2] = 0.0
    items[::4, 3] = 1.0
    expected = [quad(irt.information, -1.7, 2.4, args=(item,))[0] for item in items]

    np.testing.assert_allclose(irt.info_integral(-1.7, 2.4, items), expected, rtol=1e-9, atol=1e-12)

  def test_whole_scale_integral_of_2pl_item_is_its_discrimination(self) -> None:
    """For c = 0 and d = 1, the information integrates to a over the whole ability scale."""
    items = np.array([[0.7, -1.0, 0.0, 1.0], [1.3, 0.0, 0.0, 1.0], [2.2, 2.5, 0.0, 1.0]])

    np.testing.assert_allclose(irt.info_integral(-np.inf, np.inf, items), items[:, 0])


class TestTotalInformation:
  """Tests for total_information()."""

  def test_total_information_sums_over_the_items(self) -> None:
    """Two 2PL items at their difficulty give 1/4 + 4/4 = 1.25; keeping one item or summing another axis does not."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]])

    total = irt.total_information(0.0, items)

    assert type(total) is float
    assert total == pytest.approx(1.25, rel=1e-12)

  def test_total_information_gives_one_value_per_ability(self) -> None:
    """Two abilities give two totals; summing over the abilities as well as the items gives one number."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]])
    l1, l2 = 1 / (1 + math.exp(-1.0)), 1 / (1 + math.exp(-2.0))

    values = irt.total_information(np.array([0.0, 1.0]), items)

    np.testing.assert_allclose(values, [1.25, l1 * (1 - l1) + 4 * l2 * (1 - l2)], rtol=1e-12)


class TestVar:
  """Tests for the var() function."""

  def test_var_is_the_inverse_of_the_test_information(self) -> None:
    """A 2PL item with a = 1 at its difficulty has information 1/4, so the variance is 4; returning I gives 0.25."""
    assert irt.var(0.0, np.array([[1.0, 0.0, 0.0, 1.0]])) == pytest.approx(4.0, rel=1e-12)

  def test_var_zero_info_returns_inf(self) -> None:
    """With no information the variance is unbounded; dividing 1 by the information would raise ZeroDivisionError."""
    assert irt.var(0.0, np.empty((0, 4))) == math.inf


class TestSee:
  """Tests for the see() function (Standard Error of Estimation)."""

  def test_see_is_the_square_root_of_the_variance(self) -> None:
    """A 2PL item with a = 1 at its difficulty has variance 4, so the standard error is 2; the variance itself is 4."""
    assert irt.see(0.0, np.array([[1.0, 0.0, 0.0, 1.0]])) == pytest.approx(2.0, rel=1e-12)

  def test_see_zero_info_returns_inf(self) -> None:
    """Items without information leave the standard error unbounded."""
    assert irt.see(0.0, np.empty((0, 4))) == float("inf")


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

  def test_reliability_zero_info_is_negative_infinity(self) -> None:
    """Reliability 1 - 1 / I is unbounded below without information; it used to be +inf."""
    assert irt.reliability(0.0, np.empty((0, 4))) == float("-inf")


class TestMaxInfoTheta:
  """Tests for max_info_theta()."""

  def test_max_info_theta_of_a_2pl_item_is_its_difficulty(self) -> None:
    """A 2PL item is most informative at theta = b; any other value means the d = 1 branch is wrong."""
    assert irt.max_info_theta(np.array([1.0, 0.5, 0.0, 1.0])) == pytest.approx(0.5)

  def test_max_info_theta_shifts_with_guessing(self) -> None:
    """A 3PL item is most informative at b + log((1 + sqrt(1 + 8c)) / 2) / a (Birnbaum): 0.2671 for c = 0.2, not b."""
    assert irt.max_info_theta(np.array([1.0, 0.0, 0.2, 1.0])) == pytest.approx(0.26714189097421254, rel=1e-12)

  def test_max_info_theta_of_a_symmetric_4pl_item_is_its_difficulty(self) -> None:
    """With c + d = 1 the information is symmetric about b, so its peak is b; the d = 1 formula would give b + 0.30."""
    assert irt.max_info_theta(np.array([1.2, -0.5, 0.1, 0.9])) == pytest.approx(-0.5, abs=1e-12)

  def test_max_info_theta_result_is_not_a_view_of_the_items(self) -> None:
    """Writing into the result leaves the item matrix alone; returning the difficulty column for 2PL items would not."""
    items = np.array([[1.0, 0.5, 0.0, 1.0], [1.3, -0.2, 0.0, 1.0]])
    thetas = irt.max_info_theta(items)
    np.testing.assert_array_equal(thetas, [0.5, -0.2])

    thetas[0] = 99.0

    np.testing.assert_array_equal(items[:, 1], [0.5, -0.2])


class TestNormalizeItemBank:
  """Tests for the normalize_item_bank() function."""

  def test_normalize_1pl(self) -> None:
    """Test normalization of 1PL (difficulty only)."""
    items = np.array([[0.0], [1.0]])  # Just difficulty
    normalized = irt.normalize_item_bank(items)
    assert normalized.shape == (2, 4)
    # Check a=1, b=original, c=0, d=1
    assert np.allclose(normalized[:, 0], 1.0)  # a
    assert np.allclose(normalized[:, 2], 0.0)  # c
    assert np.allclose(normalized[:, 3], 1.0)  # d

  def test_normalize_2pl(self) -> None:
    """Test normalization of 2PL (a, b)."""
    items = np.array([[1.5, 0.0], [1.0, 1.0]])
    normalized = irt.normalize_item_bank(items)
    assert normalized.shape == (2, 4)
    assert np.allclose(normalized[:, 2], 0.0)  # c
    assert np.allclose(normalized[:, 3], 1.0)  # d

  def test_normalize_3pl(self) -> None:
    """Test normalization of 3PL (a, b, c)."""
    items = np.array([[1.5, 0.0, 0.2], [1.0, 1.0, 0.1]])
    normalized = irt.normalize_item_bank(items)
    assert normalized.shape == (2, 4)
    assert np.allclose(normalized[:, 3], 1.0)  # d

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

  def test_validate_accepts_parameters_on_the_edges_of_their_ranges(self) -> None:
    """A small positive a, c = 0 and d = 1 are valid; a strict comparison in the wrong place would reject them."""
    items = np.array([[1e-3, -5.0, 0.0, 1.0], [1.2, 0.5, 0.15, 0.98]])

    irt.validate_item_bank(items)

  def test_validate_not_numpy_array_raises(self) -> None:
    """A nested list is not an item matrix; without the type check it would fail later with an unrelated error."""
    items = [[1.0, 0.0, 0.1, 0.95]]
    with pytest.raises(TypeError, match=r"not of type numpy\.ndarray"):
      irt.validate_item_bank(items)  # type: ignore[arg-type]

  @pytest.mark.parametrize(
    "shape",
    [
      pytest.param((4,), id="one dimension"),
      pytest.param((3, 0), id="no columns"),
      pytest.param((2, 3), id="three columns"),
      pytest.param((1, 5), id="five columns"),
      pytest.param((2, 4, 1), id="three dimensions"),
    ],
  )
  def test_validate_rejects_a_matrix_that_is_not_n_by_4(self, shape: tuple[int, ...]) -> None:
    """A matrix that is not n x 4 raises ValueError naming its shape.

    Without the shape check, missing columns fail with an IndexError and a fifth column is ignored.
    """
    items = np.ones(shape)

    with pytest.raises(ValueError, match=re.escape(f"shape {shape}; it must be 2-D with exactly 4 columns")):
      irt.validate_item_bank(items)

  def test_validate_rejects_a_matrix_without_items(self) -> None:
    """A 0 x 4 matrix raises ValueError: with no rows, every parameter check passes vacuously."""
    with pytest.raises(ValueError, match="no items"):
      irt.validate_item_bank(np.empty((0, 4)))

  @pytest.mark.parametrize(
    ("item", "message"),
    [
      pytest.param([-0.5, 0.0, 0.1, 0.95], "discrimination <= 0", id="a < 0"),
      pytest.param([0.0, 0.0, 0.1, 0.95], "discrimination <= 0", id="a = 0"),
      pytest.param([1.0, 0.0, -0.1, 0.95], "pseudo-guessing < 0", id="c < 0"),
      pytest.param([1.0, 0.0, 1.5, 0.95], "pseudo-guessing > 1", id="c > 1"),
      pytest.param([1.0, 0.0, 0.1, 1.5], "upper asymptote > 1", id="d > 1"),
      pytest.param([1.0, 0.0, 0.1, -0.1], "upper asymptote < 0", id="d < 0"),
      pytest.param([1.0, 0.0, 0.3, 0.3], "pseudo-guessing >= upper asymptote", id="c = d"),
    ],
  )
  def test_validate_rejects_a_parameter_out_of_its_range(self, item: list[float], message: str) -> None:
    """Each range check raises ValueError naming the parameter; it used to print the message and return by default.

    a = 0 gives no information and no point of maximum information; c >= d makes the ICC flat or decreasing and the
    information formula divide by d - c.
    """
    items = np.array([[1.0, 0.0, 0.0, 1.0], item])

    with pytest.raises(ValueError, match=message):
      irt.validate_item_bank(items)

  @pytest.mark.parametrize(
    "item",
    [
      pytest.param([np.nan, 0.0, 0.0, 1.0], id="a = nan"),
      pytest.param([1.0, np.nan, 0.0, 1.0], id="b = nan"),
      pytest.param([1.0, 0.0, np.nan, 1.0], id="c = nan"),
      pytest.param([1.0, 0.0, 0.0, np.nan], id="d = nan"),
      pytest.param([np.inf, 0.0, 0.0, 1.0], id="a = inf"),
      pytest.param([-np.inf, 0.0, 0.0, 1.0], id="a = -inf"),
      pytest.param([1.0, np.inf, 0.0, 1.0], id="b = inf"),
      pytest.param([1.0, -np.inf, 0.0, 1.0], id="b = -inf"),
    ],
  )
  def test_validate_rejects_nan_and_infinite_parameters(self, item: list[float]) -> None:
    """NaN and infinite parameters raise ValueError.

    Every range check compares, and comparisons with NaN are false; b was never checked and a = inf passed, so such an
    item was accepted and turned the ICC, likelihoods and information sums it entered into NaN.
    """
    items = np.array([[1.0, 0.5, 0.1, 1.0], item])

    with pytest.raises(ValueError, match="NaN or infinite parameters"):
      irt.validate_item_bank(items)

  def test_validate_reports_every_violation_on_one_line(self) -> None:
    """All violations are listed, separated by "; "; the messages used to start with and be joined by newlines."""
    items = np.array([[-1.0, 0.0, -0.1, 1.0]])
    expected = "There are items with discrimination <= 0; There are items with pseudo-guessing < 0"

    with pytest.raises(ValueError, match=rf"\A{re.escape(expected)}\Z"):
      irt.validate_item_bank(items)


class TestLogLikelihood:
  """Tests for the log_likelihood() function."""

  def test_log_likelihood_of_two_hand_worked_items(self) -> None:
    """At theta = 0, a right answer to [1, -1, 0, 1] has P = 1 / (1 + e^-1), a wrong one to [1, 0, 0.2, 0.9] Q = 0.45.

    Using P for the incorrect response, or ignoring c and d (Q = 0.5), changes the sum.
    """
    items = np.array([[1.0, -1.0, 0.0, 1.0], [1.0, 0.0, 0.2, 0.9]])
    expected = math.log(1 / (1 + math.exp(-1.0))) + math.log(0.45)

    value = irt.log_likelihood(0.0, [True, False], items)

    assert type(value) is float
    assert value == pytest.approx(expected, rel=1e-12)

  @pytest.mark.parametrize("z", [-700.0, -100.0, -37.0, -30.0, -25.0, 0.0, 25.0, 30.0, 37.0, 100.0, 700.0])
  def test_log_likelihood_is_exact_far_from_the_difficulty(self, z: float) -> None:
    """A 2PL item's log P and log Q should equal -log(1 + e^-z) and -log(1 + e^z) at any logit.

    Computing log(1 - P) after P rounds toward 1 loses precision past z = 25 and gives -inf past z = 37.
    """
    item = np.array([[1.0, 0.0, 0.0, 1.0]])

    assert irt.log_likelihood(z, [True], item) == pytest.approx(-np.logaddexp(0.0, -z), rel=1e-12)
    assert irt.log_likelihood(z, [False], item) == pytest.approx(-np.logaddexp(0.0, z), rel=1e-12)

  def test_log_likelihood_reaches_the_asymptotes(self) -> None:
    """Far from the difficulty, a guessed correct answer costs log c and a slipped incorrect one log(1 - d)."""
    guessing = np.array([[1.5, 0.0, 0.2, 1.0]])
    slipping = np.array([[1.5, 0.0, 0.0, 0.9]])

    assert irt.log_likelihood(-500.0, [True], guessing) == pytest.approx(math.log(0.2), rel=1e-12)
    assert irt.log_likelihood(500.0, [False], slipping) == pytest.approx(math.log(0.1), rel=1e-12)

  def test_log_likelihood_returns_one_value_per_theta(self) -> None:
    """An array of abilities gives one log-likelihood each, in the array's shape; summing over it as well gives one."""
    items = np.array([[1.0, -1.0, 0.0, 1.0], [1.2, 0.5, 0.1, 0.95]])
    responses = [True, False]
    thetas = np.linspace(-3.0, 3.0, 6).reshape(2, 3)

    values = irt.log_likelihood(thetas, responses, items)

    assert values.shape == (2, 3)
    np.testing.assert_allclose(
      values, [[irt.log_likelihood(float(theta), responses, items) for theta in row] for row in thetas], rtol=1e-12
    )

  def test_log_likelihood_is_zero_without_items(self) -> None:
    """With no administered items, every ability has log-likelihood 0."""
    values = irt.log_likelihood(np.linspace(-1.0, 1.0, 5), [], np.empty((0, 4)))

    np.testing.assert_array_equal(values, np.zeros(5))

  def test_log_likelihood_rejects_mismatched_lengths(self) -> None:
    """A response vector that does not match the items raises instead of broadcasting against them."""
    with pytest.raises(ValueError, match="responses and items must have the same length, got 2 and 1"):
      irt.log_likelihood(0.0, [True, False], np.array([[1.0, 0.0, 0.0, 1.0]]))

  def test_log_likelihood_rejects_non_boolean_responses(self) -> None:
    """Responses other than True and False raise instead of being read as correct."""
    with pytest.raises(ValueError, match="responses must contain only Boolean elements"):
      irt.log_likelihood(0.0, [True, 2], np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]))


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


def test_importing_every_public_irt_name_collects_no_test(pytester: pytest.Pytester) -> None:
  """A public irt function named test*, such as test_info, is collected by pytest in every module that imports it.

  pytest then runs it as a test and fails on its parameters ("fixture 'theta' not found").
  """
  pytester.makepyfile(test_user_module="from catsim.irt import *\n")

  result = pytester.runpytest("--collect-only", "-q")

  assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
