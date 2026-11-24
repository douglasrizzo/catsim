"""Tests for catsim.irt module."""

import math
from collections.abc import Callable

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

  @pytest.mark.parametrize(
    ("confidence", "z"),
    [(0.90, 1.6448536270), (0.95, 1.9599639845), (0.99, 2.5758293035)],
  )
  def test_interval_is_theta_plus_or_minus_the_normal_quantile_times_the_standard_error(
    self, confidence: float, z: float
  ) -> None:
    """The interval is theta +- z SE, with z the standard normal's (1 + confidence) / 2 quantile from published tables.

    Two 1PL items one logit above theta each hold information s(1) s(-1), with s the logistic, so the SE is
    1 / sqrt(2 s(1) s(-1)). A one-sided quantile (1.2816 for 90%), the variance in place of the SE, or an interval
    centred on the items' difficulty instead of theta fails.
    """
    theta = 0.5
    items = np.array([[1.0, theta + 1, 0.0, 1.0], [1.0, theta + 1, 0.0, 1.0]])
    s_1 = 1 / (1 + math.exp(-1))
    se = 1 / math.sqrt(2 * s_1 * (1 - s_1))

    lower, upper = irt.confidence_interval(theta, items, confidence=confidence)

    assert lower == pytest.approx(theta - z * se, rel=1e-9)
    assert upper == pytest.approx(theta + z * se, rel=1e-9)

  @pytest.mark.parametrize("confidence", [0.0, 1.0, 1.5])
  def test_rejects_a_confidence_outside_zero_and_one(self, confidence: float) -> None:
    """A confidence of 0 would give an interval of zero width, 1 an infinite one and 1.5 NaN bounds; each must raise."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="between 0 and 1"):
      irt.confidence_interval(0.0, items, confidence=confidence)


class TestReliability:
  """Tests for the reliability() function."""

  def test_reliability_is_one_minus_the_variance(self) -> None:
    """Twenty 2PL items with a = 1 at their difficulty give I = 5, so Rel = 1 - 1/5 = 0.8; 1 - I or 1 / I is not 0.8."""
    assert irt.reliability(0.0, np.array([[1.0, 0.0, 0.0, 1.0]] * 20)) == pytest.approx(0.8, rel=1e-12)

  def test_reliability_zero_info_is_negative_infinity(self) -> None:
    """Reliability 1 - 1 / I is unbounded below without information; returning +inf would read as a perfect test."""
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

  def test_max_info_theta_of_asymmetric_4pl_items(self) -> None:
    """With c + d != 1 the peak moves off b: -0.0575 for [1, 0, 0.25, 0.6] and 0.5632 for [1.5, 0.5, 0.2, 0.9].

    The expected values maximise a^2 (P - c)^2 (d - P)^2 / ((d - c)^2 P (1 - P)), written out by hand, with scipy's
    bounded scalar search, which locates a flat peak to about 1e-9. A 4PL offset wrong by 1% moves them by over 5e-4.
    """
    items = np.array([[1.0, 0.0, 0.25, 0.6], [1.5, 0.5, 0.2, 0.9]])

    np.testing.assert_allclose(irt.max_info_theta(items), [-0.0574595476, 0.5631544715], rtol=0, atol=1e-7)

  def test_max_info_theta_result_is_not_a_view_of_the_items(self) -> None:
    """Writing into the result leaves the item matrix alone; returning the difficulty column for 2PL items would not."""
    items = np.array([[1.0, 0.5, 0.0, 1.0], [1.3, -0.2, 0.0, 1.0]])
    thetas = irt.max_info_theta(items)
    np.testing.assert_array_equal(thetas, [0.5, -0.2])

    thetas[0] = 99.0

    np.testing.assert_array_equal(items[:, 1], [0.5, -0.2])


class TestLogResponseProbabilities:
  """Tests for the log_response_probabilities() function."""

  def test_log_response_probabilities_of_a_hand_worked_item(self) -> None:
    """At a logit of ln 3 the logistic term is 3/4, so [1, 0, 0.2, 0.9] has P = 0.2 + 0.7 * 3/4 = 0.725 and Q = 0.275.

    Ignoring c or d, or swapping the two outputs, changes either value.
    """
    log_p, log_q = irt.log_response_probabilities(math.log(3.0), np.array([1.0, 0.0, 0.2, 0.9]))

    assert float(log_p) == pytest.approx(math.log(0.725), rel=1e-12)
    assert float(log_q) == pytest.approx(math.log(0.275), rel=1e-12)

  @pytest.mark.parametrize(
    "item",
    [
      pytest.param((1.0, 0.0, 0.0, 1.0), id="2pl"),
      pytest.param((1.5, 0.3, 0.2, 1.0), id="3pl"),
      pytest.param((1.2, -0.5, 0.0, 0.9), id="4pl_without_guessing"),
      pytest.param((1.8, 0.4, 0.15, 0.85), id="4pl"),
    ],
  )
  def test_log_response_probabilities_match_the_icc(self, item: tuple[float, float, float, float]) -> None:
    """Near the difficulty, the outputs are log P and log(1 - P) of the icc.

    c = 0 and d = 1 reach the logaddexp branches with an infinite term, the other items with two finite ones,
    so a wrong branch on any path changes a value.
    """
    thetas = np.array([-2.0, -0.5, 0.0, 1.0, 3.0])
    p = irt.icc(thetas, np.array(item))

    log_p, log_q = irt.log_response_probabilities(thetas, np.array(item))

    np.testing.assert_allclose(log_p, np.log(p), rtol=1e-12)
    np.testing.assert_allclose(log_q, np.log1p(-p), rtol=1e-10)

  def test_log_response_probabilities_are_exact_where_p_rounds_to_0_or_1(self) -> None:
    """A 2PL item's log P and log Q equal -log(1 + e^-z) and -log(1 + e^z) at logits where P rounds to 0 or 1.

    Taking logarithms of P or of 1 - P once P has rounded gives -inf beyond |z| = 37 and loses precision past 25.
    """
    z = np.array([-700.0, -40.0, 40.0, 700.0])

    log_p, log_q = irt.log_response_probabilities(z, np.array([[1.0, 0.0, 0.0, 1.0]]))

    np.testing.assert_allclose(log_p[:, 0], -np.logaddexp(0.0, -z), rtol=1e-12)
    np.testing.assert_allclose(log_q[:, 0], -np.logaddexp(0.0, z), rtol=1e-12)


# log_likelihood validates its responses through log_likelihood_function, which must do it when built
_LOG_LIKELIHOOD_BUILDERS = [
  pytest.param(lambda responses, items: irt.log_likelihood(0.0, responses, items), id="log_likelihood"),
  pytest.param(irt.log_likelihood_function, id="log_likelihood_function"),
]


class TestLogLikelihood:
  """Tests for the log_likelihood() and log_likelihood_function() functions."""

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

    # near the difficulties log P and log(1 - P) are accurate to 1e-12
    probabilities = irt.icc(thetas, items)
    expected = np.log(probabilities[..., 0]) + np.log(1.0 - probabilities[..., 1])

    values = irt.log_likelihood(thetas, responses, items)

    assert values.shape == (2, 3)
    np.testing.assert_allclose(values, expected, rtol=1e-12)

  def test_log_likelihood_is_zero_without_items(self) -> None:
    """With no administered items, every ability has log-likelihood 0."""
    values = irt.log_likelihood(np.linspace(-1.0, 1.0, 5), [], np.empty((0, 4)))

    np.testing.assert_array_equal(values, np.zeros(5))

  def test_log_likelihood_function_keeps_the_pattern_it_was_built_from(self) -> None:
    """At theta = 1 a right answer to [1, 0, 0, 1] has log P = -log(1 + e^-1), even after the caller reuses its arrays.

    A function that reads the caller's items or responses at each call gives log Q = log 0.5 for the wrong answer to
    [1, 1, 0, 1] written over them.
    """
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    responses = [True]
    log_lik = irt.log_likelihood_function(responses, items)

    items[0, 1] = 1.0
    responses[0] = False

    assert log_lik(1.0) == pytest.approx(-math.log1p(math.exp(-1.0)), rel=1e-12)

  @pytest.mark.parametrize("build", _LOG_LIKELIHOOD_BUILDERS)
  def test_log_likelihood_rejects_mismatched_lengths(self, build: Callable[[list[bool], np.ndarray], object]) -> None:
    """A response vector that does not match the items raises instead of broadcasting against them.

    log_likelihood_function raises before it is given an ability; checking only inside the returned function does not.
    """
    with pytest.raises(ValueError, match="responses and items must have the same length, got 2 and 1"):
      build([True, False], np.array([[1.0, 0.0, 0.0, 1.0]]))

  @pytest.mark.parametrize("build", _LOG_LIKELIHOOD_BUILDERS)
  def test_log_likelihood_rejects_non_boolean_responses(
    self, build: Callable[[list[bool], np.ndarray], object]
  ) -> None:
    """Responses other than True and False raise instead of being read as correct.

    log_likelihood_function raises before it is given an ability; checking only inside the returned function does not.
    """
    with pytest.raises(ValueError, match="responses must contain only Boolean elements"):
      build([True, 2], np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]]))  # ty: ignore[invalid-argument-type]

  @pytest.mark.parametrize("build", _LOG_LIKELIHOOD_BUILDERS)
  @pytest.mark.parametrize(
    ("items", "responses"),
    [
      pytest.param(np.array([1.0, 0.0, 0.0, 1.0]), [True, False, True, True], id="one item, 1-D"),
      pytest.param(np.ones((1, 1, 4)), [True], id="3-D"),
      pytest.param(np.ones((1, 3)), [True], id="three columns"),
    ],
  )
  def test_log_likelihood_rejects_items_that_are_not_an_n_by_4_matrix(
    self, build: Callable[[list[bool], np.ndarray], object], items: np.ndarray, responses: list[bool]
  ) -> None:
    """Items that are not one (a, b, c, d) row per response raise a ValueError when the function is built.

    Fails if a single 1-D item is read as four items (one row broadcast across four responses), or if a 3-D matrix or a
    missing column passes construction and fails later with an IndexError or TypeError.
    """
    with pytest.raises(ValueError, match=r"items must be an \(n, 4\) matrix"):
      build(responses, items)

  def test_log_likelihood_of_unsigned_integer_items_is_the_formula(self) -> None:
    """Items stored as unsigned integers give the log-likelihood of the same parameters as floats.

    At theta = 0.5, a correct answer to (1, 0, 0, 1) gives log sigma(0.5) and a wrong one to (2, 1, 0, 1) gives
    log sigma(1). Fails if the discrimination's sign is flipped in the items' own dtype, where -2 wraps to 254.
    """
    items = np.array([[1, 0, 0, 1], [2, 1, 0, 1]], dtype=np.uint8)
    expected = -math.log1p(math.exp(-0.5)) - math.log1p(math.exp(-1.0))

    assert irt.log_likelihood(0.5, [True, False], items) == pytest.approx(expected, rel=1e-12)  # ty: ignore[no-matching-overload]


class TestThetaToScale:
  """Tests for the theta_to_scale() function."""

  def test_theta_to_scale_maps_the_typical_range_onto_0_to_100_by_default(self) -> None:
    """By default THETA_MIN_TYPICAL = -4 becomes 0, THETA_MAX_TYPICAL = 4 becomes 100 and 2 becomes 75.

    Other default bounds move the end points, and a wrong slope or intercept moves 2.
    """
    np.testing.assert_allclose(irt.theta_to_scale(np.array([-4.0, 4.0, 2.0])), [0.0, 100.0, 75.0], rtol=1e-12)

  def test_theta_to_scale_custom_scale(self) -> None:
    """On a 200 to 800 scale each unit of theta is 75 points, so theta = 1 is 575; a slope of 1 or 100 is not."""
    assert irt.theta_to_scale(1.0, scale_min=200, scale_max=800) == pytest.approx(575.0, rel=1e-12)


class TestScaleToTheta:
  """Tests for the scale_to_theta() function."""

  def test_scale_to_theta_maps_0_to_100_onto_the_typical_range_by_default(self) -> None:
    """By default 0 becomes THETA_MIN_TYPICAL = -4, 100 becomes THETA_MAX_TYPICAL = 4 and 75 becomes 2."""
    np.testing.assert_allclose(irt.scale_to_theta(np.array([0.0, 100.0, 75.0])), [-4.0, 4.0, 2.0], rtol=1e-12)

  def test_scale_to_theta_custom_scale(self) -> None:
    """On a 200 to 800 scale 575 is 75 points above the midpoint, theta = 1; multiplying by the slope gives 5625."""
    assert irt.scale_to_theta(575.0, scale_min=200, scale_max=800) == pytest.approx(1.0, rel=1e-12)


@pytest.mark.parametrize("function", [irt.theta_to_scale, irt.scale_to_theta], ids=["theta_to_scale", "scale_to_theta"])
@pytest.mark.parametrize(
  ("bounds", "message"),
  [
    pytest.param(
      {"theta_min": 1.0, "theta_max": 1.0}, r"theta_max \(1.0\) must be greater than theta_min \(1.0\)", id="empty"
    ),
    pytest.param(
      {"scale_min": 100, "scale_max": 0}, r"scale_max \(0\) must be greater than scale_min \(100\)", id="reversed"
    ),
  ],
)
def test_scale_conversions_reject_an_empty_or_reversed_range(
  function: Callable[..., float], bounds: dict[str, float], message: str
) -> None:
  """Both conversions reject an empty or reversed range; unchecked, an empty one divides by 0, a reversed one flips."""
  with pytest.raises(ValueError, match=message):
    function(50.0, **bounds)


def test_importing_every_public_irt_name_collects_no_test(pytester: pytest.Pytester) -> None:
  """A public irt function whose name starts with test is collected by pytest in every module that imports it.

  pytest then runs it as a test, and it fails on its parameters ("fixture 'theta' not found").
  """
  pytester.makepyfile(test_user_module="from catsim.irt import *\n")

  result = pytester.runpytest("--collect-only", "-q")

  assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
