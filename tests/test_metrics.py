"""Tests for catsim.metrics: each performance measure and its Monte Carlo standard error, on hand-worked data."""

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest

from catsim import metrics
from catsim.metrics import Metric

# Four examinees whose errors (estimate minus true ability) are 0.5, -0.5, 1.5 and -0.5.
# Bias: mean 0.25, deviations 0.25, -0.75, 1.25, -0.75, whose squares sum to 2.75; error sqrt(2.75 / (4 * 3)).
# MSE: squared errors 0.25, 0.25, 2.25, 0.25, mean 0.75, deviations -0.5, -0.5, 1.5, -0.5, whose squares sum to 3;
# error sqrt(3 / (4 * 3)) = 0.5.
# RMSE: sqrt(0.75); by the delta method, error 0.5 / (2 sqrt(0.75)).
TRUE_THETAS = [0.0, 0.0, 1.0, -1.0]
THETAS = [0.5, -0.5, 2.5, -1.5]


def test_bias_is_the_mean_error_with_the_error_of_a_mean() -> None:
  """The bias is 0.25 with a Monte Carlo error of sqrt(2.75 / 12), worked out above.

  Fails if the bias subtracts the estimates from the true abilities (-0.25), or if its error divides the squared
  deviations by N^2 (sqrt(2.75 / 16)) or by N - 1 alone (sqrt(2.75 / 3)), or returns numpy floats.
  """
  value, mc_se = metrics.bias(TRUE_THETAS, THETAS)

  assert value == pytest.approx(0.25)
  assert mc_se == pytest.approx(math.sqrt(2.75 / 12))
  assert (type(value), type(mc_se)) == (float, float)


def test_mse_is_the_mean_squared_error_with_the_error_of_that_mean() -> None:
  """The MSE is 0.75 with a Monte Carlo error of 0.5, from the spread of the squared errors (Morris et al., Table 6).

  Fails if the error is taken from the spread of the signed errors (sqrt(2.75 / 12), the bias's) instead of the
  squared ones, or if the MSE averages the signed errors.
  """
  assert metrics.mse(TRUE_THETAS, THETAS) == pytest.approx((0.75, 0.5))


def test_rmse_error_is_the_mse_error_through_the_delta_method() -> None:
  """The RMSE is sqrt(0.75), and its error the MSE's error over twice the RMSE, 0.5 / (2 sqrt(0.75)).

  Fails if the RMSE reports the MSE's error unchanged (0.5) or without the factor 2 (0.5 / sqrt(0.75)).
  """
  assert metrics.rmse(TRUE_THETAS, THETAS) == pytest.approx((math.sqrt(0.75), 0.5 / (2 * math.sqrt(0.75))))


def test_rmse_of_exact_estimates_has_zero_error() -> None:
  """Estimates equal to their true abilities have an RMSE of 0 and no Monte Carlo spread.

  Fails if the delta method divides the MSE's error of 0 by an RMSE of 0, a NaN with a warning that the test settings
  turn into an error.
  """
  assert metrics.rmse([0.5, -1.0, 2.0], [0.5, -1.0, 2.0]) == (0.0, 0.0)


def test_mean_of_test_lengths_has_the_error_of_a_mean() -> None:
  """Tests of 10, 12 and 20 items: mean 14, deviations -4, -2 and 6, so the error is sqrt(56 / (3 * 2)).

  Fails if the error takes the population variance (sqrt(56 / 9)) instead of the sample variance over N.
  """
  assert metrics.mean([10, 12, 20]) == pytest.approx((14.0, math.sqrt(56 / 6)))


def test_an_infinite_value_gives_an_infinite_mean_without_an_error() -> None:
  """A session that ended before its first response has an infinite standard error, so the mean is infinite.

  Fails if the spread of the values is computed anyway, as infinity minus infinity, a NaN with a warning that the test
  settings turn into an error, or if the mean drops the infinite value.
  """
  value, mc_se = metrics.mean([0.3, math.inf])

  assert value == math.inf
  assert math.isnan(mc_se)


@pytest.mark.parametrize(
  "metric",
  [
    lambda: metrics.bias([0.5], [1.0]),
    lambda: metrics.mse([0.5], [1.0]),
    lambda: metrics.rmse([0.5], [1.0]),
    lambda: metrics.mean([12]),
    lambda: metrics.overlap_rate([[0, 1]], n_items=3),
  ],
  ids=["bias", "mse", "rmse", "mean", "overlap_rate"],
)
def test_a_single_session_has_a_value_and_no_monte_carlo_error(metric: Callable[[], Metric[float]]) -> None:
  """One session gives the metric's value but no spread to estimate its error from, so the error is NaN.

  Fails if the error is computed with a sample variance of one value (a warning the test settings turn into an
  error), or reported as 0, which would claim a value known exactly.
  """
  value, mc_se = metric()

  assert math.isfinite(value)
  assert math.isnan(mc_se)


def test_exposure_rates_are_shares_of_the_sessions_with_binomial_errors() -> None:
  """Counts 4, 1, 0 and 2 over 4 sessions: rates 1, 0.25, 0 and 0.5, and errors sqrt(r (1 - r) / 4).

  The errors are 0, sqrt(0.1875 / 4), 0 and sqrt(0.25 / 4). Fails if the rates divide by the number of items, or if
  the errors divide by N - 1 or leave out the factor 1 - r.
  """
  rates, mc_se = metrics.exposure_rates(np.array([4, 1, 0, 2]), n_sessions=4)

  assert rates.tolist() == [1.0, 0.25, 0.0, 0.5]
  assert mc_se == pytest.approx([0.0, math.sqrt(0.1875 / 4), 0.0, 0.25])


# The 5-item tests below give every item to 3 of the 5 sessions, so every exposure rate is 3 / 5.
CYCLIC_TESTS = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 0], [4, 0, 1]]


@pytest.mark.parametrize(
  ("tests", "n_items", "expected"),
  [
    # Every item seen by 3 / 5 of the examinees: S_r^2 = 0, so T = L / M = 3 / 5.
    pytest.param(CYCLIC_TESTS, 5, 0.6, id="uniform"),
    # Every examinee takes items 0 and 1: rates 1, 1, 0, 0, so T = (1 + 1) / 2 = 1.
    pytest.param([[0, 1], [1, 0]], 4, 1.0, id="one-test-for-all"),
    # Rates 0.75, 0.75, 0.5: T = (0.5625 + 0.5625 + 0.25) / 2.
    pytest.param([[0, 1], [0, 1], [0, 2], [1, 2]], 3, 0.6875, id="four-tests"),
  ],
)
def test_overlap_rate_from_the_exposure_rates(tests: list[list[int]], n_items: int, expected: float) -> None:
  """The overlap rate is the sum of the squared exposure rates over the test length (Chen et al., equation 11).

  Fails if the variance of the rates takes the divisor M - 1, if the bank size and the test length are swapped, or if
  an unused item is left out of the bank size.
  """
  assert metrics.overlap_rate(tests, n_items).value == pytest.approx(expected)


def test_overlap_rate_is_the_mean_shared_share_over_every_ordered_pair_of_tests() -> None:
  """The large-sample overlap rate is the share of items two tests share, averaged over every ordered pair of sessions.

  The pairs include each session with itself, which is why it exceeds the average over pairs of distinct sessions.
  The pairs are counted here one by one, for 30 random 4-item tests of a 9-item bank. Fails if the value from the
  rates is not this average, as with a sample variance of the rates or a test length of the wrong session.
  """
  generator = np.random.default_rng(3)
  tests = [generator.choice(9, size=4, replace=False).tolist() for _ in range(30)]

  shared = [len(set(first) & set(second)) / 4 for first, second in itertools.product(tests, repeat=2)]

  assert metrics.overlap_rate(tests, n_items=9).value == pytest.approx(np.mean(shared))


def test_overlap_rate_error_for_a_hand_worked_run() -> None:
  """Tests [0, 1], [0, 1], [0, 2] and [1, 2] of a 3-item bank have rates 0.75, 0.75 and 0.5.

  Each test's summed rates are 1.5, 1.5, 1.25 and 1.25, deviating by 0.125 from their mean, so the delta method's
  error is (2 / 2) sqrt(4 * 0.125^2 / (4 * 3)) = 0.25 / sqrt(12). Fails if the gradient's factor 2 is left out, or if
  the spread is taken over the items' rates instead of the sessions' summed rates.
  """
  assert metrics.overlap_rate([[0, 1], [0, 1], [0, 2], [1, 2]], n_items=3).mc_se == pytest.approx(0.25 / math.sqrt(12))


def test_overlap_rate_error_matches_its_spread_over_replications() -> None:
  """The overlap rate's Monte Carlo error estimates its standard deviation across reruns of the same design.

  Each of 1000 replications draws 200 sessions independently, each taking one of the 15 two-item tests of a 6-item bank
  with fixed, unequal probabilities. The standard deviation of the 1000 overlap rates is known to about 2%, and the
  mean error must match it within 8%. Fails if the error leaves out the gradient's factor 2 (half the spread) or the
  division by the test length (twice the spread, at L = 2).
  """
  generator = np.random.default_rng(0)
  tests = np.array(list(itertools.combinations(range(6), 2)))
  weights = generator.random(len(tests)) ** 3
  weights /= weights.sum()

  values, errors = zip(
    *(metrics.overlap_rate(tests[generator.choice(len(tests), size=200, p=weights)], n_items=6) for _ in range(1000)),
    strict=True,
  )

  assert np.mean(errors) == pytest.approx(np.std(values, ddof=1), rel=0.08)


@pytest.mark.parametrize(
  ("call", "error", "message"),
  [
    pytest.param(
      lambda: metrics.bias([1.0, 2.0, 3.0], [1.0]),
      ValueError,
      "true_thetas has 3 entries and thetas 1; there must be one estimate per true ability",
      id="a-single-estimate-for-three",
    ),
    pytest.param(lambda: metrics.mse([], []), ValueError, "true_thetas cannot be empty", id="no-examinees"),
    pytest.param(
      lambda: metrics.rmse([[0.0, 1.0]], [[0.0, 1.0]]),
      TypeError,
      r"true_thetas must be one-dimensional, one value per session, got shape \(1, 2\)",
      id="a-row",
    ),
    pytest.param(
      lambda: metrics.bias([0.0, 1.0], [0.0, math.nan]),
      ValueError,
      r"thetas\[1\] is nan; a metric takes one number per session, with no missing ones",
      id="nan-estimate",
    ),
    pytest.param(
      lambda: metrics.bias([0.0, math.inf], [0.0, 1.0]),
      ValueError,
      r"true_thetas\[1\] is inf; abilities and their estimates must be finite",
      id="infinite-true-ability",
    ),
    pytest.param(
      lambda: metrics.mean([True, False]),
      TypeError,
      "values holds bool values; they must be real numbers",
      id="booleans",
    ),
    pytest.param(
      lambda: metrics.exposure_rates([5, 1], n_sessions=4),
      ValueError,
      r"counts\[0\] is 5; a count must be between 0 and the 4 sessions",
      id="more-administrations-than-sessions",
    ),
    pytest.param(
      lambda: metrics.exposure_rates([0.5, 1.0], n_sessions=4),
      TypeError,
      "counts must be a one-dimensional array of integers",
      id="rates-for-counts",
    ),
    pytest.param(
      lambda: metrics.exposure_rates([0, 1], n_sessions=0),
      ValueError,
      "n_sessions is 0; a run has at least one session",
      id="no-sessions",
    ),
    pytest.param(
      lambda: metrics.overlap_rate([[0, 3]], n_items=3),
      ValueError,
      r"administered_items\[0, 1\] is 3; an item index is 0 to 2",
      id="item-outside-the-bank",
    ),
    pytest.param(
      lambda: metrics.overlap_rate([[0, 1], [2, 2]], n_items=3),
      ValueError,
      r"administered_items\[1\] gives an item twice",
      id="item-given-twice",
    ),
    pytest.param(
      lambda: metrics.overlap_rate([0, 1, 2], n_items=3),
      TypeError,
      "administered_items must be a two-dimensional array of item indices",
      id="one-test-flat",
    ),
    pytest.param(
      lambda: metrics.overlap_rate(np.zeros((2, 0), dtype=int), n_items=3),
      ValueError,
      "the overlap rate needs at least one session of one item or more",
      id="tests-of-no-items",
    ),
  ],
)
def test_a_bad_input_raises_naming_it(call: Callable[[], object], error: type[Exception], message: str) -> None:
  """Each malformed input raises, naming the input, the position and value where there is one, and the rule.

  Fails if a metric broadcasts one estimate against three abilities, averages an empty array (NaN and a warning),
  compares a missing estimate, reads Booleans as 0 and 1, accepts more administrations than sessions or rates as
  counts, or counts an item outside the bank or given twice.
  """
  with pytest.raises(error, match=message):
    call()
