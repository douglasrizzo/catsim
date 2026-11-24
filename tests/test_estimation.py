"""Tests for catsim.estimation module."""

import importlib
import inspect
from collections.abc import Callable
from functools import partial
from itertools import pairwise, product
from typing import Any

import numpy as np
import pytest
from scipy.integrate import IntegrationWarning
from scipy.optimize import brentq, minimize_scalar
from scipy.special import expit

from catsim import irt
from catsim.estimation import (
  BaseEstimator,
  EAPEstimator,
  MAPEstimator,
  NumericalSearchEstimator,
  QuadratureGrid,
  WarmLikelihoodEstimator,
  normal_log_prior,
  posterior,
  posterior_mean,
  posterior_variance,
  uniform_log_prior,
)
from catsim.estimation.bayesian import LogPrior
from catsim.item_bank import ItemBank
from tests.contexts import make_context


class TestNumericalSearchEstimatorEstimate:
  """Tests for NumericalSearchEstimator.estimate() method."""

  def test_estimate_is_the_maximum_of_the_likelihood_on_the_ability_scale(self) -> None:
    """A 4PL pattern is estimated at its likelihood's maximum, 0.5816, inside [-6, 6].

    The guessing parameters make this likelihood rise again below -6, towards a limit still below
    its maximum. SciPy's brent and golden methods read (-6, 6) as a starting bracket rather than
    as bounds, followed that slope and returned -107.7 and -41.7. The reference is the argmax of
    the log-likelihood, written out here, on a grid 1e-4 apart over the ability scale.
    """
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    administered_items = list(range(10))
    response_vector = [True, True, False, True, False, True, True, False, True, False]
    items = item_bank.get_items(administered_items)
    grid = np.linspace(irt.THETA_MIN_EXTENDED, irt.THETA_MAX_EXTENDED, 120001)
    a, b, c, d = (items[:, [k]] for k in range(4))
    p = c + (d - c) / (1.0 + np.exp(-a * (grid - b)))
    log_lik = np.sum(np.where(np.array(response_vector)[:, None], np.log(p), np.log1p(-p)), axis=0)
    reference = grid[np.argmax(log_lik)]

    theta = NumericalSearchEstimator().estimate(make_context(item_bank, administered_items, response_vector, theta=0.0))

    assert reference == pytest.approx(0.5816, abs=1e-4)
    assert theta == pytest.approx(reference, abs=1e-4)

  def test_estimate_without_items_returns_est_theta(self) -> None:
    """With no item administered the estimate is est_theta, not a search bound.

    The log-likelihood of no responses is flat, and searching it returned 5.9999995.
    """
    item_bank = ItemBank.generate_item_bank(10, seed=42)

    theta = NumericalSearchEstimator().estimate(make_context(item_bank, [], [], theta=1.7))

    assert theta == 1.7

  @pytest.mark.parametrize(
    ("est_theta", "correct", "expected"),
    [
      pytest.param(0.0, True, 1.0, id="correct_halfway_to_the_hardest_item"),
      pytest.param(0.0, False, -1.5, id="incorrect_halfway_to_the_easiest_item"),
      pytest.param(4.0, True, 4.0, id="correct_above_the_hardest_item"),
      pytest.param(-4.0, False, -4.0, id="incorrect_below_the_easiest_item"),
    ],
  )
  def test_estimate_takes_dodds_step_on_a_constant_pattern(
    self, est_theta: float, correct: bool, expected: float
  ) -> None:
    """While every response is the same, the estimate moves half the way to the bank's extreme item, never back.

    The bank's difficulties are -3, 0 and 2, and only the item at 0 is administered, so the step
    targets the bank's extremes, not the administered item. From 0 it reaches 2 / 2 = 1 or -3 / 2 = -1.5.
    Beyond the extreme, the step used to point back toward it: 4.0 fell to 3.0 after a correct
    answer and -4.0 rose to -3.5 after an incorrect one.
    """
    item_bank = ItemBank(np.array([[1.0, -3.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]]))

    theta = NumericalSearchEstimator().estimate(make_context(item_bank, [1], [correct], theta=est_theta))

    assert theta == expected

  @pytest.mark.parametrize(
    ("est_theta", "correct", "bounds", "expected"),
    [
      pytest.param(1.0, True, (-1.8, 1.2), 1.2, id="step_past_the_upper_bound"),
      pytest.param(-1.0, False, (-1.8, 1.2), -1.8, id="step_past_the_lower_bound"),
      pytest.param(8.0, True, (-6.0, 6.0), 6.0, id="correct_from_above_the_bounds"),
      pytest.param(-8.0, False, (-6.0, 6.0), -6.0, id="incorrect_from_below_the_bounds"),
    ],
  )
  def test_dodds_step_is_clipped_to_the_bounds(
    self, est_theta: float, correct: bool, bounds: tuple[float, float], expected: float
  ) -> None:
    """Dodd's step never leaves the estimator's bounds.

    On the bank with difficulties -3, 0 and 2, the step from 1.0 reaches 1.5 and the step from -1.0
    reaches -2.0, both outside [-1.8, 1.2]. From 8.0 or -8.0 the estimate does not move, and stays
    outside [-6, 6] unless it is clipped.
    """
    item_bank = ItemBank(np.array([[1.0, -3.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 2.0, 0.0, 1.0]]))

    theta = NumericalSearchEstimator(bounds=bounds).estimate(make_context(item_bank, [1], [correct], theta=est_theta))

    assert theta == expected

  def test_estimate_increases_calls_counter(self) -> None:
    """Test that estimation increases the calls counter."""
    item_bank = ItemBank.generate_item_bank(50)
    estimator = NumericalSearchEstimator()

    assert estimator.calls == 0

    estimator.estimate(make_context(item_bank, [0, 1], [True, False], theta=0.0))

    assert estimator.calls == 1

    estimator.estimate(make_context(item_bank, [0, 1, 2], [True, False, True], theta=0.0))

    assert estimator.calls == 2

  def test_estimate_increases_evaluations(self) -> None:
    """Test that estimation increases the evaluations counter."""
    item_bank = ItemBank.generate_item_bank(50)
    estimator = NumericalSearchEstimator()

    assert estimator.evaluations == 0

    estimator.estimate(make_context(item_bank, [0, 1], [True, False], theta=0.0))

    # Evaluations should be > 0 after estimation
    assert estimator.evaluations > 0

  def test_estimate_avg_evaluations(self) -> None:
    """Test average evaluations calculation."""
    item_bank = ItemBank.generate_item_bank(50)
    estimator = NumericalSearchEstimator()

    # Perform multiple estimations
    for i in range(5):
      n_items = i + 2
      estimator.estimate(
        make_context(item_bank, list(range(n_items)), ([True, False] * (n_items // 2 + 1))[:n_items], theta=0.0)
      )

    avg = estimator.avg_evaluations
    assert avg == estimator.total_evaluations / estimator.calls
    assert avg >= 0

  def test_bounded_search_honors_tol(self) -> None:
    """The default bounded method used to ignore tol and always stop at SciPy's default of 1e-5."""
    item_bank = ItemBank.generate_item_bank(30, seed=3)
    administered_items = list(range(10))
    response_vector = [True, False] * 5
    reference = minimize_scalar(
      irt.negative_log_likelihood,
      bounds=(irt.THETA_MIN_EXTENDED, irt.THETA_MAX_EXTENDED),
      method="bounded",
      args=(response_vector, item_bank.get_items(administered_items)),
      options={"xatol": 1e-12},
    ).x
    coarse, fine = NumericalSearchEstimator(tol=1e-2), NumericalSearchEstimator(tol=1e-10)

    estimates = [
      estimator.estimate(make_context(item_bank, administered_items, response_vector, theta=0.0))
      for estimator in (coarse, fine)
    ]

    assert estimates[1] == pytest.approx(reference, abs=1e-7)
    assert coarse.evaluations < fine.evaluations


ESTIMATORS = [
  pytest.param(NumericalSearchEstimator, id="mle"),
  pytest.param(EAPEstimator, id="eap"),
  pytest.param(MAPEstimator, id="map"),
  pytest.param(WarmLikelihoodEstimator, id="wle"),
]

ESTIMATOR_REPRS = {
  "NumericalSearchEstimator": "NumericalSearchEstimator(tol=0.001, bounds=(-4.0, 4.0))",
  "EAPEstimator": "EAPEstimator(grid=None, log_prior=None)",
  "MAPEstimator": "MAPEstimator(log_prior=None, tol=0.001, bounds=(-4.0, 4.0))",
  "WarmLikelihoodEstimator": "WarmLikelihoodEstimator(tol=0.001, bounds=(-4.0, 4.0))",
}


@pytest.mark.parametrize("source", ESTIMATOR_REPRS.values(), ids=ESTIMATOR_REPRS.keys())
def test_repr_is_the_constructor_call(source: str) -> None:
  """An estimator's repr names every constructor argument with its value, so evaluating it rebuilds the estimator.

  Estimators printed a display name without their tolerance or bounds, and MEISelector's repr showed its estimator
  as an object address that does not evaluate.
  """
  namespace = vars(importlib.import_module("catsim.estimation"))

  assert repr(eval(source, namespace)) == source


def test_repr_shows_a_given_grid_and_prior_rather_than_none() -> None:
  """A repr that always printed None for the grid and prior would claim the defaults for an estimator with others."""
  grid = QuadratureGrid.uniform(n_nodes=3)
  prior = normal_log_prior(0.0, 2.0)

  assert repr(EAPEstimator(grid=grid, log_prior=prior)) == f"EAPEstimator(grid={grid!r}, log_prior={prior!r})"
  assert repr(MAPEstimator(log_prior=prior)) == f"MAPEstimator(log_prior={prior!r}, tol=1e-06, bounds=(-6.0, 6.0))"


def test_repr_table_covers_every_exported_estimator() -> None:
  """An estimator left out of ESTIMATOR_REPRS could hide an argument from its repr without any test noticing."""
  exported = {
    name
    for name, cls in vars(importlib.import_module("catsim.estimation")).items()
    if isinstance(cls, type) and issubclass(cls, BaseEstimator) and not inspect.isabstract(cls)
  }

  assert exported == set(ESTIMATOR_REPRS)


@pytest.mark.parametrize("make_estimator", ESTIMATORS)
@pytest.mark.parametrize(
  ("administered_items", "response_vector"),
  [pytest.param([0], [True], id="one_item"), pytest.param([0, 3], [True, False], id="two_items")],
)
def test_estimators_accept_numpy_item_indices(
  make_estimator: Callable[[], BaseEstimator], administered_items: list[int], response_vector: list[bool]
) -> None:
  """A NumPy index array gives the same estimate as the equivalent list.

  Testing the indices for truth raises for an array of two or more, and reads the array [0] as
  empty, so WLE returned est_theta for it without estimating anything.
  """
  item_bank = ItemBank.generate_item_bank(10, seed=42)

  from_list = make_estimator().estimate(make_context(item_bank, administered_items, response_vector, theta=0.7))
  from_array = make_estimator().estimate(
    make_context(item_bank, np.array(administered_items), response_vector, theta=0.7)
  )

  assert from_array == pytest.approx(from_list, abs=1e-15)


@pytest.mark.parametrize("estimator_class", [NumericalSearchEstimator, MAPEstimator, WarmLikelihoodEstimator])
@pytest.mark.parametrize("tol", [0.0, -1e-6, float("nan")])
def test_estimators_reject_a_tolerance_that_is_not_positive(estimator_class: type[BaseEstimator], tol: float) -> None:
  """A tol that is not positive fails at construction, not mid-session inside SciPy (WLE's brentq raised there)."""
  with pytest.raises(ValueError, match="tol must be positive"):
    estimator_class(tol=tol)


@pytest.mark.parametrize("estimator_class", [NumericalSearchEstimator, MAPEstimator, WarmLikelihoodEstimator])
@pytest.mark.parametrize(
  "bounds",
  [
    pytest.param((3.0, -3.0), id="reversed"),
    pytest.param((1.0, 1.0), id="empty"),
    pytest.param((-np.inf, 6.0), id="infinite"),
    pytest.param((float("nan"), 6.0), id="nan"),
  ],
)
def test_estimators_reject_bounds_that_are_not_a_finite_interval(
  estimator_class: type[BaseEstimator], bounds: tuple[float, float]
) -> None:
  """Bounds that are not a finite interval with low < high fail at construction, not inside SciPy."""
  with pytest.raises(ValueError, match="bounds must be finite with low < high"):
    estimator_class(bounds=bounds)


@pytest.mark.parametrize(
  "make_estimator",
  [
    pytest.param(partial(NumericalSearchEstimator, bounds=(0.5, 2.0)), id="mle"),
    pytest.param(partial(MAPEstimator, bounds=(0.5, 2.0)), id="map"),
    pytest.param(partial(WarmLikelihoodEstimator, bounds=(0.5, 2.0)), id="wle"),
  ],
)
def test_estimators_search_only_inside_their_bounds(make_estimator: Callable[[], BaseEstimator]) -> None:
  """A pattern whose estimate is 0 on [-6, 6] is estimated at the lower bound of [0.5, 2].

  A correct answer to a 1PL item at b = 1 and an incorrect one to an item at b = -1 make the
  likelihood, the test information and the standard normal prior symmetric about 0 and decreasing
  beyond it, so the likelihood, the posterior and the weighted likelihood all peak at 0 and fall
  across [0.5, 2]. An estimator that ignored its bounds returned 0.
  """
  item_bank = ItemBank(np.array([[1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0]]))

  theta = make_estimator().estimate(make_context(item_bank, [0, 1], [True, False], theta=0.0))

  assert theta == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize(
  ("make_estimator", "expected"),
  [
    pytest.param(partial(NumericalSearchEstimator, bounds=(-1.0, 1.0)), 1.0, id="mle"),
    pytest.param(
      partial(EAPEstimator, grid=QuadratureGrid.uniform(n_nodes=3, low=-1.0, high=1.0)),
      np.exp(-0.5) * np.tanh(0.5) / (np.exp(-0.5) + 0.5),
      id="eap",
    ),
    pytest.param(partial(MAPEstimator, bounds=(-1.0, 1.0)), brentq(lambda t: t - expit(-t), 0.0, 1.0), id="map"),
    pytest.param(partial(WarmLikelihoodEstimator, bounds=(-1.0, 1.0)), 1.0, id="wle"),
  ],
)
@pytest.mark.parametrize("correct", [True, False], ids=["correct", "incorrect"])
def test_estimators_give_a_constant_pattern_a_finite_estimate_inside_their_bounds(
  make_estimator: Callable[[], BaseEstimator], expected: float, correct: bool
) -> None:
  """One answer to a 1PL item at b = 0 has no finite MLE; every estimator stays inside [-1, 1].

  The bank's difficulties are -4, 0 and 4, and the estimates for an incorrect answer mirror those
  for a correct one. Dodd's step from 0 reaches +-2, clipped to +-1. The WLE's root, b +- ln 3, is
  beyond the bounds, so the estimate is the bound. MAP with the standard normal prior solves
  theta = sigma(-theta). EAP weighs the nodes -1, 0 and 1 by e^(-1/2) sigma(-1), 1/2 and
  e^(-1/2) sigma(1) after a correct answer, a mean of e^(-1/2) tanh(1/2) / (e^(-1/2) + 1/2).
  An estimator that ignored its bounds returned +-2 (MLE) or +-1.10 (WLE), and the MLE without
  Dodd's step +-inf.
  """
  item_bank = ItemBank(np.array([[1.0, -4.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 4.0, 0.0, 1.0]]))

  theta = make_estimator().estimate(make_context(item_bank, [1], [correct], theta=0.0))

  assert theta == pytest.approx(expected if correct else -expected, abs=1e-5)


class TestBayesianHelpers:
  """Tests for the shared Bayesian estimation helpers."""

  def test_prior_factories_are_re_exported(self) -> None:
    """Test that the prior factories are available from the package root."""
    normal_prior = normal_log_prior()
    uniform_prior = uniform_log_prior(-1.0, 1.0)

    theta = np.array([-2.0, 0.0, 2.0])
    normal_values = normal_prior(theta)
    uniform_values = uniform_prior(theta)

    assert normal_values.shape == theta.shape
    assert uniform_values.shape == theta.shape
    assert np.isfinite(normal_values).all()
    assert uniform_values[0] == -np.inf
    assert uniform_values[1] == pytest.approx(-np.log(2.0))
    assert uniform_values[2] == -np.inf

  def test_normal_log_prior_rejects_non_positive_sd(self) -> None:
    """Normal priors must have a positive standard deviation."""
    with pytest.raises(ValueError, match="sd must be positive"):
      normal_log_prior(sd=0.0)
    with pytest.raises(ValueError, match="sd must be positive"):
      normal_log_prior(sd=-1.0)
    with pytest.raises(ValueError, match="sd must be positive"):
      normal_log_prior(sd=float("nan"))
    with pytest.raises(ValueError, match="mean must be finite"):
      normal_log_prior(mean=float("inf"))

  def test_uniform_log_prior_rejects_inverted_bounds(self) -> None:
    """Uniform priors require a strictly increasing interval."""
    with pytest.raises(ValueError, match="high must be greater than low"):
      uniform_log_prior(1.0, 1.0)
    with pytest.raises(ValueError, match="high must be greater than low"):
      uniform_log_prior(2.0, 1.0)
    with pytest.raises(ValueError, match="must be finite"):
      uniform_log_prior(float("nan"), 1.0)

  def test_quadrature_grid_uniform_defaults(self) -> None:
    """Test the default uniform quadrature grid."""
    grid = QuadratureGrid.uniform()

    assert len(grid.nodes) == 121
    assert len(grid.weights) == 121
    assert np.diff(grid.nodes) == pytest.approx(np.full(120, 0.1))
    assert grid.nodes[0] == pytest.approx(-6.0)
    assert grid.nodes[-1] == pytest.approx(6.0)
    assert np.allclose(grid.weights, grid.weights[0])

  def test_quadrature_grid_uniform_rejects_invalid_parameters(self) -> None:
    """The grid factory should validate its shape and bounds."""
    with pytest.raises(ValueError, match="at least 2"):
      QuadratureGrid.uniform(n_nodes=1)
    with pytest.raises(ValueError, match="high must be greater than low"):
      QuadratureGrid.uniform(low=1.0, high=1.0)
    with pytest.raises(ValueError, match="high must be greater than low"):
      QuadratureGrid.uniform(low=2.0, high=1.0)

  @pytest.mark.parametrize(
    ("nodes", "weights", "match"),
    [
      (np.linspace(-1.0, 1.0, 5), np.array([0.3]), "same length"),
      (np.linspace(-1.0, 1.0, 5), np.ones(4), "same length"),
      (np.ones((2, 3)), np.ones((2, 3)), "one-dimensional"),
      (np.array([0.0]), np.array([1.0]), "at least 2"),
      (np.array([0.0, np.nan]), np.ones(2), "nodes must be finite"),
      (np.array([0.0, 1.0]), np.array([1.0, -1.0]), "weights must be finite and positive"),
      (np.array([0.0, 1.0]), np.array([1.0, np.inf]), "weights must be finite and positive"),
    ],
  )
  def test_quadrature_grid_rejects_invalid_arrays(self, nodes: np.ndarray, weights: np.ndarray, match: str) -> None:
    """A grid built directly from arrays should validate them."""
    with pytest.raises(ValueError, match=match):
      QuadratureGrid(nodes=nodes, weights=weights)

  def test_posterior_normalizes_to_one(self) -> None:
    """Test that the discrete posterior is normalized."""
    grid = QuadratureGrid.uniform()
    items = np.array([[1.5, 1.5, 0.0, 1.0]], dtype=float)

    post = posterior([True], items, grid, normal_log_prior())

    assert np.isclose(post.sum(), 1.0)

  def test_posterior_rejects_impossible_prior(self) -> None:
    """A prior that assigns zero mass everywhere should be rejected."""
    grid = QuadratureGrid.uniform()
    items = np.array([[1.5, 1.5, 0.0, 1.0]], dtype=float)

    def impossible_prior(theta: np.ndarray) -> np.ndarray:
      return np.full_like(theta, -np.inf, dtype=float)

    with pytest.raises(ValueError, match="posterior is undefined"):
      posterior([True], items, grid, impossible_prior)

  def test_posterior_mean_is_zero_with_empty_response_vector(self) -> None:
    """Test that the prior mean is recovered when there is no data."""
    grid = QuadratureGrid.uniform()
    empty_items = np.empty((0, 4), dtype=float)

    post = posterior([], empty_items, grid, normal_log_prior())
    mean = posterior_mean(post, grid.nodes)

    assert mean == pytest.approx(0.0, abs=1e-12)
    assert posterior_variance(post, grid.nodes) > 0

  def test_posterior_with_no_data_is_the_weighted_prior(self) -> None:
    """With no responses the posterior is the prior times the grid weights, normalized (contract 3)."""
    nodes = np.linspace(-3.0, 3.0, 7)
    weights = np.array([1.0, 2.0, 1.0, 3.0, 1.0, 2.0, 1.0])
    grid = QuadratureGrid(nodes=nodes, weights=weights)
    log_prior = normal_log_prior(0.5, 2.0)

    post = posterior([], np.empty((0, 4)), grid, log_prior)

    expected = np.exp(log_prior(nodes)) * weights
    assert post == pytest.approx(expected / expected.sum())

  def test_posterior_matches_a_brute_force_integral(self) -> None:
    """The default grid reproduces the posterior moments of a fine numerical integral."""
    items = np.array([[1.2, -0.5, 0.0, 1.0], [0.8, 0.3, 0.2, 1.0], [1.6, 1.0, 0.1, 0.95]])
    responses = [True, False, True]
    grid = QuadratureGrid.uniform()

    post = posterior(responses, items, grid, normal_log_prior())

    fine = np.linspace(-6.0, 6.0, 20001)
    a, b, c, d = (items[:, [k]] for k in range(4))
    p = c + (d - c) / (1.0 + np.exp(-a * (fine - b)))
    likelihood = np.prod(np.where(np.array(responses)[:, None], p, 1.0 - p), axis=0)
    density = likelihood * np.exp(-0.5 * fine**2)
    density /= np.trapezoid(density, fine)
    mean = np.trapezoid(fine * density, fine)
    variance = np.trapezoid((fine - mean) ** 2 * density, fine)

    assert posterior_mean(post, grid.nodes) == pytest.approx(mean, abs=1e-6)
    assert posterior_variance(post, grid.nodes) == pytest.approx(variance, rel=1e-5)

  @pytest.mark.parametrize("discrimination", [10.0, 200.0])
  def test_posterior_is_the_prior_when_the_likelihood_is_flat(self, discrimination: float) -> None:
    """Items whose likelihood is flat across the grid should leave the prior unchanged.

    An incorrect answer to an item at b = -6 and a correct one to an item at b = 6, both very
    discriminating, have a log-likelihood of about -12a at every node, dipping by up to log 2 at the
    two ends of the grid. When log(1 - P) was computed after P rounded to 1, it was -inf at most
    nodes: a = 10 pulled the posterior mean to -2.69 and its variance to 0.094, and a = 200 left no
    node with any mass.
    """
    grid = QuadratureGrid.uniform()
    items = np.array([[discrimination, -6.0, 0.0, 1.0], [discrimination, 6.0, 0.0, 1.0]])

    post = posterior([False, True], items, grid, normal_log_prior())
    prior = posterior([], np.empty((0, 4)), grid, normal_log_prior())

    # the likelihood is symmetric about 0, so the mean stays exact; the dips at the ends move the variance by ~1e-7
    assert posterior_mean(post, grid.nodes) == pytest.approx(posterior_mean(prior, grid.nodes), abs=1e-9)
    assert posterior_variance(post, grid.nodes) == pytest.approx(posterior_variance(prior, grid.nodes), rel=1e-6)

  def test_posterior_rejects_nan_log_prior(self) -> None:
    """A NaN in the log-prior should raise instead of returning an all-NaN posterior."""
    grid = QuadratureGrid.uniform()

    def nan_prior(theta: np.ndarray) -> np.ndarray:
      values = np.zeros_like(theta, dtype=float)
      values[0] = np.nan
      return values

    with pytest.raises(ValueError, match="NaN or \\+inf"):
      posterior([], np.empty((0, 4)), grid, nan_prior)

  def test_posterior_rejects_mismatched_responses_without_items(self) -> None:
    """Responses without matching items should raise, not silently return the prior."""
    grid = QuadratureGrid.uniform()

    with pytest.raises(ValueError, match="same length"):
      posterior([True, False], np.empty((0, 4)), grid, normal_log_prior())

  def test_posterior_shifts_positive_after_correct_response(self) -> None:
    """Test that a correct response on a difficult item moves the posterior upward."""
    grid = QuadratureGrid.uniform()
    items = np.array([[1.5, 2.0, 0.0, 1.0]], dtype=float)

    post = posterior([True], items, grid, normal_log_prior())
    mean = posterior_mean(post, grid.nodes)

    assert mean > 0.0

  def test_posterior_variance_is_zero_for_point_mass(self) -> None:
    """A point-mass posterior should have zero variance."""
    nodes = np.array([-1.0, 0.0, 1.0], dtype=float)
    post = np.array([0.0, 1.0, 0.0], dtype=float)

    assert posterior_mean(post, nodes) == pytest.approx(0.0)
    assert posterior_variance(post, nodes) == pytest.approx(0.0)

  def test_posterior_variance_decreases_as_items_accumulate(self) -> None:
    """Posterior variance must shrink as more items are administered.

    This verifies the monotone-concentration property expected by downstream
    Bayesian selectors and stoppers; broken normalization could widen the
    posterior without any other test noticing.
    """
    rng = np.random.default_rng(0)
    grid = QuadratureGrid.uniform()
    log_prior = normal_log_prior()

    items = np.column_stack([
      rng.uniform(0.8, 2.0, 15),
      rng.uniform(-2.0, 2.0, 15),
      np.zeros(15),
      np.ones(15),
    ])
    theta_true = 0.5
    responses = [bool(rng.random() < irt.icc(theta_true, *items[idx])) for idx in range(15)]

    variances = []
    for n_items in [1, 3, 5, 10, 15]:
      post = posterior(responses[:n_items], items[:n_items], grid, log_prior)
      variances.append(posterior_variance(post, grid.nodes))

    for earlier, later in pairwise(variances):
      assert later < earlier

  def test_posterior_is_numerically_stable_on_long_response_vector(self) -> None:
    """The posterior should stay normalized and finite on a 50-item response vector.

    This protects the log-max-subtract stabilization: if that step regressed,
    long response vectors could underflow to zeros or NaNs before normalization.
    """
    rng = np.random.default_rng(1)
    grid = QuadratureGrid.uniform()
    items = np.column_stack([
      rng.uniform(0.8, 2.0, 50),
      rng.uniform(-2.0, 2.0, 50),
      np.zeros(50),
      np.ones(50),
    ])
    theta_true = 1.5
    responses = [bool(rng.random() < irt.icc(theta_true, *items[idx])) for idx in range(50)]

    post = posterior(responses, items, grid, normal_log_prior())

    assert np.isfinite(post).all()
    assert post.sum() == pytest.approx(1.0)
    assert (post > 0).any()


class TestEAPEstimator:
  """Tests for the EAP estimator."""

  @pytest.mark.parametrize(
    ("log_prior", "expected"),
    [
      pytest.param(None, 0.0, id="default_standard_normal"),
      pytest.param(normal_log_prior(mean=1.2, sd=0.5), 1.2, id="shifted_normal"),
    ],
  )
  def test_estimate_returns_prior_mean_with_no_data(self, log_prior: LogPrior | None, expected: float) -> None:
    """With no responses the estimate is the prior mean, whatever est_theta says.

    An estimator that fell back to est_theta, or ignored a custom prior, would return 0.7 or 0.0.
    """
    estimator = EAPEstimator(log_prior=log_prior)

    theta = estimator.estimate(make_context(ItemBank.generate_item_bank(10, seed=42), [], [], theta=0.7))

    assert theta == pytest.approx(expected, abs=1e-12)

  def test_estimate_is_the_posterior_mean_over_the_given_grid_prior_and_items(self) -> None:
    """The estimate weighs the given grid's nodes by the given prior and the administered items only.

    On the 3-node grid {-1, 0, 1} with a flat prior, a correct answer to item 1 (a = 1, b = 0) gives
    the nodes masses proportional to sigma(-1), 1/2 and sigma(1), which sum to 3/2, so the posterior mean
    is (sigma(1) - sigma(-1)) / (3/2) = tanh(1/2) / 1.5 = 0.3081. Using item 0 instead gives 0.548, the
    default grid 2.759 and the default normal prior 0.253.
    """
    item_bank = ItemBank(np.array([[1.0, 3.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]))
    estimator = EAPEstimator(grid=QuadratureGrid.uniform(n_nodes=3, low=-1.0, high=1.0), log_prior=uniform_log_prior())

    theta = estimator.estimate(make_context(item_bank, [1], [True], theta=0.0))

    assert theta == pytest.approx(np.tanh(0.5) / 1.5, rel=1e-12)

  @pytest.mark.parametrize("sign", [1.0, -1.0], ids=["all_correct", "all_incorrect"])
  def test_estimate_is_pulled_toward_prior_on_extreme_short_pattern(self, sign: float) -> None:
    """A centered prior should shrink EAP more than a flat prior on an extreme pattern.

    All correct answers to hard items and all incorrect answers to easy items are the
    patterns where MLE diverges. This checks the core Bayesian behavior on sparse data,
    where prior shrinkage matters most and a purely likelihood-driven implementation
    would look too extreme.
    """
    item_bank = ItemBank(
      np.array(
        [
          [1.5, sign * 2.0, 0.0, 1.0],
          [1.2, sign * 2.5, 0.0, 1.0],
          [1.8, sign * 1.5, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    administered_items = [0, 1, 2]
    response_vector = [sign > 0] * 3

    normal_estimator = EAPEstimator(log_prior=normal_log_prior(mean=0.0, sd=1.0))
    normal_theta = normal_estimator.estimate(make_context(item_bank, administered_items, response_vector, theta=0.0))

    flat_estimator = EAPEstimator(log_prior=uniform_log_prior())
    flat_theta = flat_estimator.estimate(make_context(item_bank, administered_items, response_vector, theta=0.0))

    assert 0.0 < sign * normal_theta < 5.0
    assert abs(normal_theta) < abs(flat_theta)

  def test_evaluations_count_one_per_grid_node(self) -> None:
    """Each estimate evaluates the likelihood once per grid node, and the counters say so."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = EAPEstimator(grid=QuadratureGrid.uniform(n_nodes=31))

    estimator.estimate(make_context(item_bank, [0], [True], theta=0.0))
    estimator.estimate(make_context(item_bank, [0, 1], [True, False], theta=0.0))

    assert estimator.calls == 2
    assert estimator.evaluations == 31
    assert estimator.total_evaluations == 62


class TestMAPEstimator:
  """Tests for the Maximum a Posteriori estimator."""

  def test_estimate_empty_response_vector_returns_prior_mode(self) -> None:
    """With no responses the estimate is the default prior's mode, 0, not the est_theta passed in."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = MAPEstimator()

    theta = estimator.estimate(make_context(item_bank, [], [], theta=1.7))

    assert theta == pytest.approx(0.0, abs=1e-6)

  def test_estimate_maximizes_the_log_posterior(self) -> None:
    """Every response pattern's estimate is the argmax of the log-likelihood plus the log-prior.

    The reference is a dense grid search written out independently of catsim, on 3PL and 4PL items
    with the prior N(0.5, 0.8^2). It covers the all-correct and all-incorrect patterns, where the MLE
    diverges, and fails if the prior is dropped, mis-scaled or added with the wrong sign.
    """
    items = np.array([[1.2, -1.5, 0.2, 1.0], [0.8, -0.5, 0.0, 1.0], [1.6, 0.5, 0.25, 0.95], [1.0, 1.5, 0.1, 1.0]])
    mean, sd = 0.5, 0.8
    fine = np.linspace(irt.THETA_MIN_EXTENDED, irt.THETA_MAX_EXTENDED, 240_001)
    a, b, c, d = (items[:, [k]] for k in range(4))
    p = c + (d - c) / (1.0 + np.exp(-a * (fine - b)))
    log_prior = -0.5 * ((fine - mean) / sd) ** 2
    estimator = MAPEstimator(log_prior=normal_log_prior(mean, sd))

    for pattern in product([False, True], repeat=len(items)):
      log_likelihood = np.sum(np.where(np.array(pattern)[:, None], np.log(p), np.log1p(-p)), axis=0)
      reference = fine[np.argmax(log_likelihood + log_prior)]

      theta = estimator.estimate(make_context(ItemBank(items), list(range(len(items))), list(pattern), theta=0.0))

      assert theta == pytest.approx(reference, abs=1e-4), pattern

  def test_counters_add_the_refinement_to_the_search_points(self) -> None:
    """Each estimate counts the 201 search points plus the refinement steps, and the totals add up."""
    item_bank = ItemBank.generate_item_bank(20, seed=7)
    estimator = MAPEstimator()

    estimator.estimate(make_context(item_bank, [0, 1, 2, 3], [True, False, True, True], theta=0.0))
    first = estimator.evaluations
    estimator.estimate(make_context(item_bank, [0, 1, 2, 3, 4], [True, False, True, True, False], theta=0.0))

    assert first > 201
    assert estimator.evaluations > 201
    assert (estimator.calls, estimator.total_evaluations) == (2, first + estimator.evaluations)

  @staticmethod
  def _hard_items() -> ItemBank:
    """Three items too hard for any ability inside a narrow prior."""
    return ItemBank(np.array([[1.2, 2.0, 0.0, 1.0], [1.4, 2.5, 0.0, 1.0], [1.1, 3.0, 0.0, 1.0]], dtype=float))

  @pytest.mark.parametrize(
    ("low", "high", "response_vector", "expected"),
    [
      pytest.param(-1.0, 1.0, [True, False, True], 1.0, id="mode_at_upper_edge"),
      pytest.param(-1.0, 1.0, [False, False, False], -1.0, id="mode_at_lower_edge"),
      pytest.param(-6.0, -5.0, [True, False, True], -5.0, id="support_away_from_the_data"),
    ],
  )
  def test_estimate_stays_inside_a_uniform_prior_narrower_than_the_bounds(
    self, low: float, high: float, response_vector: list[bool], expected: float
  ) -> None:
    """The estimate must lie in the prior's support, at the edge the likelihood points to.

    Outside the support the negative log-posterior is +inf, a flat region a bounded scalar
    search cannot cross, so searching the whole bounds at once returned 6.0 in every case.
    """
    estimator = MAPEstimator(log_prior=uniform_log_prior(low, high))

    theta = estimator.estimate(make_context(self._hard_items(), [0, 1, 2], response_vector, theta=0.0))

    assert theta == pytest.approx(expected, abs=1e-5)

  def test_estimate_stays_inside_a_prior_narrower_than_a_grid_step(self) -> None:
    """Both neighbours of the only grid point in the support lie outside it.

    The refinement then searches where the objective is mostly +inf; the estimate must
    still lie inside the support rather than wherever that search stops.
    """
    estimator = MAPEstimator(log_prior=uniform_log_prior(-0.01, 0.01))

    theta = estimator.estimate(make_context(self._hard_items(), [0, 1, 2], [True, False, True], theta=0.0))

    assert -0.01 <= theta <= 0.01

  def test_estimate_raises_when_the_prior_excludes_the_bounds(self) -> None:
    """A prior with no support inside the search bounds leaves no mode to find."""
    estimator = MAPEstimator(log_prior=uniform_log_prior(7.0, 8.0))

    with pytest.raises(ValueError, match="not finite at any"):
      estimator.estimate(make_context(self._hard_items(), [0], [True], theta=0.0))

  def test_estimate_locates_an_off_grid_prior_mode_within_tol(self) -> None:
    """With no responses the estimate is the prior mode to within tol, not to the grid's 0.06."""
    estimator = MAPEstimator(log_prior=normal_log_prior(mean=0.5, sd=1.0), tol=1e-8)

    theta = estimator.estimate(make_context(self._hard_items(), [], [], theta=0.0))

    assert theta == pytest.approx(0.5, abs=1e-6)

  def test_estimate_refines_a_mode_between_two_tied_points(self) -> None:
    """A peak halfway between two grid points makes them tie, which is not a flat posterior.

    The prior mode 0.03 sits midway between the points 0.0 and 0.06, which tie exactly;
    the estimate must still be refined to the mode instead of returning one of them.
    """
    estimator = MAPEstimator(log_prior=normal_log_prior(mean=0.03, sd=1.0), tol=1e-8)

    theta = estimator.estimate(make_context(self._hard_items(), [], [], theta=0.0))

    assert theta == pytest.approx(0.03, abs=1e-6)

  def test_estimate_raises_when_the_support_falls_between_grid_points(self) -> None:
    """A support narrower than the grid spacing that holds no grid point cannot be searched."""
    estimator = MAPEstimator(log_prior=uniform_log_prior(0.01, 0.05))

    with pytest.raises(ValueError, match="must contain at least one"):
      estimator.estimate(make_context(self._hard_items(), [0], [True], theta=0.0))

  def test_estimate_finds_the_global_mode_of_a_bimodal_prior(self) -> None:
    """The grid search picks the taller of two separated modes, wherever a local search would stop."""

    def log_prior(theta: np.ndarray) -> np.ndarray:
      low_mode = np.log(0.3) - 0.5 * ((theta + 2.0) / 0.3) ** 2
      high_mode = np.log(0.7) - 0.5 * ((theta - 2.0) / 0.3) ** 2
      return np.logaddexp(low_mode, high_mode)

    estimator = MAPEstimator(log_prior=log_prior)

    theta = estimator.estimate(make_context(self._hard_items(), [], [], theta=0.0))

    assert theta == pytest.approx(2.0, abs=1e-5)

  def test_estimate_takes_the_middle_of_a_flat_posterior(self) -> None:
    """A uniform prior with no responses has no unique mode, so the middle of the bounds is returned."""
    estimator = MAPEstimator(log_prior=uniform_log_prior())

    theta = estimator.estimate(make_context(self._hard_items(), [], [], theta=0.0))

    assert theta == pytest.approx(0.0, abs=1e-12)


class TestWarmLikelihoodEstimator:
  """Tests for WarmLikelihoodEstimator."""

  def test_estimate_empty_response_returns_input_theta(self) -> None:
    """No administered items should return the input estimate unchanged."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = WarmLikelihoodEstimator()

    theta = estimator.estimate(make_context(item_bank, [], [], theta=1.25))

    assert theta == pytest.approx(1.25)
    assert estimator.calls == 1

  @staticmethod
  def _two_candidate_bank() -> ItemBank:
    """Two 3PL/4PL items whose modified score yields two maxima candidates after two correct answers."""
    return ItemBank(np.array([[2.13, -2.78, 0.13, 0.95], [1.11, 2.02, 0.3, 0.9]]))

  def test_evaluations_count_the_scan_the_root_search_and_the_ranking(self) -> None:
    """The counters include every modified-score evaluation, including the ranking integrals.

    A single root costs the 121-point scan plus a few Brent steps; ranking two candidates adds
    two integrals of at least 21 evaluations each.
    """
    estimator = WarmLikelihoodEstimator()
    single_bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.2, 0.5, 0.0, 1.0]]))

    estimator.estimate(make_context(single_bank, [0, 1], [True, False], theta=0.0))
    single = estimator.evaluations
    estimator.estimate(make_context(self._two_candidate_bank(), [0, 1], [True, True], theta=0.0))
    ranked = estimator.evaluations

    assert 121 < single < 121 + 50
    assert ranked >= 121 + 2 * 21
    assert (estimator.calls, estimator.total_evaluations) == (2, single + ranked)

  def test_ranking_integral_warnings_reach_the_caller(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """A ranking integral that does not converge still warns, although quad returns the message."""
    wle_module = importlib.import_module("catsim.estimation.wle")
    real_quad = wle_module.quad

    def quad_with_warning(*args: Any, **kwargs: Any) -> tuple[Any, ...]:
      return (*real_quad(*args, **kwargs)[:3], "the maximum number of subdivisions has been achieved")

    monkeypatch.setattr(wle_module, "quad", quad_with_warning)

    with pytest.warns(IntegrationWarning, match="subdivisions"):
      WarmLikelihoodEstimator().estimate(make_context(self._two_candidate_bank(), [0, 1], [True, True], theta=0.0))

  @pytest.mark.parametrize(
    ("b", "correct", "expected"),
    [
      pytest.param(10.0, True, irt.THETA_MAX_EXTENDED, id="correct_beyond_the_upper_bound"),
      pytest.param(-10.0, False, irt.THETA_MIN_EXTENDED, id="incorrect_beyond_the_lower_bound"),
    ],
  )
  def test_estimate_falls_back_to_the_bound_the_score_points_to(self, b: float, correct: bool, expected: float) -> None:
    """With no root in the search range the estimate is the bound the modified score points to.

    For one 2PL item the weighted likelihood P^1.5 Q^0.5 peaks where P = 3/4 after a correct
    answer (P = 1/4 after an incorrect one), at b +- ln 3 / a, beyond [-6, 6] for b = +-10.
    """
    estimator = WarmLikelihoodEstimator()

    theta = estimator.estimate(make_context(ItemBank(np.array([[1.0, b, 0.0, 1.0]])), [0], [correct], theta=0.0))

    assert theta == expected

  def test_estimate_picks_the_candidate_with_the_largest_weighted_likelihood(self) -> None:
    """Among several maxima candidates the estimate is the global maximum of the weighted likelihood.

    The weighted log-likelihood is the integral of the modified score, so its maximum is found
    here by integrating the independent reference score on a fine grid.
    """
    items = self._two_candidate_bank().items
    response_vector = [True, True]
    scan = np.linspace(irt.THETA_MIN_EXTENDED, irt.THETA_MAX_EXTENDED, 121)
    scan_scores = np.array([_warm_modified_score(theta, response_vector, items) for theta in scan])
    sign_changes = int(np.sum((scan_scores[:-1] > 0) & (scan_scores[1:] <= 0)))
    assert sign_changes + int(scan_scores[0] <= 0) + int(scan_scores[-1] > 0) >= 2

    fine = np.linspace(irt.THETA_MIN_EXTENDED, irt.THETA_MAX_EXTENDED, 24_001)
    fine_scores = np.array([_warm_modified_score(theta, response_vector, items) for theta in fine])
    steps = (fine_scores[1:] + fine_scores[:-1]) / 2 * np.diff(fine)
    weighted_log_likelihood = np.concatenate([[0.0], np.cumsum(steps)])

    theta = WarmLikelihoodEstimator().estimate(
      make_context(self._two_candidate_bank(), [0, 1], response_vector, theta=0.0)
    )

    assert theta == pytest.approx(fine[int(np.argmax(weighted_log_likelihood))], abs=1e-3)

  @pytest.mark.parametrize(
    ("b", "theta", "correct"),
    [
      pytest.param(5.0, -2.0, True, id="correct_on_a_far_too_hard_item"),
      pytest.param(-5.0, 2.0, False, id="incorrect_on_a_far_too_easy_item"),
    ],
  )
  def test_modified_score_is_exact_where_the_probability_rounds_off(
    self, b: float, theta: float, correct: bool
  ) -> None:
    """At a logit of +-70 one of P and Q is below 1e-30, and the score must keep it.

    For one item without guessing or slipping the modified score is a Q + a (Q - P) / 2 after
    a correct answer and -a P + a (Q - P) / 2 after an incorrect one. Computing Q as 1 - P and
    clipping P Q at machine epsilon lost the likelihood term entirely, leaving only the
    correction: 5 instead of 15 for the far-too-hard item.
    """
    a = 10.0
    items = np.array([[a, b, 0.0, 1.0]])
    z = a * (theta - b)
    p, q = expit(z), expit(-z)
    expected = (a * q if correct else -a * p) + a * (q - p) / 2.0

    modified_score = WarmLikelihoodEstimator.__dict__["_modified_score"].__func__
    value = modified_score(theta, [correct], items)[0]

    assert value == pytest.approx(expected, rel=1e-12)

  def test_estimate_solves_warm_modified_score_equation(self) -> None:
    """Every estimate must be an interior root of Warm's modified score equation.

    The bank has guessing parameters, where the WLE is not the maximizer of L * sqrt(I), so this
    checks Warm's (1989) defining equation directly. It covers every 5-item response pattern,
    including those whose likelihood is maximized at the search bound: an earlier one-step
    correction (MLE + J / (2 I^2)) returned about 2670 for the pattern 00010 on this bank.
    """
    item_bank = ItemBank.generate_item_bank(100, seed=0)
    administered_items = list(range(5))
    items = item_bank.get_items(administered_items)
    estimator = WarmLikelihoodEstimator()

    for pattern in product([False, True], repeat=len(administered_items)):
      response_vector = list(pattern)
      theta = estimator.estimate(make_context(item_bank, administered_items, response_vector, theta=0.0))

      assert irt.THETA_MIN_EXTENDED < theta < irt.THETA_MAX_EXTENDED
      assert _warm_modified_score(theta, response_vector, items) == pytest.approx(0.0, abs=1e-4)

  def test_estimate_maximizes_jeffreys_weighted_likelihood_without_guessing(self) -> None:
    """Without guessing or slipping, J = I' and the WLE maximizes L(theta) * sqrt(I(theta)).

    The reference is a dense grid search of log L + 0.5 log I over every response pattern.
    """
    items = np.column_stack([np.linspace(0.8, 2.0, 5), np.linspace(-1.5, 1.5, 5), np.zeros(5), np.ones(5)])
    item_bank = ItemBank(items)
    grid = np.linspace(irt.THETA_MIN_EXTENDED, irt.THETA_MAX_EXTENDED, 12001)
    p = 1.0 / (1.0 + np.exp(-items[:, 0] * (grid[:, None] - items[:, 1])))
    log_info = np.log(np.sum(items[:, 0] ** 2 * p * (1.0 - p), axis=1))
    estimator = WarmLikelihoodEstimator()

    for pattern in product([False, True], repeat=len(items)):
      responses = np.array(pattern)
      log_lik = np.sum(np.where(responses, np.log(p), np.log1p(-p)), axis=1)
      reference = grid[np.argmax(log_lik + 0.5 * log_info)]

      theta = estimator.estimate(make_context(item_bank, list(range(len(items))), list(pattern), theta=0.0))

      assert theta == pytest.approx(reference, abs=2e-3)

  def test_exact_bias_is_small_and_below_first_order_mle_bias(self) -> None:
    """WLE removes the first-order bias of the MLE on a short 2PL test (Warm, 1989).

    The expected bias is computed exactly by enumerating all 2^8 response patterns weighted by
    their probabilities, so the check is deterministic and needs no conditioning on mixed
    patterns (the MLE diverges on the all-correct and all-incorrect ones). The MLE reference is
    its first-order bias -J / (2 I^2) (Lord, 1983); wherever that is non-negligible, WLE's exact
    bias must be less than half its size.
    """
    items = np.column_stack([np.linspace(0.8, 2.0, 8), np.linspace(-2.0, 2.0, 8), np.zeros(8), np.ones(8)])
    item_bank = ItemBank(items)
    patterns = [list(pattern) for pattern in product([False, True], repeat=len(items))]
    estimator = WarmLikelihoodEstimator()
    estimates = np.array([
      estimator.estimate(make_context(item_bank, list(range(len(items))), pattern, theta=0.0)) for pattern in patterns
    ])

    for theta in (-2.0, -1.0, 0.0, 1.0, 2.0):
      p = 1.0 / (1.0 + np.exp(-items[:, 0] * (theta - items[:, 1])))
      probabilities = np.array([np.prod(np.where(pattern, p, 1.0 - p)) for pattern in patterns])
      wle_bias = float(probabilities @ (estimates - theta))

      dp = items[:, 0] * p * (1.0 - p)
      d2p = items[:, 0] * dp * (1.0 - 2.0 * p)
      info = np.sum(dp**2 / (p * (1.0 - p)))
      mle_bias = -np.sum(dp * d2p / (p * (1.0 - p))) / (2.0 * info**2)

      assert abs(wle_bias) < 0.1
      if abs(mle_bias) > 0.05:
        assert abs(wle_bias) < 0.5 * abs(mle_bias)


def _warm_modified_score(theta: float, response_vector: list[bool], items: np.ndarray) -> float:
  """Warm's (1989) modified score, written out independently of the estimator under test."""
  a, b, c, d = items.T
  logistic = 1.0 / (1.0 + np.exp(-a * (theta - b)))
  p = c + (d - c) * logistic
  dp = a * (d - c) * logistic * (1.0 - logistic)
  d2p = a * dp * (1.0 - 2.0 * logistic)
  pq = p * (1.0 - p)
  u = np.asarray(response_vector, dtype=float)
  info = np.sum(dp**2 / pq)
  j = np.sum(dp * d2p / pq)
  return float(np.sum((u - p) * dp / pq) + j / (2.0 * info))
