"""Tests for catsim.estimation module."""

from itertools import pairwise, product
from typing import Any

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from catsim import irt
from catsim.estimation import (
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

# SciPy reads `bracket` as where to start looking, not as a bound. From (-6, 6), brent and golden search downhill
# past -6 into the flat tail that a guessing parameter gives the likelihood, and stop far off the ability scale
# at a point less likely than the maximum inside it (.plans/maintenance/todo/M05-unbounded-scipy-estimators.md).
UNBOUNDED_SCIPY_SEARCH = pytest.mark.xfail(strict=True, reason="brent and golden leave the ability scale")


class TestNumericalSearchEstimatorInit:
  """Tests for NumericalSearchEstimator initialization."""

  def test_init_with_invalid_method_raises(self) -> None:
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="must be one of"):
      NumericalSearchEstimator(method="invalid_method")


class TestNumericalSearchEstimatorMethods:
  """Tests for NumericalSearchEstimator available methods."""

  def test_available_methods_contains_expected(self) -> None:
    """Test that available methods contains expected methods."""
    methods = NumericalSearchEstimator.available_methods()
    expected = {"ternary", "dichotomous", "fibonacci", "golden", "brent", "bounded", "golden2"}
    assert methods == expected

  def test_all_methods_can_be_instantiated(self) -> None:
    """Test that all available methods can be instantiated."""
    for method in NumericalSearchEstimator.available_methods():
      estimator = NumericalSearchEstimator(method=method)
      assert str(estimator) == f"Numerical Search Estimator ({method})"


class TestNumericalSearchEstimatorEstimate:
  """Tests for NumericalSearchEstimator.estimate() method."""

  def test_estimate_basic(self) -> None:
    """Test basic estimation with mixed responses."""
    item_bank = ItemBank.generate_item_bank(50)
    estimator = NumericalSearchEstimator()

    # Mixed responses
    administered_items = [0, 1, 2, 3, 4]
    response_vector = [True, True, False, True, False]

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert isinstance(theta, float)
    # Theta should be finite
    assert np.isfinite(theta)

  def test_estimate_all_correct(self) -> None:
    """Test estimation with all correct responses."""
    item_bank = ItemBank.generate_item_bank(50)
    estimator = NumericalSearchEstimator(dodd=True)

    administered_items = [0, 1, 2]
    response_vector = [True, True, True]

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert isinstance(theta, float)
    assert np.isfinite(theta)
    # With all correct, theta should increase from initial
    assert theta > 0.0

  def test_estimate_all_incorrect(self) -> None:
    """Test estimation with all incorrect responses."""
    item_bank = ItemBank.generate_item_bank(50)
    estimator = NumericalSearchEstimator(dodd=True)

    administered_items = [0, 1, 2]
    response_vector = [False, False, False]

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert isinstance(theta, float)
    assert np.isfinite(theta)
    # With all incorrect, theta should decrease from initial
    assert theta < 0.0

  def test_estimate_increases_calls_counter(self) -> None:
    """Test that estimation increases the calls counter."""
    item_bank = ItemBank.generate_item_bank(50)
    estimator = NumericalSearchEstimator()

    assert estimator.calls == 0

    estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1],
      response_vector=[True, False],
      est_theta=0.0,
    )

    assert estimator.calls == 1

    estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1, 2],
      response_vector=[True, False, True],
      est_theta=0.0,
    )

    assert estimator.calls == 2

  def test_estimate_increases_evaluations(self) -> None:
    """Test that estimation increases the evaluations counter."""
    item_bank = ItemBank.generate_item_bank(50)
    estimator = NumericalSearchEstimator()

    assert estimator.evaluations == 0

    estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1],
      response_vector=[True, False],
      est_theta=0.0,
    )

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
        item_bank=item_bank,
        administered_items=list(range(n_items)),
        response_vector=([True, False] * (n_items // 2 + 1))[:n_items],
        est_theta=0.0,
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
      estimator.estimate(
        item_bank=item_bank, administered_items=administered_items, response_vector=response_vector, est_theta=0.0
      )
      for estimator in (coarse, fine)
    ]

    assert estimates[1] == pytest.approx(reference, abs=1e-7)
    assert coarse.evaluations < fine.evaluations


class TestNumericalSearchEstimatorWithMethods:
  """Test estimation with different methods."""

  @pytest.mark.parametrize("method", sorted(NumericalSearchEstimator.available_methods()))
  def test_estimate_with_all_methods(self, method: str) -> None:
    """Test that all methods produce valid estimates."""
    item_bank = ItemBank.generate_item_bank(50)
    estimator = NumericalSearchEstimator(method=method)

    administered_items = [0, 1, 2, 3, 4]
    response_vector = [True, True, False, True, False]

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert isinstance(theta, float)
    assert np.isfinite(theta)

  @pytest.mark.parametrize(
    "method",
    [
      "bounded",
      "dichotomous",
      "fibonacci",
      "golden2",
      "ternary",
      pytest.param("brent", marks=UNBOUNDED_SCIPY_SEARCH),
      pytest.param("golden", marks=UNBOUNDED_SCIPY_SEARCH),
    ],
  )
  def test_methods_agree_on_the_maximum_likelihood_estimate(self, method: str) -> None:
    """Every method should reach the bounded optimum to within a few times the default tol of 1e-6."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)

    administered_items = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    response_vector = [True, True, False, True, False, True, True, False, True, False]

    # Use bounded as reference
    ref_estimator = NumericalSearchEstimator(method="bounded")
    ref_theta = ref_estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    estimator = NumericalSearchEstimator(method=method)
    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert theta == pytest.approx(ref_theta, abs=1e-5)


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

    theta = estimator.estimate(ItemBank.generate_item_bank(10, seed=42), [], [], 0.7)

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

    theta = estimator.estimate(item_bank, [1], [True], 0.0)

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
    normal_theta = normal_estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    flat_estimator = EAPEstimator(log_prior=uniform_log_prior())
    flat_theta = flat_estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert 0.0 < sign * normal_theta < 5.0
    assert abs(normal_theta) < abs(flat_theta)

  def test_evaluations_count_one_per_grid_node(self) -> None:
    """Each estimate evaluates the likelihood once per grid node, and the counters say so."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = EAPEstimator(grid=QuadratureGrid.uniform(n_nodes=31))

    estimator.estimate(item_bank=item_bank, administered_items=[0], response_vector=[True], est_theta=0.0)
    estimator.estimate(item_bank=item_bank, administered_items=[0, 1], response_vector=[True, False], est_theta=0.0)

    assert estimator.calls == 2
    assert estimator.evaluations == 31
    assert estimator.total_evaluations == 62

  def test_rejected_estimate_leaves_the_counters_alone(self) -> None:
    """A call that fails validation is not counted as an estimate."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = EAPEstimator()

    with pytest.raises(ValueError, match="same length"):
      estimator.estimate(item_bank=item_bank, administered_items=[0, 1], response_vector=[True], est_theta=0.0)

    assert (estimator.calls, estimator.evaluations, estimator.total_evaluations) == (0, 0, 0)

  @pytest.mark.parametrize("missing", ["item_bank", "administered_items", "response_vector"])
  def test_estimate_rejects_missing_inputs(self, missing: str) -> None:
    """None inputs raise the shared estimator error instead of failing inside NumPy."""
    kwargs = {"item_bank": ItemBank.generate_item_bank(5, seed=1), "administered_items": [0], "response_vector": [True]}
    kwargs[missing] = None

    with pytest.raises(ValueError, match=f"{missing} parameter cannot be None"):
      EAPEstimator().estimate(**kwargs, est_theta=0.0)

  def test_estimate_accepts_numpy_item_indices(self) -> None:
    """A NumPy index array gives the same estimate as the equivalent list."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = EAPEstimator()

    from_list = estimator.estimate(item_bank, [0, 3], [True, False], 0.0)
    from_array = estimator.estimate(item_bank, np.array([0, 3]), [True, False], 0.0)

    assert from_array == pytest.approx(from_list, abs=1e-15)


class TestMAPEstimator:
  """Tests for the Maximum a Posteriori estimator."""

  @pytest.mark.parametrize(
    ("kwargs", "match"),
    [
      pytest.param({"tol": 0.0}, "tol must be positive", id="zero_tol"),
      pytest.param({"tol": -1e-6}, "tol must be positive", id="negative_tol"),
      pytest.param({"bounds": (3.0, -3.0)}, "low < high", id="reversed_bounds"),
      pytest.param({"bounds": (1.0, 1.0)}, "low < high", id="empty_bounds"),
      pytest.param({"bounds": (-np.inf, 6.0)}, "finite", id="infinite_bound"),
    ],
  )
  def test_init_rejects_invalid_search_settings(self, kwargs: dict[str, Any], match: str) -> None:
    """Invalid bounds or tolerances fail at construction, not inside SciPy."""
    with pytest.raises(ValueError, match=match):
      MAPEstimator(**kwargs)

  def test_estimate_empty_response_vector_returns_prior_mode(self) -> None:
    """With no responses the estimate is the default prior's mode, 0, not the est_theta passed in."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = MAPEstimator()

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[],
      response_vector=[],
      est_theta=1.7,
    )

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

      theta = estimator.estimate(ItemBank(items), list(range(len(items))), list(pattern), 0.0)

      assert theta == pytest.approx(reference, abs=1e-4), pattern

  def test_counters_add_the_refinement_to_the_search_points(self) -> None:
    """Each estimate counts the 201 search points plus the refinement steps, and the totals add up."""
    item_bank = ItemBank.generate_item_bank(20, seed=7)
    estimator = MAPEstimator()

    estimator.estimate(item_bank, [0, 1, 2, 3], [True, False, True, True], 0.0)
    first = estimator.evaluations
    estimator.estimate(item_bank, [0, 1, 2, 3, 4], [True, False, True, True, False], 0.0)

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

    theta = estimator.estimate(self._hard_items(), [0, 1, 2], response_vector, 0.0)

    assert theta == pytest.approx(expected, abs=1e-5)

  def test_estimate_stays_inside_a_prior_narrower_than_a_grid_step(self) -> None:
    """Both neighbours of the only grid point in the support lie outside it.

    The refinement then searches where the objective is mostly +inf; the estimate must
    still lie inside the support rather than wherever that search stops.
    """
    estimator = MAPEstimator(log_prior=uniform_log_prior(-0.01, 0.01))

    theta = estimator.estimate(self._hard_items(), [0, 1, 2], [True, False, True], 0.0)

    assert -0.01 <= theta <= 0.01

  def test_estimate_raises_when_the_prior_excludes_the_bounds(self) -> None:
    """A prior with no support inside the search bounds leaves no mode to find."""
    estimator = MAPEstimator(log_prior=uniform_log_prior(7.0, 8.0))

    with pytest.raises(ValueError, match="not finite at any"):
      estimator.estimate(self._hard_items(), [0], [True], 0.0)

  def test_estimate_locates_an_off_grid_prior_mode_within_tol(self) -> None:
    """With no responses the estimate is the prior mode to within tol, not to the grid's 0.06."""
    estimator = MAPEstimator(log_prior=normal_log_prior(mean=0.5, sd=1.0), tol=1e-8)

    theta = estimator.estimate(self._hard_items(), [], [], 0.0)

    assert theta == pytest.approx(0.5, abs=1e-6)

  def test_estimate_refines_a_mode_between_two_tied_points(self) -> None:
    """A peak halfway between two grid points makes them tie, which is not a flat posterior.

    The prior mode 0.03 sits midway between the points 0.0 and 0.06, which tie exactly;
    the estimate must still be refined to the mode instead of returning one of them.
    """
    estimator = MAPEstimator(log_prior=normal_log_prior(mean=0.03, sd=1.0), tol=1e-8)

    theta = estimator.estimate(self._hard_items(), [], [], 0.0)

    assert theta == pytest.approx(0.03, abs=1e-6)

  def test_estimate_raises_when_the_support_falls_between_grid_points(self) -> None:
    """A support narrower than the grid spacing that holds no grid point cannot be searched."""
    estimator = MAPEstimator(log_prior=uniform_log_prior(0.01, 0.05))

    with pytest.raises(ValueError, match="must contain at least one"):
      estimator.estimate(self._hard_items(), [0], [True], 0.0)

  def test_estimate_finds_the_global_mode_of_a_bimodal_prior(self) -> None:
    """The grid search picks the taller of two separated modes, wherever a local search would stop."""

    def log_prior(theta: np.ndarray) -> np.ndarray:
      low_mode = np.log(0.3) - 0.5 * ((theta + 2.0) / 0.3) ** 2
      high_mode = np.log(0.7) - 0.5 * ((theta - 2.0) / 0.3) ** 2
      return np.logaddexp(low_mode, high_mode)

    estimator = MAPEstimator(log_prior=log_prior)

    theta = estimator.estimate(self._hard_items(), [], [], 0.0)

    assert theta == pytest.approx(2.0, abs=1e-5)

  def test_estimate_takes_the_middle_of_a_flat_posterior(self) -> None:
    """A uniform prior with no responses has no unique mode, so the middle of the bounds is returned."""
    estimator = MAPEstimator(log_prior=uniform_log_prior())

    theta = estimator.estimate(self._hard_items(), [], [], 0.0)

    assert theta == pytest.approx(0.0, abs=1e-12)

  def test_estimate_rejects_responses_without_items(self) -> None:
    """A response vector longer than the administered items is an error, even with no items."""
    with pytest.raises(ValueError, match="same length"):
      MAPEstimator().estimate(self._hard_items(), [], [True], 0.0)


class TestWarmLikelihoodEstimator:
  """Tests for WarmLikelihoodEstimator."""

  def test_init_default(self) -> None:
    """Test default initialization."""
    estimator = WarmLikelihoodEstimator()
    assert estimator.calls == 0
    assert str(estimator) == "Warm Weighted Likelihood Estimator"

  def test_estimate_empty_response_returns_input_theta(self) -> None:
    """No administered items should return the input estimate unchanged."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = WarmLikelihoodEstimator()

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[],
      response_vector=[],
      est_theta=1.25,
    )

    assert theta == pytest.approx(1.25)
    assert estimator.calls == 1

  def test_estimate_all_correct_remains_finite(self) -> None:
    """Warm correction should remain finite when plain MLE diverges."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    administered_items = [0, 1, 2, 3, 4]
    response_vector = [True] * len(administered_items)

    mle = NumericalSearchEstimator(dodd=False)
    mle_theta = mle.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    wle = WarmLikelihoodEstimator()
    wle_theta = wle.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert np.isfinite(wle_theta)
    assert np.isinf(mle_theta)
    assert wle.calls == 1
    assert wle.evaluations > 0
    assert wle.total_evaluations == wle.evaluations

  def test_estimate_all_incorrect_remains_finite(self) -> None:
    """Warm correction should remain finite when plain MLE diverges."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    administered_items = [0, 1, 2, 3, 4]
    response_vector = [False] * len(administered_items)

    mle = NumericalSearchEstimator(dodd=False)
    mle_theta = mle.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    wle = WarmLikelihoodEstimator()
    wle_theta = wle.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert np.isfinite(wle_theta)
    assert np.isinf(mle_theta)
    assert wle.calls == 1
    assert wle.evaluations > 0
    assert wle.total_evaluations == wle.evaluations

  def test_estimate_handles_mixed_responses(self) -> None:
    """WLE should remain finite for ordinary response patterns."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    administered_items = [0, 1, 2, 3, 4]
    response_vector = [True, False, True, True, False]

    estimator = WarmLikelihoodEstimator()
    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert isinstance(theta, float)
    assert np.isfinite(theta)
    assert estimator.calls == 1
    assert estimator.evaluations > 0
    assert estimator.total_evaluations == estimator.evaluations

  def test_estimate_accumulates_counters_across_calls(self) -> None:
    """Test that repeated estimates update the internal call and evaluation counters."""
    item_bank = ItemBank.generate_item_bank(30, seed=123)
    estimator = WarmLikelihoodEstimator()

    first_theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1, 2],
      response_vector=[True, False, True],
      est_theta=0.0,
    )
    first_evaluations = estimator.evaluations

    second_theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1, 2, 3],
      response_vector=[True, False, True, False],
      est_theta=0.0,
    )

    assert estimator.calls == 2
    assert estimator.evaluations > 0
    assert estimator.total_evaluations >= first_evaluations
    assert np.isfinite(first_theta)
    assert np.isfinite(second_theta)

  def test_estimate_records_evaluation_counts_for_mixed_responses(self) -> None:
    """The estimator should track the underlying MLE evaluations on ordinary inputs."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    administered_items = [0, 1, 2, 3, 4]
    response_vector = [True, False, True, True, False]

    estimator = WarmLikelihoodEstimator()
    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert np.isfinite(theta)
    assert estimator.calls == 1
    assert estimator.evaluations > 0
    assert estimator.total_evaluations == estimator.evaluations

  def test_estimate_applies_bias_correction_when_mle_is_finite(self, monkeypatch: pytest.MonkeyPatch) -> None:
    """WLE should add the Warm correction term on top of a finite MLE."""
    item_bank = ItemBank(
      np.array(
        [
          [1.0, -1.0, 0.0, 1.0],
          [1.5, 0.0, 0.0, 1.0],
          [2.0, 1.0, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    administered_items = [0, 1, 2]
    response_vector = [True, False, True]

    estimator = WarmLikelihoodEstimator()

    class StubMLE:
      evaluations = 11

      @staticmethod
      def estimate(
        item_bank: ItemBank,
        administered_items: list[int],
        response_vector: list[bool],
        est_theta: float,
      ) -> float:
        del item_bank, administered_items, response_vector, est_theta
        return 0.75

    monkeypatch.setattr(estimator, "_mle", StubMLE(), raising=False)

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    items = item_bank.items[administered_items, :4]
    a = items[:, 0]
    b = items[:, 1]
    c = items[:, 2]
    d = items[:, 3]
    z = np.exp(-a * (0.75 - b))
    one_plus_z = 1.0 + z
    p = c + (d - c) / one_plus_z
    p_prime = a * (d - c) * z / (one_plus_z**2)
    p_double = a * a * (d - c) * z * (z - 1.0) / (one_plus_z**3)
    denom = np.clip(p * (1.0 - p), np.finfo(float).eps, None)
    info = float(np.sum((p_prime**2) / denom))
    j = float(np.sum((p_prime * p_double) / denom))
    expected = 0.75 if info <= 0 else 0.75 + j / (2.0 * info * info)

    assert theta == pytest.approx(expected)
    assert estimator.calls == 1
    assert estimator.evaluations == 11
    assert estimator.total_evaluations == 11
