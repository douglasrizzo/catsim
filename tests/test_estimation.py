"""Tests for catsim.estimation module."""

from itertools import pairwise

import numpy as np
import pytest
from scipy.optimize import minimize_scalar

from catsim import irt
from catsim.estimation import (
  EAPEstimator,
  NumericalSearchEstimator,
  QuadratureGrid,
  normal_log_prior,
  posterior,
  posterior_mean,
  posterior_variance,
  uniform_log_prior,
)
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
    expected = {
      "ternary",
      "dichotomous",
      "fibonacci",
      "golden",
      "brent",
      "bounded",
      "golden2",
    }
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

  @pytest.mark.parametrize(
    "method",
    sorted(NumericalSearchEstimator.available_methods()),
  )
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

  def test_estimate_returns_prior_mean_with_no_data(self) -> None:
    """Test that an empty response vector returns the prior mean."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = EAPEstimator()

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[],
      response_vector=[],
      est_theta=0.0,
    )

    assert theta == pytest.approx(0.0, abs=1e-12)

  def test_estimate_uses_custom_prior_mean_with_no_data(self) -> None:
    """Custom priors should control the posterior mean when there is no data."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = EAPEstimator(log_prior=normal_log_prior(mean=1.2, sd=0.5))

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[],
      response_vector=[],
      est_theta=0.0,
    )

    assert theta == pytest.approx(1.2, abs=1e-12)
    assert estimator.calls == 1

  def test_estimate_shifts_positive_after_correct_response(self) -> None:
    """Test that a correct response on a difficult item moves EAP upward."""
    item_bank = ItemBank(np.array([[1.5, 2.0, 0.0, 1.0]], dtype=float))
    estimator = EAPEstimator()

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[0],
      response_vector=[True],
      est_theta=0.0,
    )

    assert theta > 0.0

  def test_estimate_is_pulled_toward_prior_on_extreme_short_pattern(self) -> None:
    """A centered prior should shrink EAP more than a flat prior on all-correct data.

    This checks the core Bayesian behavior on sparse data, where prior shrinkage
    matters most and a purely likelihood-driven implementation would look too extreme.
    """
    item_bank = ItemBank(
      np.array(
        [
          [1.5, 2.0, 0.0, 1.0],
          [1.2, 2.5, 0.0, 1.0],
          [1.8, 1.5, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    administered_items = [0, 1, 2]
    response_vector = [True, True, True]

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

    assert normal_theta > 0.0
    assert normal_theta < 5.0
    assert abs(normal_theta) < abs(flat_theta)

  def test_estimate_uses_custom_grid_and_prior(self) -> None:
    """Test that custom grid and prior parameters shape the posterior as expected."""
    grid = QuadratureGrid.uniform(n_nodes=21, low=-3.0, high=3.0)
    estimator = EAPEstimator(grid=grid, log_prior=normal_log_prior(mean=1.5, sd=0.5))
    item_bank = ItemBank.generate_item_bank(5, seed=7)

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[],
      response_vector=[],
      est_theta=0.0,
    )

    assert theta == pytest.approx(1.5, abs=0.05)
    assert estimator.grid is grid

  def test_estimate_converges_toward_mle_on_longer_test(self) -> None:
    """Test that EAP tracks MLE on a longer, information-rich response pattern."""
    item_bank = ItemBank.generate_item_bank(30, seed=1)
    theta_true = 0.25
    administered_items = list(range(30))
    response_vector = [irt.icc(theta_true, *item_bank.items[i, :4]) >= 0.5 for i in administered_items]

    mle = NumericalSearchEstimator()
    mle_theta = mle.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    estimator = EAPEstimator(
      grid=QuadratureGrid.uniform(n_nodes=81),
      log_prior=uniform_log_prior(),
    )
    eap_theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert np.isfinite(mle_theta)
    assert np.isfinite(eap_theta)
    assert abs(eap_theta - mle_theta) < 0.3
