"""Tests for catsim.estimation module."""

from itertools import pairwise

import numpy as np
import pytest

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
from catsim.item_bank import ItemBank


class TestNumericalSearchEstimatorInit:
  """Tests for NumericalSearchEstimator initialization."""

  def test_init_default(self) -> None:
    """Test default initialization."""
    estimator = NumericalSearchEstimator()
    assert estimator.calls == 0
    assert estimator.evaluations == 0

  def test_init_with_method(self) -> None:
    """Test initialization with specific method."""
    estimator = NumericalSearchEstimator(method="brent")
    assert str(estimator) == "Numerical Search Estimator (brent)"

  def test_init_with_invalid_method_raises(self) -> None:
    """Test that invalid method raises ValueError."""
    with pytest.raises(ValueError, match="must be one of"):
      NumericalSearchEstimator(method="invalid_method")


class TestNumericalSearchEstimatorMethods:
  """Tests for NumericalSearchEstimator available methods."""

  def test_available_methods_returns_frozenset(self) -> None:
    """Test that available_methods returns a frozenset."""
    methods = NumericalSearchEstimator.available_methods()
    assert isinstance(methods, frozenset)

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
    # Only test stable methods that are known to produce consistent results
    ["bounded", "dichotomous", "fibonacci", "ternary", "golden2"],
  )
  def test_stable_methods_produce_similar_results(self, method: str) -> None:
    """Test that stable methods produce similar results for the same input."""
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

    # Stable methods should produce results within reasonable range of each other
    assert abs(theta - ref_theta) < 1.0, f"Method {method} produced {theta}, expected ~{ref_theta}"


class TestBaseEstimatorAbstract:
  """Tests for BaseEstimator abstract base class."""

  def test_cannot_instantiate_base_estimator(self) -> None:
    """Test that BaseEstimator cannot be instantiated directly."""
    with pytest.raises(TypeError):
      BaseEstimator()  # type: ignore[abstract]

  def test_subclass_must_implement_estimate(self) -> None:
    """Test that subclass must implement estimate method."""

    class IncompleteEstimator(BaseEstimator):
      pass

    with pytest.raises(TypeError):
      IncompleteEstimator()  # type: ignore[abstract]


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

  def test_uniform_log_prior_rejects_inverted_bounds(self) -> None:
    """Uniform priors require a strictly increasing interval."""
    with pytest.raises(ValueError, match="high must be greater than low"):
      uniform_log_prior(1.0, 1.0)
    with pytest.raises(ValueError, match="high must be greater than low"):
      uniform_log_prior(2.0, 1.0)

  def test_quadrature_grid_uniform_defaults(self) -> None:
    """Test the default uniform quadrature grid."""
    grid = QuadratureGrid.uniform()

    assert len(grid.nodes) == 41
    assert len(grid.weights) == 41
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
    assert estimator.last_posterior is not None
    assert np.isclose(estimator.last_posterior.sum(), 1.0)
    assert estimator.last_posterior_variance() > 0

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
    assert estimator.last_posterior is not None
    assert estimator.calls == 1
    assert estimator.last_posterior_variance() > 0

  def test_last_posterior_variance_is_infinite_before_first_estimate(self) -> None:
    """The variance accessor should signal that no posterior has been computed yet."""
    estimator = EAPEstimator()

    assert estimator.last_posterior is None
    assert estimator.last_posterior_variance() == np.inf

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

  def test_last_posterior_variance_decreases_as_items_accumulate(self) -> None:
    """Each additional item should reduce the posterior variance overall.

    This protects the convergence signal exposed to downstream stopping and
    selection rules; broken posterior normalization could keep variance flat or rising.
    """
    rng = np.random.default_rng(7)
    item_bank = ItemBank.generate_item_bank(20, seed=3)
    theta_true = 0.8
    estimator = EAPEstimator()

    administered_items: list[int] = []
    response_vector: list[bool] = []
    variances: list[float] = []

    for idx in range(10):
      administered_items.append(idx)
      probability = irt.icc(theta_true, *item_bank.items[idx, :4])
      response_vector.append(bool(rng.random() < probability))
      estimator.estimate(
        item_bank=item_bank,
        administered_items=administered_items,
        response_vector=response_vector,
        est_theta=0.0,
      )
      variances.append(estimator.last_posterior_variance())

    assert variances[-1] < variances[0]
    assert variances[4] < variances[0]

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
    assert estimator.last_posterior is not None
    assert estimator.last_posterior.shape == (21,)
    assert np.isclose(estimator.last_posterior.sum(), 1.0)

  def test_last_posterior_updates_between_calls(self) -> None:
    """Test that the estimator state tracks the most recent posterior only."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = EAPEstimator()

    estimator.estimate(
      item_bank=item_bank,
      administered_items=[0],
      response_vector=[True],
      est_theta=0.0,
    )
    first_posterior = estimator.last_posterior
    assert first_posterior is not None

    estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1],
      response_vector=[True, False],
      est_theta=0.0,
    )

    assert estimator.last_posterior is not None
    assert estimator.last_posterior is not first_posterior
    assert np.isclose(estimator.last_posterior.sum(), 1.0)
    assert estimator.last_posterior_variance() != pytest.approx(
      posterior_variance(first_posterior, estimator.grid.nodes)
    )

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


class TestMAPEstimator:
  """Tests for the Maximum a Posteriori estimator."""

  def test_init_default(self) -> None:
    """Test default initialization."""
    estimator = MAPEstimator()

    assert str(estimator) == "Maximum a Posteriori Estimator"
    assert estimator.calls == 0
    assert estimator.evaluations == 0

  def test_estimate_empty_response_vector_returns_prior_mode(self) -> None:
    """Test that the prior mode is returned when there is no data."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = MAPEstimator()

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[],
      response_vector=[],
      est_theta=1.7,
    )

    assert theta == pytest.approx(0.0)
    assert estimator.calls == 1
    assert estimator.evaluations == 0

  def test_estimate_all_correct_is_finite(self) -> None:
    """Test that MAP remains finite for an extreme all-correct response pattern."""
    item_bank = ItemBank(
      np.array(
        [
          [1.2, 2.0, 0.0, 1.0],
          [1.4, 2.5, 0.0, 1.0],
          [1.1, 3.0, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    estimator = MAPEstimator()

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1, 2],
      response_vector=[True, True, True],
      est_theta=0.0,
    )

    assert np.isfinite(theta)
    assert theta > 0.0

    mle = NumericalSearchEstimator(dodd=False)
    mle_theta = mle.estimate(
      item_bank=item_bank,
      administered_items=[0, 1, 2],
      response_vector=[True, True, True],
      est_theta=0.0,
    )
    assert np.isposinf(mle_theta)

  def test_estimate_uses_custom_prior_mode_when_empty(self) -> None:
    """Test that a shifted prior changes the no-data MAP estimate."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = MAPEstimator(log_prior=normal_log_prior(mean=1.5, sd=0.5))

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[],
      response_vector=[],
      est_theta=0.0,
    )

    assert theta == pytest.approx(1.5, abs=1e-12)
    assert estimator.calls == 1
    assert estimator.evaluations == 0

  def test_estimate_tracks_evaluation_counts_across_calls(self) -> None:
    """Test that the estimator counters reflect repeated posterior maximization calls."""
    item_bank = ItemBank.generate_item_bank(20, seed=7)
    estimator = MAPEstimator()

    first_theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1, 2, 3],
      response_vector=[True, False, True, True],
      est_theta=0.0,
    )
    first_evaluations = estimator.evaluations

    second_theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1, 2, 3, 4],
      response_vector=[True, False, True, True, False],
      est_theta=0.0,
    )

    assert estimator.calls == 2
    assert estimator.evaluations > 0
    assert estimator.total_evaluations >= first_evaluations
    assert np.isfinite(first_theta)
    assert np.isfinite(second_theta)

  def test_estimate_all_incorrect_is_finite(self) -> None:
    """Test that MAP remains finite for an extreme all-incorrect response pattern."""
    item_bank = ItemBank(
      np.array(
        [
          [1.2, -2.0, 0.0, 1.0],
          [1.4, -2.5, 0.0, 1.0],
          [1.1, -3.0, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    estimator = MAPEstimator()

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1, 2],
      response_vector=[False, False, False],
      est_theta=0.0,
    )

    assert np.isfinite(theta)
    assert theta < 0.0

  def test_estimate_records_evaluation_counts_for_mixed_responses(self) -> None:
    """Mixed response patterns should update the evaluation counters."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    estimator = MAPEstimator()

    theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=[0, 1, 2, 3, 4],
      response_vector=[True, False, True, True, False],
      est_theta=0.0,
    )

    assert np.isfinite(theta)
    assert estimator.calls == 1
    assert estimator.evaluations > 0
    assert estimator.total_evaluations == estimator.evaluations

  def test_estimate_is_pulled_toward_prior_mean_vs_mle(self) -> None:
    """MAP should move the estimate toward the prior mean relative to the MLE.

    This captures MAP's defining short-test behavior; if it matched MLE here,
    the implementation would be ignoring prior information in the case that matters.
    """
    item_bank = ItemBank(
      np.array(
        [
          [1.5, 2.0, 0.0, 1.0],
          [1.2, 2.5, 0.0, 1.0],
          [1.8, 1.5, 0.0, 1.0],
          [1.0, 3.0, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    administered_items = [0, 1, 2, 3]
    response_vector = [True, True, True, False]

    mle = NumericalSearchEstimator(dodd=False)
    mle_theta = mle.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    estimator = MAPEstimator(log_prior=normal_log_prior(mean=0.0, sd=1.0))
    map_theta = estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert np.isfinite(mle_theta)
    assert np.isfinite(map_theta)
    assert mle_theta > 0.0
    assert map_theta > 0.0
    assert abs(map_theta) < abs(mle_theta)

  def test_estimate_and_posterior_mean_agree_on_a_long_test(self) -> None:
    """MAP and posterior mean should agree on an information-rich test.

    On a smooth concentrated posterior, the mode and mean should nearly coincide,
    so this cross-check catches bugs in either posterior scoring or optimization.
    """
    item_bank = ItemBank.generate_item_bank(30, seed=5)
    theta_true = 0.3
    rng = np.random.default_rng(42)
    administered_items = list(range(20))
    response_vector = [
      bool(rng.random() < posterior_prob)
      for posterior_prob in [irt.icc(theta_true, *item_bank.items[idx, :4]) for idx in administered_items]
    ]
    log_prior = normal_log_prior()

    map_estimator = MAPEstimator(log_prior=log_prior)
    map_theta = map_estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )
    grid = QuadratureGrid.uniform()
    posterior_mass = posterior(response_vector, item_bank.items[administered_items, :4], grid, log_prior)
    eap_theta = posterior_mean(posterior_mass, grid.nodes)

    assert np.isfinite(map_theta)
    assert np.isfinite(eap_theta)
    assert abs(map_theta - eap_theta) < 0.15

  def test_estimate_shifted_prior_pulls_post_data_map_estimate(self) -> None:
    """A shifted prior should move the post-data MAP estimate toward its mean.

    The prior must still influence the estimate after data arrive; otherwise the
    no-data default would work while real Bayesian shrinkage silently failed.
    """
    item_bank = ItemBank.generate_item_bank(10, seed=9)
    administered_items = [0, 1, 2, 3]
    response_vector = [True, False, True, False]

    default_estimator = MAPEstimator()
    default_theta = default_estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    shifted_estimator = MAPEstimator(log_prior=normal_log_prior(mean=2.0, sd=1.0))
    shifted_theta = shifted_estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=0.0,
    )

    assert np.isfinite(default_theta)
    assert np.isfinite(shifted_theta)
    assert shifted_theta > default_theta


class TestWarmLikelihoodEstimator:
  """Tests for WarmLikelihoodEstimator."""

  def test_init_default(self) -> None:
    """Test default initialization."""
    estimator = WarmLikelihoodEstimator()
    assert estimator.calls == 0
    assert str(estimator) == "Warm Weighted Likelihood Estimator"

  @pytest.mark.xfail(
    strict=True,
    reason="Current Warm WLE branch does not yet show lower short-test bias than MLE on a reproducible simulation.",
  )
  def test_estimate_has_lower_bias_than_mle_on_short_tests(self) -> None:
    """WLE absolute error should be smaller than MLE on average for a 5-item test.

    This xfail documents Warm's headline short-test claim, so a plausible-looking
    correction that moves in the wrong direction cannot hide behind formula-only tests.
    """
    rng = np.random.default_rng(42)
    item_bank = ItemBank.generate_item_bank(100, seed=0)
    mle = NumericalSearchEstimator(dodd=True)
    wle = WarmLikelihoodEstimator()

    theta_true = 1.5
    administered_items = list(range(5))
    items = item_bank.items[administered_items, :4]

    mle_errors: list[float] = []
    wle_errors: list[float] = []
    for _ in range(500):
      response_vector = [
        bool(rng.random() < irt.icc(theta_true, *items[idx])) for idx in range(len(administered_items))
      ]
      if all(response_vector) or not any(response_vector):
        continue

      mle_theta = mle.estimate(
        item_bank=item_bank,
        administered_items=administered_items,
        response_vector=response_vector,
        est_theta=theta_true,
      )
      wle_theta = wle.estimate(
        item_bank=item_bank,
        administered_items=administered_items,
        response_vector=response_vector,
        est_theta=theta_true,
      )
      if np.isfinite(mle_theta) and np.isfinite(wle_theta):
        mle_errors.append(abs(mle_theta - theta_true))
        wle_errors.append(abs(wle_theta - theta_true))

    assert mle_errors
    assert wle_errors
    assert float(np.mean(wle_errors)) < float(np.mean(mle_errors))

  def test_estimate_pulls_toward_center_relative_to_mle(self) -> None:
    """Warm correction should reduce the outward bias of a mixed-response MLE.

    This is the direction-of-correction sanity check: the estimate should move
    back toward the population center rather than farther away.
    """
    item_bank = ItemBank(
      np.array(
        [
          [1.5, 1.5, 0.0, 1.0],
          [1.2, 1.8, 0.0, 1.0],
          [1.8, 1.2, 0.0, 1.0],
          [1.0, 2.0, 0.0, 1.0],
          [2.0, 1.0, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    administered_items = [0, 1, 2, 3, 4]
    response_vector = [True, True, True, True, False]

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

    assert np.isfinite(mle_theta)
    assert np.isfinite(wle_theta)
    assert mle_theta > 0.0
    assert wle_theta > 0.0
    assert abs(wle_theta) < abs(mle_theta)

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
