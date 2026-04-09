"""Tests for catsim.estimation module."""

import numpy as np
import pytest

from catsim.estimation import (
  BaseEstimator,
  NumericalSearchEstimator,
  QuadratureGrid,
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

  def test_init_with_tolerance(self) -> None:
    """Test initialization with custom tolerance."""
    estimator = NumericalSearchEstimator(tol=1e-8)
    # Tolerance is set internally
    assert estimator is not None

  def test_init_with_dodd_disabled(self) -> None:
    """Test initialization with Dodd heuristic disabled."""
    estimator = NumericalSearchEstimator(dodd=False)
    assert estimator is not None


class TestNumericalSearchEstimatorMethods:
  """Tests for NumericalSearchEstimator available methods."""

  def test_available_methods_returns_frozenset(self) -> None:
    """Test that available_methods returns a frozenset."""
    methods = NumericalSearchEstimator.available_methods()
    assert isinstance(methods, frozenset)

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

  def test_quadrature_grid_uniform_defaults(self) -> None:
    """Test the default uniform quadrature grid."""
    grid = QuadratureGrid.uniform()

    assert len(grid.nodes) == 41
    assert len(grid.weights) == 41
    assert grid.nodes[0] == pytest.approx(-6.0)
    assert grid.nodes[-1] == pytest.approx(6.0)
    assert np.allclose(grid.weights, grid.weights[0])

  def test_posterior_normalizes_to_one(self) -> None:
    """Test that the discrete posterior is normalized."""
    grid = QuadratureGrid.uniform()
    items = np.array([[1.5, 1.5, 0.0, 1.0]], dtype=float)

    post = posterior([True], items, grid, normal_log_prior())

    assert np.isclose(post.sum(), 1.0)

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
