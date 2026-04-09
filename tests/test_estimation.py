"""Tests for catsim.estimation module."""

import numpy as np
import pytest

from catsim.estimation import BaseEstimator, NumericalSearchEstimator, WarmLikelihoodEstimator
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
