"""Tests for the refactored simulation architecture."""

from __future__ import annotations

import numpy as np
import pytest

from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import RandomSelector
from catsim.simulation import SimulationRunner
from catsim.state import SessionStatus, SimulationResult
from catsim.stopping import TestLengthStopper as LengthStopper


class TestSimulationRunnerInit:
  """Tests for SimulationRunner initialization."""

  def test_init_with_item_bank(self, item_bank: ItemBank) -> None:
    """Runner should accept an ItemBank directly."""
    runner = SimulationRunner(
      item_bank,
      FixedPointInitializer(0.0),
      RandomSelector(),
      estimator=pytest.importorskip("catsim.estimation").NumericalSearchEstimator(),
      stopper=LengthStopper(max_items=5),
      seed=42,
    )
    assert runner.item_bank is item_bank

  def test_init_with_numpy_array(self) -> None:
    """Runner should auto-convert numpy arrays into ItemBank."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.2, 0.5, 0.0, 1.0]])
    estimator_mod = pytest.importorskip("catsim.estimation")
    runner = SimulationRunner(
      items,
      FixedPointInitializer(0.0),
      RandomSelector(),
      estimator=estimator_mod.NumericalSearchEstimator(),
      stopper=LengthStopper(max_items=5),
      seed=42,
    )
    assert isinstance(runner.item_bank, ItemBank)
    assert runner.item_bank.n_items == 2

  def test_init_with_invalid_item_bank_raises(self) -> None:
    """Runner should reject invalid item bank objects."""
    estimator_mod = pytest.importorskip("catsim.estimation")
    with pytest.raises(TypeError, match=r"must be an ItemBank or numpy\.ndarray"):
      SimulationRunner(  # type: ignore[arg-type]
        [[1.0, 0.0, 0.0, 1.0]],
        FixedPointInitializer(0.0),
        RandomSelector(),
        estimator=estimator_mod.NumericalSearchEstimator(),
        stopper=LengthStopper(max_items=5),
      )


class TestSimulationRunnerRun:
  """Tests for SimulationRunner.run()."""

  def test_run_with_integer_examinees_returns_result(self, simulation_runner: SimulationRunner) -> None:
    """Running with an examinee count should return a SimulationResult."""
    result = simulation_runner.run(5)
    assert isinstance(result, SimulationResult)
    assert len(result.sessions) == 5

  def test_run_with_explicit_examinees_preserves_values(self, simulation_runner: SimulationRunner) -> None:
    """Running with explicit theta values should preserve them in the result."""
    thetas = np.array([-1.0, 0.0, 1.0])
    result = simulation_runner.run(thetas)
    assert np.allclose(result.examinees, thetas)

  def test_run_with_invalid_examinee_count_raises(self, simulation_runner: SimulationRunner) -> None:
    """Non-positive examinee counts should fail."""
    with pytest.raises(ValueError, match="must be positive"):
      simulation_runner.run(0)

  def test_run_with_empty_examinee_array_raises(self, simulation_runner: SimulationRunner) -> None:
    """Empty examinee arrays should fail."""
    with pytest.raises(ValueError, match="cannot be empty"):
      simulation_runner.run([])

  def test_run_with_multidimensional_examinees_raises(self, simulation_runner: SimulationRunner) -> None:
    """Only one-dimensional examinee arrays are valid."""
    with pytest.raises(TypeError, match="one-dimensional"):
      simulation_runner.run(np.array([[0.0, 1.0]]))

  def test_run_seed_reproducibility(self, item_bank: ItemBank) -> None:
    """Same seed and inputs should produce the same latest estimations."""
    from catsim.estimation import NumericalSearchEstimator

    runner1 = SimulationRunner(
      item_bank,
      FixedPointInitializer(0.0),
      RandomSelector(),
      NumericalSearchEstimator(),
      LengthStopper(max_items=5),
      seed=42,
    )
    runner2 = SimulationRunner(
      item_bank,
      FixedPointInitializer(0.0),
      RandomSelector(),
      NumericalSearchEstimator(),
      LengthStopper(max_items=5),
      seed=42,
    )

    result1 = runner1.run(5)
    result2 = runner2.run(5)

    assert result1.latest_estimations == pytest.approx(result2.latest_estimations)
    assert np.array_equal(result1.exposure_counts, result2.exposure_counts)

  def test_run_populates_sessions(self, simulation_result: SimulationResult) -> None:
    """Completed runs should produce stopped sessions with histories."""
    assert len(simulation_result.sessions) == 5
    assert all(session.status == SessionStatus.STOPPED for session in simulation_result.sessions)
    assert all(len(session.theta_history) >= 2 for session in simulation_result.sessions)
    assert all(len(session.administered_item_ids) == 10 for session in simulation_result.sessions)


class TestSimulationResult:
  """Tests for SimulationResult derived properties."""

  def test_latest_estimations_matches_session_thetas(self, simulation_result: SimulationResult) -> None:
    """Latest estimations should match each session's final theta."""
    assert simulation_result.latest_estimations == [session.latest_theta for session in simulation_result.sessions]

  def test_estimations_returns_theta_histories(self, simulation_result: SimulationResult) -> None:
    """Estimations should expose one theta-history per session."""
    assert len(simulation_result.estimations) == len(simulation_result.sessions)
    assert all(len(history) >= 2 for history in simulation_result.estimations)

  def test_response_vectors_have_expected_length(self, simulation_result: SimulationResult) -> None:
    """Response vectors should match administered item counts."""
    for responses, items in zip(simulation_result.response_vectors, simulation_result.administered_items, strict=False):
      assert len(responses) == len(items)

  def test_exposure_rates_shape_matches_item_bank(self, simulation_result: SimulationResult) -> None:
    """Exposure rates should have one value per item."""
    assert simulation_result.exposure_rates.shape == (simulation_result.item_bank.n_items,)

  def test_overlap_rate_for_fixed_length_run(self, simulation_result: SimulationResult) -> None:
    """Fixed-length simulations should produce an overlap rate."""
    assert simulation_result.overlap_rate is not None

  def test_metrics_are_floats(self, simulation_result: SimulationResult) -> None:
    """Bias, MSE, and RMSE should be available as float metrics."""
    assert isinstance(simulation_result.bias, float)
    assert isinstance(simulation_result.mse, float)
    assert isinstance(simulation_result.rmse, float)
