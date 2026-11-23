"""Tests for the refactored simulation architecture."""

from __future__ import annotations

import numpy as np
import pytest

from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import BaseSelector, RandomSelector
from catsim.simulation import SimulationRunner
from catsim.state import SessionStatus, SimulationResult
from catsim.stopping import TestLengthStopper as LengthStopper


class FirstRemainingSelector(BaseSelector):
  """Select the lowest-index remaining item and record the exposure rates shown at each call."""

  def __init__(self) -> None:
    super().__init__()
    self.seen_exposure_rates: list[np.ndarray] = []

  def select(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: np.random.Generator | None = None,
    exposure_rates: np.ndarray | None = None,
    response_vector: list[bool] | None = None,
  ) -> int:
    del est_theta, rng, response_vector
    assert exposure_rates is not None, "SimulationRunner should always pass exposure rates"
    self.seen_exposure_rates.append(exposure_rates.copy())
    return next(i for i in range(item_bank.n_items) if i not in administered_items)


class TestSimulationRunnerInit:
  """Tests for SimulationRunner initialization."""

  def test_init_with_item_bank(self, item_bank: ItemBank) -> None:
    """Runner should accept an ItemBank directly."""
    runner = SimulationRunner(
      item_bank,
      FixedPointInitializer(0.0),
      RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=LengthStopper(max_items=5),
      seed=42,
    )
    assert runner.item_bank is item_bank

  def test_init_with_numpy_array(self) -> None:
    """Runner should auto-convert numpy arrays into ItemBank."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.2, 0.5, 0.0, 1.0]])
    runner = SimulationRunner(
      items,
      FixedPointInitializer(0.0),
      RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=LengthStopper(max_items=5),
      seed=42,
    )
    assert isinstance(runner.item_bank, ItemBank)
    assert runner.item_bank.n_items == 2

  def test_init_with_invalid_item_bank_raises(self) -> None:
    """Runner should reject invalid item bank objects."""
    with pytest.raises(TypeError, match=r"must be an ItemBank or numpy\.ndarray") as exc_info:
      SimulationRunner(  # type: ignore[arg-type]
        [[1.0, 0.0, 0.0, 1.0]],
        FixedPointInitializer(0.0),
        RandomSelector(),
        estimator=NumericalSearchEstimator(),
        stopper=LengthStopper(max_items=5),
      )

    assert str(exc_info.value) == "item_bank must be an ItemBank or numpy.ndarray"


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
    with pytest.raises(TypeError, match="one-dimensional") as exc_info:
      simulation_runner.run(np.array([[0.0, 1.0]]))

    assert str(exc_info.value) == "Examinees array must be one-dimensional"

  def test_run_seed_reproducibility(self, item_bank: ItemBank) -> None:
    """Same seed and inputs should produce the same latest estimations."""
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

  def test_verbose_run_matches_a_quiet_run(self, item_bank: ItemBank) -> None:
    """verbose=True only adds a progress bar: the run completes and gives the same sessions as a quiet run."""

    def run(verbose: bool) -> SimulationResult:
      runner = SimulationRunner(
        item_bank, FixedPointInitializer(0.0), RandomSelector(), NumericalSearchEstimator(), LengthStopper(max_items=3)
      )
      return runner.run(3, verbose=verbose)

    quiet, verbose = run(verbose=False), run(verbose=True)

    assert verbose.administered_items == quiet.administered_items
    assert verbose.latest_estimations == pytest.approx(quiet.latest_estimations)

  def test_each_session_sees_the_exposure_of_the_sessions_before_it(self, item_bank: ItemBank) -> None:
    """Selectors see the share of examinees in the whole run who were already given each item.

    Exposure control (``r_max``) works only if the tracker is shared across sessions and divided by the
    number of examinees in the run.
    """
    selector = FirstRemainingSelector()
    runner = SimulationRunner(
      item_bank, FixedPointInitializer(0.0), selector, NumericalSearchEstimator(), LengthStopper(max_items=2)
    )

    result = runner.run(np.zeros(4))

    # every session administers items 0 and 1, so the selector runs twice per session
    first_calls = selector.seen_exposure_rates[::2]
    assert [rates[:2].tolist() for rates in first_calls] == [[0.0, 0.0], [0.25, 0.25], [0.5, 0.5], [0.75, 0.75]]
    assert not any(rates[2:].any() for rates in first_calls)
    assert result.exposure_rates[:2].tolist() == [1.0, 1.0]


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

  def test_exposure_rates_sum_to_the_test_length(self, simulation_result: SimulationResult) -> None:
    """With one rate per item, and every examinee given 10 items, the rates sum to 10."""
    assert simulation_result.exposure_rates.shape == (simulation_result.item_bank.n_items,)
    assert simulation_result.exposure_rates.sum() == pytest.approx(10.0)

  def test_metrics_compare_final_estimates_with_true_abilities(self, simulation_result: SimulationResult) -> None:
    """Bias is the mean of estimate minus true ability, MSE the mean squared error, RMSE its square root."""
    errors = np.asarray(simulation_result.latest_estimations) - simulation_result.examinees

    assert simulation_result.bias == pytest.approx(errors.mean())
    assert simulation_result.mse == pytest.approx(np.mean(errors**2))
    assert simulation_result.rmse == pytest.approx(np.sqrt(np.mean(errors**2)))
