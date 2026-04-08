"""Tests for runtime state and aggregate simulation result models."""

from __future__ import annotations

import numpy as np

from catsim.state import CatSessionState, ExposureTracker, SessionStatus, SimulationResult


def test_session_state_initializes_theta_history_and_count() -> None:
  """Sessions should seed theta history from the initial theta."""
  session = CatSessionState(session_id=7, current_theta=1.25, status=SessionStatus.IN_PROGRESS)

  assert session.theta_history == [1.25]
  assert session.latest_theta == 1.25
  assert session.administered_count == 0


def test_exposure_tracker_returns_zero_rates_for_non_positive_session_counts() -> None:
  """Exposure rates should stay zero when no sessions have been completed."""
  tracker = ExposureTracker(4)
  tracker.record(1)
  tracker.record(1)

  assert np.array_equal(tracker.rates(0), np.zeros(4))
  assert np.array_equal(tracker.rates(-1), np.zeros(4))


def test_simulation_result_uses_zero_for_missing_true_theta(item_bank) -> None:
  """Manual sessions without true theta should still expose examinee vectors."""
  sessions = [
    CatSessionState(session_id=0, current_theta=0.2, true_theta=None, theta_history=[0.2]),
    CatSessionState(session_id=1, current_theta=-0.4, true_theta=1.0, theta_history=[-0.4]),
  ]
  result = SimulationResult(
    item_bank=item_bank,
    sessions=sessions,
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.1,
  )

  assert np.array_equal(result.examinees, np.array([0.0, 1.0]))


def test_simulation_result_empty_sessions_exposure_and_overlap(item_bank) -> None:
  """Empty simulation results should expose safe default aggregates."""
  result = SimulationResult(
    item_bank=item_bank,
    sessions=[],
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.0,
  )

  assert np.array_equal(result.exposure_rates, np.zeros(item_bank.n_items))
  assert result.overlap_rate is None


def test_simulation_result_overlap_requires_uniform_non_zero_test_lengths(item_bank) -> None:
  """Overlap is undefined for heterogeneous or zero-length sessions."""
  heterogeneous_sessions = [
    CatSessionState(session_id=0, current_theta=0.0, administered_item_ids=[0], theta_history=[0.0]),
    CatSessionState(session_id=1, current_theta=0.0, administered_item_ids=[0, 1], theta_history=[0.0]),
  ]
  heterogeneous = SimulationResult(
    item_bank=item_bank,
    sessions=heterogeneous_sessions,
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.0,
  )

  zero_length_sessions = [
    CatSessionState(session_id=0, current_theta=0.0, theta_history=[0.0]),
    CatSessionState(session_id=1, current_theta=0.0, theta_history=[0.0]),
  ]
  zero_length = SimulationResult(
    item_bank=item_bank,
    sessions=zero_length_sessions,
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.0,
  )

  assert heterogeneous.overlap_rate is None
  assert zero_length.overlap_rate is None
