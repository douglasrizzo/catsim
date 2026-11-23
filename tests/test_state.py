"""Tests for runtime state and aggregate simulation result models."""

from __future__ import annotations

import json

import numpy as np
import pytest

from catsim.item_bank import ItemBank
from catsim.state import CatSessionState, ExposureTracker, SimulationResult, StopReason


def finished_session(session_id: int, items: list[int]) -> CatSessionState:
  """Return a session that administered ``items``, all answered correctly, with a constant estimate."""
  return CatSessionState(
    session_id=session_id,
    theta_history=[0.0] * (len(items) + 1),
    administered_item_ids=items,
    responses=[True] * len(items),
  )


@pytest.mark.parametrize(
  ("items", "responses", "theta_history", "match"),
  [
    pytest.param([0, 1], [True], [0.0, 0.5], "one response per administered item", id="item-without-response"),
    pytest.param([0], [True], [0.0], "one estimate per response", id="missing-estimate"),
    pytest.param([0], [True], [0.0, 0.5, 0.7], "one estimate per response", id="extra-estimate"),
    pytest.param([], [], [], "the initial estimate", id="no-initial-estimate"),
  ],
)
def test_session_state_rejects_a_history_that_does_not_match_its_responses(
  items: list[int], responses: list[bool], theta_history: list[float], match: str
) -> None:
  """A session holds one response per item and one estimate more than responses.

  Fails if such a session is accepted: its current estimate, the last entry of the history, would not
  be the estimate after its last response (or, with an empty history, would not exist).
  """
  with pytest.raises(ValueError, match=match):
    CatSessionState(session_id=0, theta_history=theta_history, administered_item_ids=items, responses=responses)


def test_session_state_turns_a_persisted_stop_reason_back_into_the_enum() -> None:
  """A stop reason restored from storage as its string is the StopReason member again; an unknown one is rejected.

  Fails if the string is kept, so that ``stop_reason is StopReason.STOPPER`` is False for a restored session, or if
  a reason catsim never sets is accepted.
  """
  persisted = json.loads(json.dumps({"stop_reason": StopReason.STOPPER, "unknown": "paused"}))

  restored = CatSessionState(session_id=0, theta_history=[0.0], stop_reason=persisted["stop_reason"])

  assert restored.stop_reason is StopReason.STOPPER
  with pytest.raises(ValueError, match="'paused' is not a valid StopReason"):
    CatSessionState(session_id=0, theta_history=[0.0], stop_reason=persisted["unknown"])


def test_exposure_tracker_gives_each_session_the_rates_of_the_sessions_before_it() -> None:
  """A session sees each item's share of the sessions started before it, as catR does, from start to end.

  Three sessions administer [0, 1], [0, 2] and [0, 3] on a 4-item bank, and a fourth starts. Fails if the first
  session sees anything but zeros, if the rates divide by a count that includes the session in progress (the
  second session would see 0.5 for items 0 and 1), or if a session's own administrations change the rates of the
  items it can still be offered.
  """
  tracker = ExposureTracker(4)
  assert tracker.rates().tolist() == [0.0, 0.0, 0.0, 0.0]

  at_start, offered_at_end = [], []
  for items in ([0, 1], [0, 2], [0, 3], []):
    tracker.start_session()
    at_start.append(tracker.rates())
    for item_id in items:
      tracker.record(item_id)
    offered_at_end.append(np.delete(tracker.rates(), items))

  expected = [[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0], [1.0, 0.5, 0.5, 0.0], [1.0, 1 / 3, 1 / 3, 1 / 3]]
  assert np.allclose(at_start, expected)
  for rates, offered, items in zip(expected, offered_at_end, ([0, 1], [0, 2], [0, 3], []), strict=True):
    assert np.allclose(offered, np.delete(rates, items))
  assert tracker.n_sessions == 4
  assert tracker.counts.tolist() == [3, 1, 1, 1]


def test_exposure_tracker_gives_sessions_side_by_side_each_others_administrations_over_the_others() -> None:
  """Sessions that run side by side see each other's administrations as they happen, never counting themselves.

  Session P gives item 0, then A starts, then B starts and gives item 1, all on one tracker. A is selected
  against item 0 from P and item 1 from B, each over the two other sessions. Fails if the rates stay as they
  were when the latest session started ([0.5, 0, 0], item 1 missing), or if A is in its own denominator.
  """
  tracker = ExposureTracker(3)
  tracker.start_session()
  tracker.record(0)
  tracker.start_session()
  assert tracker.rates().tolist() == [1.0, 0.0, 0.0]

  tracker.start_session()
  tracker.record(1)

  assert tracker.rates().tolist() == [0.5, 0.5, 0.0]


def test_simulation_result_uses_zero_for_missing_true_theta(item_bank) -> None:
  """A session without a true theta counts as an examinee at 0.

  Fails if examinees drops such a session or holds None for it, so it no longer lines up with
  latest_estimations.
  """
  sessions = [
    CatSessionState(session_id=0, theta_history=[0.2], true_theta=None),
    CatSessionState(session_id=1, theta_history=[-0.4], true_theta=1.0),
  ]
  result = SimulationResult(
    item_bank=item_bank,
    sessions=sessions,
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.1,
  )

  assert np.array_equal(result.examinees, np.array([0.0, 1.0]))


def test_simulation_result_empty_sessions_exposure_and_overlap(item_bank) -> None:
  """A result with no sessions has zero exposure rates and no overlap rate.

  Fails if exposure_rates divides the counts by zero sessions (NaN, with a warning that the test settings turn into
  an error), or if overlap_rate computes a number for a run without tests.
  """
  result = SimulationResult(
    item_bank=item_bank,
    sessions=[],
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.0,
  )

  assert np.array_equal(result.exposure_rates, np.zeros(item_bank.n_items))
  assert result.overlap_rate is None


def test_simulation_result_overlap_rate_for_a_hand_worked_run() -> None:
  """Two 2-item tests, [0, 1] and [0, 2], on a 5-item bank give exposure rates [1, 0.5, 0.5, 0, 0].

  The rates average Q / N, so (N / Q) var(r) + Q / N reduces to sum(r^2) / Q = 1.5 / 2 = 0.75. A wrong test
  length of 1 would give 0.9.
  """
  sessions = [
    finished_session(0, [0, 1]),
    finished_session(1, [0, 2]),
  ]
  result = SimulationResult(
    item_bank=ItemBank.generate_item_bank(5, seed=0),
    sessions=sessions,
    exposure_counts=np.array([2, 1, 1, 0, 0]),
    duration=0.0,
  )

  assert result.exposure_rates.tolist() == [1.0, 0.5, 0.5, 0.0, 0.0]
  assert result.overlap_rate == pytest.approx(0.75)


def test_simulation_result_overlap_requires_uniform_non_zero_test_lengths(item_bank) -> None:
  """Overlap is None for sessions of different lengths or of no items.

  Fails if overlap_rate computes a number from a test length the sessions do not share, or divides
  by a test length of zero.
  """
  heterogeneous_sessions = [
    finished_session(0, [0]),
    finished_session(1, [0, 1]),
  ]
  heterogeneous = SimulationResult(
    item_bank=item_bank,
    sessions=heterogeneous_sessions,
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.0,
  )

  zero_length_sessions = [
    finished_session(0, []),
    finished_session(1, []),
  ]
  zero_length = SimulationResult(
    item_bank=item_bank,
    sessions=zero_length_sessions,
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.0,
  )

  assert heterogeneous.overlap_rate is None
  assert zero_length.overlap_rate is None
