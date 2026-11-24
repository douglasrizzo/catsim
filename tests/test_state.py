"""Tests for the runtime state of sessions and runs."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from catsim import Estimate
from catsim.state import CatSessionState, ExposureTracker, StopReason


@pytest.mark.parametrize(
  ("items", "responses", "thetas", "match"),
  [
    pytest.param([0, 1], [True], [0.0, 0.5], "one response per administered item", id="item-without-response"),
    pytest.param([0], [True], [0.0], "one estimate per response", id="missing-estimate"),
    pytest.param([0], [True], [0.0, 0.5, 0.7], "one estimate per response", id="extra-estimate"),
    pytest.param([], [], [], "the initial estimate", id="no-initial-estimate"),
  ],
)
def test_session_state_rejects_a_history_that_does_not_match_its_responses(
  items: list[int], responses: list[bool], thetas: list[float], match: str
) -> None:
  """A session holds one response per item and one estimate more than responses.

  Fails if such a session is accepted: its current estimate, the last entry of the history, would not
  be the estimate after its last response (or, with an empty history, would not exist).
  """
  with pytest.raises(ValueError, match=match):
    CatSessionState(
      session_id=0,
      estimates=[Estimate(theta, math.inf) for theta in thetas],
      administered_item_ids=items,
      responses=responses,
    )


@pytest.mark.parametrize(
  "estimates",
  [
    pytest.param([0.0], id="bare-theta"),
    pytest.param([{"theta": 0.0, "se": math.inf, "evaluations": 0}], id="mapping"),
    pytest.param([Estimate(0.0, math.inf), 0.5], id="one-bare-theta-among-estimates"),
  ],
)
def test_session_state_rejects_estimates_that_are_not_estimate_objects(estimates: list[object]) -> None:
  """A session restored with bare thetas or mappings in its estimates is rejected when it is built.

  Fails if such a session is accepted: its theta history, its current SE and the engine's next step would fail on it
  later with an AttributeError.
  """
  n_responses = len(estimates) - 1

  with pytest.raises(TypeError, match="needs Estimate entries in estimates"):
    CatSessionState(
      session_id=0,
      estimates=estimates,  # type: ignore[arg-type]
      administered_item_ids=list(range(n_responses)),
      responses=[True] * n_responses,
    )


def test_a_snapshot_keeps_the_estimates_and_both_derive_the_history_from_them() -> None:
  """A snapshot holds the session's estimates, and session and snapshot read the history and current SE from them.

  Fails if the snapshot drops the standard errors or evaluation counts, or if theta_history, current_theta or
  current_se read another field or another entry than the last.
  """
  estimates = [Estimate(0.0, math.inf, 0), Estimate(0.6, 1.1, 14), Estimate(0.4, 0.8, 12)]
  state = CatSessionState(session_id=0, estimates=estimates, administered_item_ids=[2, 5], responses=[True, False])

  snapshot = state.snapshot()

  assert snapshot.estimates == tuple(estimates)
  for view in (state, snapshot):
    assert (view.theta_history, view.current_theta, view.current_se) == ((0.0, 0.6, 0.4), 0.4, 0.8)


def test_session_state_turns_a_persisted_stop_reason_back_into_the_enum() -> None:
  """A stop reason restored from storage as its string is the StopReason member again; an unknown one is rejected.

  Fails if the string is kept, so that ``stop_reason is StopReason.STOPPER`` is False for a restored session, or if
  a reason catsim never sets is accepted.
  """
  persisted = json.loads(json.dumps({"stop_reason": StopReason.STOPPER, "unknown": "paused"}))

  restored = CatSessionState(session_id=0, estimates=[Estimate(0.0, math.inf)], stop_reason=persisted["stop_reason"])

  assert restored.stop_reason is StopReason.STOPPER
  with pytest.raises(ValueError, match="'paused' is not a valid StopReason"):
    CatSessionState(session_id=0, estimates=[Estimate(0.0, math.inf)], stop_reason=persisted["unknown"])


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
