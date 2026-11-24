"""Tests for the runtime state of sessions and runs."""

from __future__ import annotations

import json
import math

import numpy as np
import numpy.typing as npt
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
      administered_items=items,
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
      estimates=estimates,  # ty: ignore[invalid-argument-type]
      administered_items=list(range(n_responses)),
      responses=[True] * n_responses,
    )


def test_a_snapshot_keeps_the_estimates_and_both_derive_the_history_from_them() -> None:
  """A snapshot holds the session's estimates, and session and snapshot read the history and current SE from them.

  Fails if the snapshot drops the standard errors or evaluation counts, or if theta_history, current_theta or
  current_se read another field or another entry than the last.
  """
  estimates = [Estimate(0.0, math.inf, 0), Estimate(0.6, 1.1, 14), Estimate(0.4, 0.8, 12)]
  state = CatSessionState(session_id=0, estimates=estimates, administered_items=[2, 5], responses=[True, False])

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


@pytest.mark.parametrize(
  ("excluded", "match"),
  [
    pytest.param([4, 4], r"repeats an item in excluded_items, got \[4, 4\]", id="repeated"),
    pytest.param([1, 4], r"has administered items \[1\], which excluded_items also holds", id="administered"),
  ],
)
def test_session_state_rejects_excluded_items_it_cannot_hold(excluded: list[int], match: str) -> None:
  """A session restored with an excluded item twice, or with an item both administered and excluded, is rejected.

  Fails if such a session is accepted: an item is given or declined once per session, so a save holding either is
  corrupt, and its proposal counts (administrations plus declines) would count an item twice for one examinee.
  """
  with pytest.raises(ValueError, match=match):
    CatSessionState(
      session_id=0,
      estimates=[Estimate(0.0, math.inf), Estimate(0.5, 1.0)],
      administered_items=[1],
      responses=[True],
      excluded_items=excluded,
    )


@pytest.mark.parametrize("field_name", ["administered_items", "excluded_items"])
@pytest.mark.parametrize("entry", [pytest.param(2.5, id="float"), pytest.param("3", id="str")])
def test_session_state_rejects_item_indices_that_are_not_integers(field_name: str, entry: object) -> None:
  """A restored session whose item lists hold a float or a string fails when it is built, naming the field.

  Fails if 2.5 is accepted: the engine's membership checks never match it while ``remaining`` truncates it to item 2,
  so an excluded item could be given or an administered one given again. Fails too if "3" is accepted and the session
  breaks later inside a strategy with a TypeError that says nothing about the saved data.
  """
  administered, excluded = ([entry], []) if field_name == "administered_items" else ([], [entry])
  with pytest.raises(TypeError, match=rf"needs integer item indices in {field_name}, got {entry!r}"):
    CatSessionState(
      session_id=0,
      estimates=[Estimate(0.0, math.inf)] * (len(administered) + 1),
      administered_items=administered,  # ty: ignore[invalid-argument-type]
      responses=[True] * len(administered),
      excluded_items=excluded,  # ty: ignore[invalid-argument-type]
    )


def test_session_state_stores_numpy_integer_indices_as_python_ints() -> None:
  """Item indices restored as NumPy integers, such as from a saved array, are stored as builtin ints.

  Fails if the session keeps the numpy.int64 entries: saving it again with json.dumps would raise TypeError.
  """
  state = CatSessionState(
    session_id=0,
    estimates=[Estimate(0.0, math.inf), Estimate(0.5, 1.0)],
    administered_items=[np.int64(4)],  # ty: ignore[invalid-argument-type]
    responses=[True],
    excluded_items=[np.int64(7)],  # ty: ignore[invalid-argument-type]
  )

  assert json.dumps([state.administered_items, state.excluded_items]) == "[[4], [7]]"
  assert [type(item) for item in state.administered_items + state.excluded_items] == [int, int]


def test_a_session_and_its_snapshot_keep_their_own_excluded_items() -> None:
  """The session copies the excluded items it is given, and a snapshot keeps them as they were.

  Fails if the session keeps the caller's list (the engine's declines would change the application's saved copy) or
  if the snapshot shares the session's list (a later decline would show up in an earlier step result).
  """
  saved = [6, 2]
  state = CatSessionState(session_id=0, estimates=[Estimate(0.0, math.inf)], excluded_items=saved)
  snapshot = state.snapshot()

  state.excluded_items.append(9)

  assert saved == [6, 2]
  assert state.excluded_items == [6, 2, 9]
  assert snapshot.excluded_items == (6, 2)


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


def test_a_tracker_rebuilt_from_saved_counts_continues_where_the_saved_one_was() -> None:
  """A tracker rebuilt from a saved one's counts and n_sessions gives the rates the saved one would have given.

  Four sessions gave items [0, 1], [0, 2], [0, 3] and [0] on a 4-item bank, so the counts are [4, 1, 1, 1] over 4
  sessions. The next session is selected against them over the 4 others: [1, 0.25, 0.25, 0.25]. Fails if the
  rebuilt tracker ignores the saved counts (all zero) or the saved sessions (the rates over no other session).
  """
  original = ExposureTracker(4)
  for items in ([0, 1], [0, 2], [0, 3], [0]):
    original.start_session()
    for item_id in items:
      original.record(item_id)
  saved = json.loads(json.dumps({"counts": original.counts.tolist(), "n_sessions": original.n_sessions}))
  rebuilt = ExposureTracker(4, counts=saved["counts"], n_sessions=saved["n_sessions"])

  original.start_session()
  rebuilt.start_session()

  assert rebuilt.rates().tolist() == original.rates().tolist() == [1.0, 0.25, 0.25, 0.25]


def test_a_rebuilt_tracker_owns_its_counts() -> None:
  """Recording on a rebuilt tracker leaves the caller's array alone.

  Fails if the tracker keeps the array it was given, so the application's saved copy would change with every
  administration and could be saved twice.
  """
  saved = np.array([2, 0, 1])
  tracker = ExposureTracker(3, counts=saved, n_sessions=2)

  tracker.record(1)

  assert saved.tolist() == [2, 0, 1]
  assert tracker.counts.tolist() == [2, 1, 1]


@pytest.mark.parametrize(
  ("counts", "n_sessions", "error", "match"),
  [
    pytest.param([1, 0, 2], 3, ValueError, r"one count per item \(4\), got shape \(3,\)", id="too-few-counts"),
    pytest.param(
      [[1, 0], [2, 0]], 3, ValueError, r"one count per item \(4\), got shape \(2, 2\)", id="two-dimensional"
    ),
    pytest.param([1, -1, 0, 0], 3, ValueError, "counts cannot be negative, got -1", id="negative-count"),
    pytest.param([1.0, 0.0, 2.0, 0.0], 3, TypeError, "counts must be integers, got dtype float64", id="float-counts"),
    pytest.param([1, 0, 2, 0], -1, ValueError, "n_sessions cannot be negative, got -1", id="negative-sessions"),
    pytest.param([1, 0, 2, 0], 3.0, TypeError, "n_sessions must be an integer, got float 3.0", id="float-sessions"),
    pytest.param([1, 0, 2, 0], True, TypeError, "n_sessions must be an integer, got bool True", id="bool-sessions"),
    pytest.param(
      [5, 0, 2, 0], 3, ValueError, "an item has 5 administrations but only 3 sessions", id="counts-above-sessions"
    ),
  ],
)
def test_a_tracker_rejects_saved_counts_it_cannot_use(
  counts: npt.ArrayLike, n_sessions: int, error: type[Exception], match: str
) -> None:
  """Counts that are not one non-negative integer per item, or a session count that is not a non-negative integer, fail.

  Fails if the tracker accepts counts saved for another bank (rates() would then broadcast or fail mid-session), a
  negative count or session count (negative rates, which no r_max cap would hold back), fractional values, a bool
  for the session count, or counts from another save than the session count (a rate above 1 for every session).
  """
  with pytest.raises(error, match=match):
    ExposureTracker(4, counts=counts, n_sessions=n_sessions)


def test_a_decline_counts_as_a_proposal_but_not_as_an_administration() -> None:
  """record_decline adds to an item's proposals only, so declines never raise its exposure rate.

  Three sessions started; item 0 was administered once and declined once, item 1 declined twice. Fails if a decline
  is counted as an administration (item 1 would then be held back by an r_max cap it never reached) or not counted
  as a proposal (Sympson-Hetter's calibration would see too few proposals).
  """
  tracker = ExposureTracker(3)
  for _ in range(3):
    tracker.start_session()
  tracker.record(0)
  for item_id in (0, 1, 1):
    tracker.record_decline(item_id)

  assert tracker.counts.tolist() == [1, 0, 0]
  assert tracker.proposals.tolist() == [2, 2, 0]
  assert tracker.rates().tolist() == [0.5, 0.0, 0.0]


def test_a_tracker_rebuilt_without_proposals_takes_its_counts_as_its_own_copy() -> None:
  """A tracker rebuilt from counts alone has one proposal per administration, in an array of its own.

  Fails if the rebuilt tracker starts its proposals at zero (every item would have more administrations than
  proposals) or shares the counts array (each administration would then add two to both).
  """
  tracker = ExposureTracker(3, counts=[2, 0, 1], n_sessions=2)

  tracker.record(1)

  assert tracker.counts.tolist() == [2, 1, 1]
  assert tracker.proposals.tolist() == [2, 1, 1]


@pytest.mark.parametrize(
  ("proposals", "error", "match"),
  [
    pytest.param(
      [2, 1, 1], ValueError, r"proposals must hold one count per item \(4\), got shape \(3,\)", id="too-few"
    ),
    pytest.param([2, -1, 1, 0], ValueError, "proposals cannot be negative, got -1", id="negative"),
    pytest.param([2.0, 1.0, 1.0, 0.0], TypeError, "proposals must be integers, got dtype float64", id="float"),
    pytest.param([2, 0, 1, 0], ValueError, "item 1 has 1 administrations but only 0 proposals", id="below-counts"),
    pytest.param([2, 1, 4, 0], ValueError, "an item has 4 proposals but only 3 sessions", id="above-sessions"),
  ],
)
def test_a_tracker_rejects_saved_proposals_it_cannot_use(
  proposals: npt.ArrayLike, error: type[Exception], match: str
) -> None:
  """Proposals that are not one non-negative integer per item, or that do not fit the counts and sessions, fail.

  The saved counts are [2, 1, 1, 0] over 3 sessions. Fails if the tracker accepts proposals saved for another bank,
  negative or fractional ones, fewer proposals than administrations (every administration is a proposal) or more
  proposals than sessions (a session proposes an item at most once, so the proposal rate would exceed 1).
  """
  with pytest.raises(error, match=match):
    ExposureTracker(4, counts=[2, 1, 1, 0], n_sessions=3, proposals=proposals)


def test_a_tracker_rebuilt_from_compact_counts_counts_past_their_type() -> None:
  """Counts saved as uint8 kept that type, so the 256th administration wrapped the count to 0 and lifted the cap."""
  tracker = ExposureTracker(2, counts=np.array([255, 0], dtype=np.uint8), n_sessions=300)

  tracker.record(0)

  assert tracker.counts.tolist() == [256, 0]
