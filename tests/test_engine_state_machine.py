"""Hypothesis stateful tests for :class:`catsim.engine.CatEngine` orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
  RuleBasedStateMachine,
  initialize,
  invariant,
  rule,
  run_state_machine_as_test,
)
from typing_extensions import override

from catsim import Estimate, StepContext
from catsim.engine import CatEngine, RunContext
from catsim.estimation import BaseEstimator, NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.exposure import BaseExposureControl
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import BaseSelector, MaxInfoSelector
from catsim.state import CatSessionState, ExposureTracker, StopReason
from catsim.stopping import MaxLengthStopper


@dataclass
class FixedResponseProvider:
  """Deterministic response provider for state machine runs."""

  response: bool

  def answer(self, item_bank, item_id, session, rng) -> bool:  # noqa: ARG002
    """Return the fixed response."""
    return self.response


@dataclass
class RecordingEstimator(BaseEstimator):
  """Estimator stub returning a fixed theta."""

  theta: float

  @override
  def estimate(self, context: StepContext) -> Estimate:
    """Return the fixed theta with a standard error of 1."""
    return Estimate(theta=float(self.theta), se=1.0)


class QueueOrderSelector(BaseSelector):
  """Select the next item by scanning a fixed permutation, skipping administered ids."""

  def __init__(self, order: tuple[int, ...] | list[int]) -> None:
    self._order = tuple(order)

  @override
  def select(self, context: StepContext) -> int:
    """Return the first item of the permutation not yet administered, or else the lowest-index one."""
    administered_set = set(context.administered_items)
    n = context.item_bank.n_items
    for idx in self._order:
      if 0 <= idx < n and idx not in administered_set:
        return int(idx)
    for i in range(n):
      if i not in administered_set:
        return int(i)
    msg = "There are no more items to apply."
    raise NoItemsAvailableError(msg)


class AlternatingResponseProvider:
  """Responses alternate False/True across administrations."""

  def __init__(self) -> None:
    self._count = 0

  def answer(self, item_bank, item_id, session, rng) -> bool:  # noqa: ARG002
    """Answer incorrectly, then correctly, in turn."""
    self._count += 1
    return self._count % 2 == 0


class RandomlyDecliningControl(BaseExposureControl):
  """Exposure control that declines each proposal with a fixed probability, drawn from the run's generator."""

  def __init__(self, probability: float) -> None:
    self._probability = probability

  @override
  def accept(self, context, item_id) -> bool:
    return bool(context.rng.random() >= self._probability)


_ORCHESTRATION_SETTINGS = settings(
  max_examples=25,
  stateful_step_count=60,
)

_INTEGRATION_SETTINGS = settings(
  max_examples=10,
  stateful_step_count=40,
)


class CatEngineOrchestrationMachine(RuleBasedStateMachine):
  """Layer A: deterministic stubs, random legal step counts and selection permutations."""

  @initialize(
    selection_order=st.permutations((0, 1, 2, 3, 4, 5, 6, 7)),
    max_items=st.integers(min_value=4, max_value=7),
  )
  def init_engine(self, selection_order: tuple[int, ...], max_items: int) -> None:
    """Start a session with stub strategies, which begins unstopped at the initializer's estimate."""
    self.item_bank = ItemBank.generate(8, rng=0)
    # max_items stays below the bank size, so only the stopper can end the session
    self.max_items = max_items
    self.selector = QueueOrderSelector(selection_order)
    self.engine = CatEngine(
      self.item_bank,
      FixedPointInitializer(0.0),
      self.selector,
      RecordingEstimator(theta=0.5),
      MaxLengthStopper(max_items),
    )
    self.context = RunContext(rng=np.random.default_rng(12345))
    self.response_provider = FixedResponseProvider(True)
    # Start the session during initialization so every rule is valid at every step; rules
    # guarded by mutually exclusive conditions made Hypothesis discard most rule draws.
    self.session: CatSessionState = self.engine.start_session(
      session_id=0,
      context=self.context,
      true_theta=0.0,
    )
    assert self.session.stop_reason is None
    assert self.session.current_theta == 0.0
    assert len(self.session.theta_history) == 1

  @rule()
  def step(self) -> None:
    """Stepping a stopped session changes nothing; a live step records one item with its response and estimate.

    Fails if a stopped session administers another item or changes its stop reason, if a step records an item
    without its estimate or response, or if the session stops before max_items items or for a reason other than the
    stopper.
    """
    prev_hist = len(self.session.theta_history)
    prev_admin = len(self.session.administered_items)
    was_stopped = self.session.is_stopped
    prev_reason = self.session.stop_reason
    result = self.engine.step(self.session, self.response_provider, self.context)
    if was_stopped:
      # Stepping a terminal session must be a no-op.
      assert result.stopped is True
      assert result.selected_item_id is None
      assert len(self.session.administered_items) == prev_admin
      assert self.session.stop_reason == prev_reason
    elif not self.session.is_stopped:
      if result.selected_item_id is not None:
        assert len(self.session.theta_history) == prev_hist + 1
        assert len(self.session.administered_items) == prev_admin + 1
        assert result.response is True
    else:
      assert result.stopped is True
      assert self.session.stop_reason is StopReason.STOPPER
      assert self.session.administered_count == self.max_items

  @invariant()
  def session_lists_consistent(self) -> None:
    """The administered items are distinct items of the bank, each with one response and one estimate after the first.

    Fails if a step administers an item twice or from outside the bank, or records an item without its response or
    estimate, or if the session stops for a reason other than the stopper.
    """
    s = self.session
    n = self.item_bank.n_items
    for i in s.administered_items:
      assert 0 <= int(i) < n
    assert len(s.responses) == len(s.administered_items)
    assert len(set(s.administered_items)) == len(s.administered_items)
    assert len(s.theta_history) == 1 + len(s.administered_items)
    assert s.stop_reason in {None, StopReason.STOPPER}


class CatEngineIntegrationMachine(RuleBasedStateMachine):
  """Layer B: real MaxInfoSelector, bounded numerical estimator, alternating responses."""

  @initialize(
    bank_seed=st.integers(min_value=0, max_value=500),
    max_items=st.integers(min_value=3, max_value=5),
  )
  def init_engine(self, bank_seed: int, max_items: int) -> None:
    """Start a session with the maximum-information selector and the numerical estimator."""
    self.item_bank = ItemBank.generate(12, rng=bank_seed)
    # max_items stays below the bank size, so only the stopper can end the session
    self.max_items = max_items
    self.engine = CatEngine(
      self.item_bank,
      FixedPointInitializer(0.0),
      MaxInfoSelector(),
      NumericalSearchEstimator(),
      MaxLengthStopper(max_items),
    )
    self.context = RunContext(rng=np.random.default_rng(999))
    self.response_provider = AlternatingResponseProvider()
    self.session: CatSessionState = self.engine.start_session(
      session_id=0,
      context=self.context,
      true_theta=0.25,
    )
    assert self.session.stop_reason is None

  @rule()
  def step(self) -> None:
    """With the real strategies, a step behaves as with the stubs and leaves a finite estimate.

    Fails if a stopped session administers another item or changes its stop reason, if a step records an item
    without its estimate, if the estimator returns a non-finite estimate, or if the session stops before max_items
    items or for a reason other than the stopper.
    """
    prev_hist = len(self.session.theta_history)
    prev_admin = len(self.session.administered_items)
    was_stopped = self.session.is_stopped
    prev_reason = self.session.stop_reason
    result = self.engine.step(self.session, self.response_provider, self.context)
    if was_stopped:
      assert result.stopped is True
      assert result.selected_item_id is None
      assert len(self.session.administered_items) == prev_admin
      assert self.session.stop_reason == prev_reason
    elif not self.session.is_stopped:
      if result.selected_item_id is not None:
        assert len(self.session.theta_history) == prev_hist + 1
        assert len(self.session.administered_items) == prev_admin + 1
        assert np.isfinite(self.session.current_theta)
    else:
      assert result.stopped is True
      assert self.session.stop_reason is StopReason.STOPPER
      assert self.session.administered_count == self.max_items

  @invariant()
  def session_lists_consistent(self) -> None:
    """The administered items are distinct items of the bank, each with one response and one estimate after the first.

    Fails if a step administers an item twice or from outside the bank, or records an item without its response or
    estimate, or if the session stops for a reason other than the stopper.
    """
    s = self.session
    n = self.item_bank.n_items
    for i in s.administered_items:
      assert 0 <= int(i) < n
    assert len(s.responses) == len(s.administered_items)
    assert len(set(s.administered_items)) == len(s.administered_items)
    assert len(s.theta_history) == 1 + len(s.administered_items)
    assert s.stop_reason in {None, StopReason.STOPPER}


class ExposureControlMachine(RuleBasedStateMachine):
  """Sessions that share one tracker under a control that declines at random, stepped and reselected in any order."""

  @initialize(
    probability=st.floats(min_value=0.0, max_value=0.9),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    max_items=st.integers(min_value=2, max_value=5),
  )
  def init_engine(self, probability: float, seed: int, max_items: int) -> None:
    """Start one session of an engine whose exposure control declines each proposal with the drawn probability."""
    self.item_bank = ItemBank.generate(8, rng=0)
    self.engine = CatEngine(
      self.item_bank,
      FixedPointInitializer(0.0),
      MaxInfoSelector(),
      RecordingEstimator(theta=0.5),
      MaxLengthStopper(max_items),
      exposure_control=RandomlyDecliningControl(probability),
    )
    self.tracker = ExposureTracker(self.item_bank.n_items)
    self.context = RunContext(rng=np.random.default_rng(seed), exposure_tracker=self.tracker)
    self.sessions = [self.engine.start_session(session_id=0, context=self.context)]

  @rule()
  def step(self) -> None:
    """Step the newest session, whose control may decline proposals first."""
    self.engine.step(self.sessions[-1], FixedResponseProvider(True), self.context)

  @rule()
  def select_without_answering(self) -> None:
    """Select an item for the newest session and never answer it, as on a page reload."""
    self.engine.select_next(self.sessions[-1], self.context)

  @rule()
  def start_another_session(self) -> None:
    """Start another session on the shared exposure tracker."""
    self.sessions.append(self.engine.start_session(session_id=len(self.sessions), context=self.context))

  @invariant()
  def proposals_are_administrations_plus_declines(self) -> None:
    """Each item's proposals are its administrations plus its declines, and no more than the sessions started.

    Fails if a decline is not recorded as an exclusion of the session, or if a selection left unanswered is counted,
    so that its reselection counts the item twice.
    """
    declined = [item_id for session in self.sessions for item_id in session.excluded_items]
    declines = np.bincount(declined, minlength=self.item_bank.n_items)
    assert np.array_equal(self.tracker.proposals, self.tracker.counts + declines)
    assert int(np.max(self.tracker.proposals)) <= self.tracker.n_sessions

  @invariant()
  def no_item_is_both_administered_and_excluded(self) -> None:
    """A session never administers an item its control declined, and records each decline once.

    Fails if the engine hands a declined item back to the session, or excludes an item twice.
    """
    for session in self.sessions:
      assert not set(session.administered_items) & set(session.excluded_items)
      assert len(set(session.excluded_items)) == len(session.excluded_items)


def test_cat_engine_orchestration_state_machine() -> None:
  """Stepping a session with stub strategies keeps its items, responses and estimates in line, in any step order.

  Fails if a step records an item without its response or estimate, administers an item twice, or changes a session
  that has already stopped.
  """
  run_state_machine_as_test(CatEngineOrchestrationMachine, settings=_ORCHESTRATION_SETTINGS)


def test_cat_engine_integration_state_machine() -> None:
  """The same invariants hold with the real maximum-information selector and numerical estimator.

  Fails if a real strategy makes a step record an item without its response or estimate, give an item twice or a
  non-finite estimate, or change a session that has already stopped.
  """
  run_state_machine_as_test(CatEngineIntegrationMachine, settings=_INTEGRATION_SETTINGS)


def test_exposure_control_state_machine() -> None:
  """Under a control that declines at random, the tracker's proposals are the administrations plus the declines.

  Sessions are stepped, reselected without an answer and started in any order. Fails if a decline reaches the
  tracker but not the session (or the session but not the tracker), if a second select_next before the answer
  counts a proposal again, or if a session administers an item it excluded.
  """
  run_state_machine_as_test(ExposureControlMachine, settings=_ORCHESTRATION_SETTINGS)
