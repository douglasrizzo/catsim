"""Hypothesis stateful tests for :class:`catsim.CatEngine`: sessions stepped through select and answer in any order."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, rule, run_state_machine_as_test

from catsim import CatEngine, Estimate, ItemBank, Session, StepContext, StopReason
from catsim.compat import override
from catsim.estimation import BaseEstimator, NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.exposure import BaseExposureControl
from catsim.initialization import FixedPointInitializer
from catsim.selection import BaseSelector, MaxInfoSelector
from catsim.stopping import MaxLengthStopper


@dataclass
class FixedEstimator(BaseEstimator):
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


class RandomlyDecliningControl(BaseExposureControl):
  """Exposure control that declines each proposal with a fixed probability, drawn from the call's generator."""

  def __init__(self, probability: float) -> None:
    self._probability = probability

  @override
  def accept(self, context: StepContext, item: int) -> bool:
    return bool(context.rng.random() >= self._probability)


_ORCHESTRATION_SETTINGS = settings(max_examples=25, stateful_step_count=60)
_INTEGRATION_SETTINGS = settings(max_examples=10, stateful_step_count=40)


class SteppedSessionMachine(RuleBasedStateMachine):
  """One session stepped through select and answer, by an engine whose stopper alone can end it."""

  engine: CatEngine
  max_items: int

  def __init__(self) -> None:
    super().__init__()
    self.rng = np.random.default_rng(12345)
    self.responses_given = 0

  def respond(self) -> bool:
    """Answer incorrectly, then correctly, in turn."""
    self.responses_given += 1
    return self.responses_given % 2 == 0

  def begin(self) -> None:
    """Start the session, which begins running at the initializer's estimate."""
    # Starting it in the initializer makes every rule valid at every step; rules guarded by mutually exclusive
    # conditions made Hypothesis discard most rule draws.
    self.session: Session = self.engine.start(rng=self.rng)
    assert self.session.stop_reason is None
    assert self.session.theta_history == (0.0,)

  @rule()
  def step(self) -> None:
    """A step on a stopped session changes nothing; a running one records one item with its response and estimate.

    Fails if a stopped session is given another item or changes its stop reason, if a step records an item without its
    estimate or response, if the new session is not a session after the old one, or if the session stops before
    max_items items or for a reason other than the stopper.
    """
    before = self.session
    selection = self.engine.select(before, rng=self.rng)
    if before.stop_reason is not None:
      assert selection.item is None
      assert selection.session is before
      return
    assert selection.item is not None
    self.session = self.engine.answer(selection.session, selection.item, self.respond(), rng=self.rng)
    assert self.session.administered_items == (*before.administered_items, selection.item)
    assert self.session.estimates[:-1] == before.estimates
    assert np.isfinite(self.session.current_theta)
    if self.session.stop_reason is not None:
      assert self.session.stop_reason is StopReason.STOPPER
      assert len(self.session.administered_items) == self.max_items

  @invariant()
  def session_lists_consistent(self) -> None:
    """The administered items are distinct items of the bank, each with one response and one estimate after the first.

    Fails if a step administers an item twice or from outside the bank, records an item without its response or
    estimate, or ends the session for a reason other than the stopper.
    """
    s = self.session
    assert all(0 <= item < self.engine.item_bank.n_items for item in s.administered_items)
    assert len(s.responses) == len(s.administered_items)
    assert len(set(s.administered_items)) == len(s.administered_items)
    assert len(s.theta_history) == 1 + len(s.administered_items)
    assert s.stop_reason in {None, StopReason.STOPPER}


class StubbedEngineMachine(SteppedSessionMachine):
  """Deterministic stubs, random test lengths and selection permutations."""

  @initialize(
    selection_order=st.permutations((0, 1, 2, 3, 4, 5, 6, 7)),
    max_items=st.integers(min_value=4, max_value=7),
  )
  def init_engine(self, selection_order: tuple[int, ...], max_items: int) -> None:
    """Start a session with stub strategies, on an 8-item bank that outlasts the test."""
    self.max_items = max_items
    self.engine = CatEngine(
      ItemBank.generate(8, rng=0),
      FixedPointInitializer(0.0),
      QueueOrderSelector(selection_order),
      FixedEstimator(theta=0.5),
      MaxLengthStopper(max_items),
    )
    self.begin()


class RealStrategiesMachine(SteppedSessionMachine):
  """The maximum-information selector and the numerical estimator on random banks."""

  @initialize(bank_seed=st.integers(min_value=0, max_value=500), max_items=st.integers(min_value=3, max_value=5))
  def init_engine(self, bank_seed: int, max_items: int) -> None:
    """Start a session with the real strategies, on a 12-item bank that outlasts the test."""
    self.max_items = max_items
    self.engine = CatEngine(
      ItemBank.generate(12, rng=bank_seed),
      FixedPointInitializer(0.0),
      MaxInfoSelector(),
      NumericalSearchEstimator(),
      MaxLengthStopper(max_items),
    )
    self.begin()


class ExposureControlMachine(RuleBasedStateMachine):
  """Sessions under a control that declines at random, stepped, reselected and started in any order.

  The machine plays an application that counts each session's proposals as ADR-0011 tells it to: the items a
  selection declined, and the item it administers.
  """

  @initialize(
    probability=st.floats(min_value=0.0, max_value=0.9),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    max_items=st.integers(min_value=2, max_value=5),
  )
  def init_engine(self, probability: float, seed: int, max_items: int) -> None:
    """Start one session of an engine whose exposure control declines each proposal with the drawn probability."""
    self.engine = CatEngine(
      ItemBank.generate(8, rng=0),
      FixedPointInitializer(0.0),
      MaxInfoSelector(),
      FixedEstimator(theta=0.5),
      MaxLengthStopper(max_items),
      exposure_control=RandomlyDecliningControl(probability),
    )
    self.rng = np.random.default_rng(seed)
    self.sessions: list[Session] = []
    self.proposals: list[np.ndarray] = []
    self.start_another_session()

  def _select(self) -> tuple[int | None, Session]:
    """Select for the newest session, count the declines and keep the session the selection returned."""
    selection = self.engine.select(self.sessions[-1], rng=self.rng)
    self.proposals[-1][list(selection.declined)] += 1
    self.sessions[-1] = selection.session
    return selection.item, selection.session

  @rule()
  def step(self) -> None:
    """Select for the newest session and answer the item, counting it as a proposal."""
    item, session = self._select()
    if item is not None:
      self.proposals[-1][item] += 1
      self.sessions[-1] = self.engine.answer(session, item, True, rng=self.rng)

  @rule()
  def select_without_answering(self) -> None:
    """Select and keep the session, but never answer the item shown, as on a page reload."""
    self._select()

  @rule()
  def select_and_discard(self) -> None:
    """Select and keep nothing, as an application that loses the selection before storing it."""
    self.engine.select(self.sessions[-1], rng=self.rng)

  @rule()
  def start_another_session(self) -> None:
    """Start another session."""
    self.sessions.append(self.engine.start(rng=self.rng))
    self.proposals.append(np.zeros(self.engine.item_bank.n_items, dtype=int))

  @invariant()
  def proposals_are_administrations_plus_declines(self) -> None:
    """Each session's counted proposals are its administered items plus its excluded ones, each at most once.

    Fails if a selection's declines are not its session's new exclusions (or the reverse), or if an item is proposed
    twice for one examinee, as a selector reproposing a declined item, or a reselection counting a decline again,
    would make it; Sympson-Hetter's calibration needs each item proposed at most once per examinee.
    """
    for session, proposals in zip(self.sessions, self.proposals, strict=True):
      expected = np.bincount(
        [*session.administered_items, *session.excluded_items], minlength=self.engine.item_bank.n_items
      )
      assert proposals.tolist() == expected.tolist()
      assert int(proposals.max()) <= 1

  @invariant()
  def no_item_is_both_administered_and_excluded(self) -> None:
    """A session never administers an item its control declined, and records each decline once.

    Fails if the engine hands a declined item back to the session, or excludes an item twice.
    """
    for session in self.sessions:
      assert not set(session.administered_items) & set(session.excluded_items)
      assert len(set(session.excluded_items)) == len(session.excluded_items)


def test_stepping_a_session_with_stub_strategies_keeps_its_history_in_line() -> None:
  """Stepping a session with stub strategies keeps its items, responses and estimates in line, in any step order.

  Fails if a step records an item without its response or estimate, administers an item twice, or changes a session
  that has already stopped.
  """
  run_state_machine_as_test(StubbedEngineMachine, settings=_ORCHESTRATION_SETTINGS)


def test_stepping_a_session_with_real_strategies_keeps_its_history_in_line() -> None:
  """The same invariants hold with the real maximum-information selector and numerical estimator.

  Fails if a real strategy makes a step record an item without its response or estimate, give an item twice or a
  non-finite estimate, or change a session that has already stopped.
  """
  run_state_machine_as_test(RealStrategiesMachine, settings=_INTEGRATION_SETTINGS)


def test_exposure_control_state_machine() -> None:
  """Under a control that declines at random, the declines a selection returns are its session's new exclusions.

  Sessions are stepped, reselected with and without keeping the selection, and started in any order. Fails if a
  decline reaches the session but not the selection (or the reverse), if an item is proposed twice for one session,
  or if a session administers an item it excluded.
  """
  run_state_machine_as_test(ExposureControlMachine, settings=_ORCHESTRATION_SETTINGS)
