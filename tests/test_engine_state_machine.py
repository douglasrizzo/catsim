"""Hypothesis stateful tests for :class:`catsim.engine.CatEngine` orchestration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.stateful import (
  RuleBasedStateMachine,
  initialize,
  invariant,
  precondition,
  rule,
  run_state_machine_as_test,
)

from catsim.engine import CatEngine, RunContext
from catsim.estimation import NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector
from catsim.state import CatSessionState, SessionStatus
from catsim.stopping import TestLengthStopper as LengthStopper


@dataclass
class FixedResponseProvider:
  """Deterministic response provider for state machine runs."""

  response: bool

  def answer(self, state, item_bank, item_id, context) -> bool:  # noqa: ARG002
    return self.response


@dataclass
class RecordingEstimator:
  """Estimator stub returning a fixed theta."""

  theta: float

  def estimate(self, *, item_bank, administered_items, response_vector, est_theta) -> float:  # noqa: ARG002
    return float(self.theta)


class QueueOrderSelector:
  """Select the next item by scanning a fixed permutation, skipping administered ids."""

  def __init__(self, order: tuple[int, ...] | list[int]) -> None:
    self._order = tuple(order)

  def select(self, item_bank, administered_items, est_theta, rng=None, exposure_rates=None):  # noqa: ARG002
    administered_set = set(administered_items)
    n = item_bank.n_items
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

  def answer(self, state, item_bank, item_id, context) -> bool:  # noqa: ARG002
    self._count += 1
    return self._count % 2 == 0


_ORCHESTRATION_SETTINGS = settings(
  max_examples=25,
  stateful_step_count=60,
  deadline=None,
)

_INTEGRATION_SETTINGS = settings(
  max_examples=10,
  stateful_step_count=40,
  deadline=None,
)


class CatEngineOrchestrationMachine(RuleBasedStateMachine):
  """Layer A: deterministic stubs, random legal step counts and selection permutations."""

  @initialize(
    selection_order=st.permutations((0, 1, 2, 3, 4, 5, 6, 7)),
    max_items=st.integers(min_value=4, max_value=7),
  )
  def init_engine(self, selection_order: tuple[int, ...], max_items: int) -> None:
    self.item_bank = ItemBank.generate_item_bank(8, seed=0)
    self.selector = QueueOrderSelector(selection_order)
    self.engine = CatEngine(
      FixedPointInitializer(0.0),
      self.selector,
      RecordingEstimator(theta=0.5),
      LengthStopper(max_items=max_items),
    )
    self.context = RunContext(rng=np.random.default_rng(12345))
    self.response_provider = FixedResponseProvider(True)
    self.session: CatSessionState | None = None

  @precondition(lambda self: self.session is None)
  @rule()
  def start_session(self) -> None:
    self.session = self.engine.start_session(
      session_id=0,
      item_bank=self.item_bank,
      context=self.context,
      true_theta=0.0,
    )
    assert self.session.status == SessionStatus.IN_PROGRESS
    assert self.session.current_theta == 0.0
    assert len(self.session.theta_history) == 1

  @precondition(lambda self: self.session is not None and self.session.status == SessionStatus.IN_PROGRESS)
  @rule()
  def step(self) -> None:
    assert self.session is not None
    prev_hist = len(self.session.theta_history)
    prev_admin = len(self.session.administered_item_ids)
    result = self.engine.step(self.session, self.item_bank, self.response_provider, self.context)
    if self.session.status == SessionStatus.IN_PROGRESS:
      if result.selected_item_id is not None:
        assert len(self.session.theta_history) == prev_hist + 1
        assert len(self.session.administered_item_ids) == prev_admin + 1
        assert result.response is True
    else:
      assert result.stopped is True
      assert self.session.stop_reason in {"stop_rule", "item_bank_exhausted", "selector_stopped"}

  @precondition(lambda self: self.session is not None and self.session.status == SessionStatus.STOPPED)
  @rule()
  def noop_after_stop(self) -> None:
    """Allow Hypothesis to continue stepping once the session is terminal."""

  @invariant()
  def session_lists_consistent(self) -> None:
    if self.session is None:
      return
    s = self.session
    n = self.item_bank.n_items
    for i in s.administered_item_ids:
      assert 0 <= int(i) < n
    assert len(s.responses) == len(s.administered_item_ids)
    assert len(set(s.administered_item_ids)) == len(s.administered_item_ids)
    assert len(s.theta_history) == 1 + len(s.administered_item_ids)
    if s.status == SessionStatus.STOPPED:
      assert s.stop_reason in {"stop_rule", "item_bank_exhausted", "selector_stopped"}


class CatEngineIntegrationMachine(RuleBasedStateMachine):
  """Layer B: real MaxInfoSelector, bounded numerical estimator, alternating responses."""

  @initialize(
    bank_seed=st.integers(min_value=0, max_value=500),
    max_items=st.integers(min_value=3, max_value=5),
  )
  def init_engine(self, bank_seed: int, max_items: int) -> None:
    self.item_bank = ItemBank.generate_item_bank(12, seed=bank_seed)
    self.engine = CatEngine(
      FixedPointInitializer(0.0),
      MaxInfoSelector(),
      NumericalSearchEstimator(method="bounded", dodd=True),
      LengthStopper(max_items=max_items),
    )
    self.context = RunContext(rng=np.random.default_rng(999))
    self.response_provider = AlternatingResponseProvider()
    self.session: CatSessionState | None = None

  @precondition(lambda self: self.session is None)
  @rule()
  def start_session(self) -> None:
    self.session = self.engine.start_session(
      session_id=0,
      item_bank=self.item_bank,
      context=self.context,
      true_theta=0.25,
    )
    assert self.session.status == SessionStatus.IN_PROGRESS

  @precondition(lambda self: self.session is not None and self.session.status == SessionStatus.IN_PROGRESS)
  @rule()
  def step(self) -> None:
    assert self.session is not None
    prev_hist = len(self.session.theta_history)
    prev_admin = len(self.session.administered_item_ids)
    result = self.engine.step(self.session, self.item_bank, self.response_provider, self.context)
    if self.session.status == SessionStatus.IN_PROGRESS:
      if result.selected_item_id is not None:
        assert len(self.session.theta_history) == prev_hist + 1
        assert len(self.session.administered_item_ids) == prev_admin + 1
        assert np.isfinite(self.session.current_theta)
    else:
      assert result.stopped is True
      assert self.session.stop_reason in {"stop_rule", "item_bank_exhausted", "selector_stopped"}

  @precondition(lambda self: self.session is not None and self.session.status == SessionStatus.STOPPED)
  @rule()
  def noop_after_stop(self) -> None:
    """Allow Hypothesis to continue stepping once the session is terminal."""

  @invariant()
  def session_lists_consistent(self) -> None:
    if self.session is None:
      return
    s = self.session
    n = self.item_bank.n_items
    for i in s.administered_item_ids:
      assert 0 <= int(i) < n
    assert len(s.responses) == len(s.administered_item_ids)
    assert len(set(s.administered_item_ids)) == len(s.administered_item_ids)
    assert len(s.theta_history) == 1 + len(s.administered_item_ids)
    if s.status == SessionStatus.STOPPED:
      assert s.stop_reason in {"stop_rule", "item_bank_exhausted", "selector_stopped"}


def test_cat_engine_orchestration_state_machine() -> None:
  run_state_machine_as_test(CatEngineOrchestrationMachine, settings=_ORCHESTRATION_SETTINGS)


@pytest.mark.slow
def test_cat_engine_integration_state_machine() -> None:
  run_state_machine_as_test(CatEngineIntegrationMachine, settings=_INTEGRATION_SETTINGS)
