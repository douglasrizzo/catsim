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
  rule,
  run_state_machine_as_test,
)

from catsim import Estimate
from catsim.engine import CatEngine, RunContext
from catsim.estimation import NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector
from catsim.state import CatSessionState, StopReason
from catsim.stopping import MaxLengthStopper


@dataclass
class FixedResponseProvider:
  """Deterministic response provider for state machine runs."""

  response: bool

  def answer(self, item_bank, item_id, session, rng) -> bool:  # noqa: ARG002
    return self.response


@dataclass
class RecordingEstimator:
  """Estimator stub returning a fixed theta."""

  theta: float

  def estimate(self, context) -> Estimate:  # noqa: ARG002
    return Estimate(theta=float(self.theta), se=1.0)


class QueueOrderSelector:
  """Select the next item by scanning a fixed permutation, skipping administered ids."""

  def __init__(self, order: tuple[int, ...] | list[int]) -> None:
    self._order = tuple(order)

  def select(self, context):
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
    prev_hist = len(self.session.theta_history)
    prev_admin = len(self.session.administered_item_ids)
    was_stopped = self.session.is_stopped
    prev_reason = self.session.stop_reason
    result = self.engine.step(self.session, self.response_provider, self.context)
    if was_stopped:
      # Stepping a terminal session must be a no-op.
      assert result.stopped is True
      assert result.selected_item_id is None
      assert len(self.session.administered_item_ids) == prev_admin
      assert self.session.stop_reason == prev_reason
    elif not self.session.is_stopped:
      if result.selected_item_id is not None:
        assert len(self.session.theta_history) == prev_hist + 1
        assert len(self.session.administered_item_ids) == prev_admin + 1
        assert result.response is True
    else:
      assert result.stopped is True
      assert self.session.stop_reason is StopReason.STOPPER
      assert self.session.administered_count == self.max_items

  @invariant()
  def session_lists_consistent(self) -> None:
    s = self.session
    n = self.item_bank.n_items
    for i in s.administered_item_ids:
      assert 0 <= int(i) < n
    assert len(s.responses) == len(s.administered_item_ids)
    assert len(set(s.administered_item_ids)) == len(s.administered_item_ids)
    assert len(s.theta_history) == 1 + len(s.administered_item_ids)
    assert s.stop_reason in {None, StopReason.STOPPER}


class CatEngineIntegrationMachine(RuleBasedStateMachine):
  """Layer B: real MaxInfoSelector, bounded numerical estimator, alternating responses."""

  @initialize(
    bank_seed=st.integers(min_value=0, max_value=500),
    max_items=st.integers(min_value=3, max_value=5),
  )
  def init_engine(self, bank_seed: int, max_items: int) -> None:
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
    prev_hist = len(self.session.theta_history)
    prev_admin = len(self.session.administered_item_ids)
    was_stopped = self.session.is_stopped
    prev_reason = self.session.stop_reason
    result = self.engine.step(self.session, self.response_provider, self.context)
    if was_stopped:
      assert result.stopped is True
      assert result.selected_item_id is None
      assert len(self.session.administered_item_ids) == prev_admin
      assert self.session.stop_reason == prev_reason
    elif not self.session.is_stopped:
      if result.selected_item_id is not None:
        assert len(self.session.theta_history) == prev_hist + 1
        assert len(self.session.administered_item_ids) == prev_admin + 1
        assert np.isfinite(self.session.current_theta)
    else:
      assert result.stopped is True
      assert self.session.stop_reason is StopReason.STOPPER
      assert self.session.administered_count == self.max_items

  @invariant()
  def session_lists_consistent(self) -> None:
    s = self.session
    n = self.item_bank.n_items
    for i in s.administered_item_ids:
      assert 0 <= int(i) < n
    assert len(s.responses) == len(s.administered_item_ids)
    assert len(set(s.administered_item_ids)) == len(s.administered_item_ids)
    assert len(s.theta_history) == 1 + len(s.administered_item_ids)
    assert s.stop_reason in {None, StopReason.STOPPER}


def test_cat_engine_orchestration_state_machine() -> None:
  run_state_machine_as_test(CatEngineOrchestrationMachine, settings=_ORCHESTRATION_SETTINGS)


@pytest.mark.slow
def test_cat_engine_integration_state_machine() -> None:
  run_state_machine_as_test(CatEngineIntegrationMachine, settings=_INTEGRATION_SETTINGS)
