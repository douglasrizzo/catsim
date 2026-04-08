"""Tests for the explicit CAT engine architecture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from catsim.engine import CatEngine, RunContext, SimulatedResponseProvider
from catsim.exceptions import NoItemsAvailableError
from catsim.state import CatSessionState, SessionStatus
from catsim.stopping import TestLengthStopper as LengthStopper


@dataclass
class FixedResponseProvider:
  """Simple response provider for deterministic engine tests."""

  response: bool

  def answer(self, state, item_bank, item_id, context) -> bool:  # noqa: ARG002
    return self.response


class NullSelector:
  """Selector stub that explicitly stops item selection."""

  def select(self, item_bank, administered_items, est_theta=None, **kwargs):  # noqa: ARG002
    return None


class ExhaustedSelector:
  """Selector stub that behaves like an exhausted item bank."""

  def select(self, item_bank, administered_items, est_theta=None, **kwargs):  # noqa: ARG002
    msg = "No items remain"
    raise NoItemsAvailableError(msg)


def test_start_session_initializes_theta_and_state(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
  fixed_length_stopper,
  rng,
) -> None:
  """Engine should create a live session with the initializer theta."""
  engine = CatEngine(fixed_initializer, random_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=rng)

  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=1.0)

  assert session.session_id == 0
  assert session.current_theta == 0.0
  assert session.true_theta == 1.0
  assert session.theta_history == [0.0]
  assert session.status == SessionStatus.IN_PROGRESS


def test_apply_response_updates_session_histories(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
  fixed_length_stopper,
  rng,
) -> None:
  """Applying a response should append item, response, and theta history."""
  engine = CatEngine(fixed_initializer, random_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=rng)
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  result = engine.apply_response(session, item_bank, item_id=0, response=True, context=context)

  assert result.selected_item_id == 0
  assert result.response is True
  assert session.administered_item_ids == [0]
  assert session.responses == [True]
  assert len(session.theta_history) == 2
  assert session.current_theta == session.theta_history[-1]


def test_step_executes_full_transition(
  item_bank,
  fixed_initializer,
  max_info_selector,
  estimator,
  fixed_length_stopper,
) -> None:
  """One engine step should select an item, answer it, and update theta."""
  engine = CatEngine(fixed_initializer, max_info_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42), total_sessions=1)
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)
  response_provider = FixedResponseProvider(True)

  result = engine.step(session, item_bank, response_provider, context)

  assert result.selected_item_id is not None
  assert result.response is True
  assert len(session.administered_item_ids) == 1
  assert len(session.responses) == 1
  assert len(session.theta_history) == 2


def test_run_session_stops_at_fixed_length(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
) -> None:
  """Run session should stop once the stopper condition is reached."""
  from catsim.stopping import TestLengthStopper

  stopper = TestLengthStopper(max_items=3)
  engine = CatEngine(fixed_initializer, random_selector, estimator, stopper)
  context = RunContext(rng=np.random.default_rng(42), total_sessions=1)
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  final_session = engine.run_session(session, item_bank, FixedResponseProvider(True), context)

  assert final_session.status == SessionStatus.STOPPED
  assert final_session.stop_reason == "stop_rule"
  assert len(final_session.administered_item_ids) == 3


def test_select_next_passes_exposure_snapshot(
  item_bank,
  fixed_initializer,
  max_info_selector,
  estimator,
  fixed_length_stopper,
) -> None:
  """Selectors should be able to consume explicit exposure-rate snapshots."""
  engine = CatEngine(fixed_initializer, max_info_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42), total_sessions=10)
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  selected = engine.select_next(session, item_bank, context)

  assert selected is not None
  assert 0 <= selected < item_bank.n_items


def test_simulated_response_provider_requires_true_theta(item_bank) -> None:
  """Simulated responses require a true theta on the session."""
  provider = SimulatedResponseProvider()
  context = RunContext(rng=np.random.default_rng(42))
  state = CatSessionState(session_id=0, current_theta=0.0)

  with pytest.raises(ValueError, match="true_theta is required"):
    provider.answer(state, item_bank, 0, context)


def test_select_next_returns_none_for_stopped_session(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
  fixed_length_stopper,
) -> None:
  """Stopped sessions should not request another item."""
  engine = CatEngine(fixed_initializer, random_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)
  session.status = SessionStatus.STOPPED

  assert engine.select_next(session, item_bank, context) is None


def test_apply_response_rejects_invalid_item_index(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
  fixed_length_stopper,
) -> None:
  """Applying a response should validate item ids."""
  engine = CatEngine(fixed_initializer, random_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  with pytest.raises(ValueError, match="Invalid item index"):
    engine.apply_response(session, item_bank, item_bank.n_items, True, context)


def test_apply_response_rejects_duplicate_items(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
  fixed_length_stopper,
) -> None:
  """A session cannot administer the same item twice."""
  engine = CatEngine(fixed_initializer, random_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)
  session.administered_item_ids.append(0)
  session.responses.append(True)

  with pytest.raises(ValueError, match="has already been administered"):
    engine.apply_response(session, item_bank, 0, False, context)


def test_step_stops_immediately_when_stop_rule_is_already_met(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
) -> None:
  """The engine should stop before selecting when the rule is already met."""
  engine = CatEngine(fixed_initializer, random_selector, estimator, LengthStopper(max_items=1))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)
  session.administered_item_ids.append(0)

  result = engine.step(session, item_bank, FixedResponseProvider(True), context)

  assert result.stopped is True
  assert result.selected_item_id is None
  assert session.status == SessionStatus.STOPPED
  assert session.stop_reason == "stop_rule"


def test_step_stops_when_selector_returns_none(item_bank, fixed_initializer, estimator) -> None:
  """Selectors can explicitly signal that the session should stop."""
  engine = CatEngine(fixed_initializer, NullSelector(), estimator, LengthStopper(max_items=10))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  result = engine.step(session, item_bank, FixedResponseProvider(True), context)

  assert result.stopped is True
  assert result.stop_reason == "selector_stopped"
  assert session.status == SessionStatus.STOPPED


def test_step_stops_when_selector_raises_no_items(item_bank, fixed_initializer, estimator) -> None:
  """Exhausted selectors should map to an item-bank-exhausted stop reason."""
  engine = CatEngine(fixed_initializer, ExhaustedSelector(), estimator, LengthStopper(max_items=10))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  result = engine.step(session, item_bank, FixedResponseProvider(True), context)

  assert result.stopped is True
  assert result.stop_reason == "item_bank_exhausted"
  assert session.status == SessionStatus.STOPPED
