"""Tests for the explicit CAT engine architecture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from catsim.engine import CatEngine, RunContext, SimulatedResponseProvider
from catsim.exceptions import NoItemsAvailableError
from catsim.selection import RandomSelector
from catsim.state import CatSessionState, ExposureTracker, SessionStatus
from catsim.stopping import TestLengthStopper as LengthStopper


@dataclass
class FixedResponseProvider:
  """Simple response provider for deterministic engine tests."""

  response: bool

  def answer(self, state, item_bank, item_id, context) -> bool:  # noqa: ARG002
    return self.response


class NullSelector:
  """Selector stub that explicitly stops item selection."""

  def select(self, item_bank, administered_items, est_theta, rng=None, exposure_rates=None):  # noqa: ARG002
    return None


class ExhaustedSelector:
  """Selector stub that behaves like an exhausted item bank."""

  def select(self, item_bank, administered_items, est_theta, rng=None, exposure_rates=None):  # noqa: ARG002
    msg = "No items remain"
    raise NoItemsAvailableError(msg)


@dataclass
class RecordingInitializer:
  """Initializer stub that records the arguments it receives."""

  theta: float
  seen_item_bank: object | None = None
  seen_rng: object | None = None

  def initialize(self, *, item_bank, rng) -> float:
    self.seen_item_bank = item_bank
    self.seen_rng = rng
    return self.theta


@dataclass
class RecordingSelector:
  """Selector stub that records the selection inputs."""

  item_id: int | None
  seen_item_bank: object | None = None
  seen_administered_items: object | None = None
  seen_theta: float | None = None
  seen_rng: object | None = None
  seen_exposure_rates: object | None = None

  def select(self, item_bank, administered_items, est_theta, rng=None, exposure_rates=None):
    self.seen_item_bank = item_bank
    self.seen_administered_items = administered_items
    self.seen_theta = est_theta
    self.seen_rng = rng
    self.seen_exposure_rates = exposure_rates
    return self.item_id


@dataclass
class RecordingEstimator:
  """Estimator stub that records the estimation inputs."""

  theta: float
  seen_item_bank: object | None = None
  seen_administered_items: object | None = None
  seen_response_vector: object | None = None
  seen_theta: float | None = None

  def estimate(self, *, item_bank, administered_items, response_vector, est_theta) -> float:
    self.seen_item_bank = item_bank
    self.seen_administered_items = administered_items
    self.seen_response_vector = response_vector
    self.seen_theta = est_theta
    return self.theta


@dataclass
class RecordingStopper:
  """Stopper stub that records the stopping inputs."""

  should_stop: bool
  seen_item_bank: object | None = None
  seen_administered_items: object | None = None
  seen_theta: float | None = None

  def stop(self, *, item_bank, administered_items, theta) -> bool:
    self.seen_item_bank = item_bank
    self.seen_administered_items = administered_items
    self.seen_theta = theta
    return self.should_stop


@dataclass
class FakeRng:
  """Minimal RNG stub for deterministic response-provider tests."""

  sample: float

  def uniform(self) -> float:
    return self.sample


@dataclass
class FakeItemBank:
  """Minimal item-bank stub for response-provider tests."""

  item: np.ndarray
  seen_item_id: int | None = None

  def get_item(self, item_id: int) -> np.ndarray:
    self.seen_item_id = item_id
    return self.item


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


def test_start_session_passes_item_bank_and_rng_to_initializer(item_bank) -> None:
  """Engine should forward the configured item bank and RNG to the initializer."""
  initializer = RecordingInitializer(theta=1.25)
  selector = RandomSelector()
  estimator = pytest.importorskip("catsim.estimation").NumericalSearchEstimator()
  stopper = LengthStopper(max_items=5)
  engine = CatEngine(initializer, selector, estimator, stopper)
  rng = np.random.default_rng(123)
  context = RunContext(rng=rng)

  session = engine.start_session(session_id=7, item_bank=item_bank, context=context, true_theta=-0.5)

  assert initializer.seen_item_bank is item_bank
  assert initializer.seen_rng is rng
  assert session.session_id == 7
  assert session.current_theta == pytest.approx(1.25)
  assert session.true_theta == pytest.approx(-0.5)
  assert session.theta_history == [pytest.approx(1.25)]


def test_should_stop_passes_runtime_state_to_stopper(item_bank) -> None:
  """Stopping checks should forward the item bank, administered items, and latest theta."""
  stopper = RecordingStopper(should_stop=True)
  engine = CatEngine(RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.0), stopper)
  state = CatSessionState(session_id=3, current_theta=1.75)
  state.administered_item_ids.extend([2, 4])

  stopped = engine.should_stop(state, item_bank)

  assert stopped is True
  assert stopper.seen_item_bank is item_bank
  assert stopper.seen_administered_items is state.administered_item_ids
  assert stopper.seen_theta == pytest.approx(1.75)


def test_select_next_passes_complete_context_to_selector(item_bank) -> None:
  """Selection should forward state, RNG, and exposure rates when available."""
  selector = RecordingSelector(item_id=4)
  tracker = ExposureTracker(item_bank.n_items)
  tracker.record(0)
  tracker.record(0)
  tracker.record(3)
  rng = np.random.default_rng(99)
  context = RunContext(rng=rng, exposure_tracker=tracker, total_sessions=4)
  engine = CatEngine(RecordingInitializer(0.0), selector, RecordingEstimator(0.0), RecordingStopper(False))
  state = CatSessionState(session_id=0, current_theta=-0.25)
  state.administered_item_ids.extend([1, 2])

  selected = engine.select_next(state, item_bank, context)

  assert selected == 4
  assert selector.seen_item_bank is item_bank
  assert selector.seen_administered_items is state.administered_item_ids
  assert selector.seen_theta == pytest.approx(-0.25)
  assert selector.seen_rng is rng
  assert selector.seen_exposure_rates is not None
  assert np.allclose(selector.seen_exposure_rates, tracker.rates(4))


def test_select_next_passes_none_exposure_rates_without_total_sessions(item_bank) -> None:
  """Selection should not derive exposure rates without the total session count."""
  selector = RecordingSelector(item_id=2)
  tracker = ExposureTracker(item_bank.n_items)
  rng = np.random.default_rng(11)
  context = RunContext(rng=rng, exposure_tracker=tracker, total_sessions=None)
  engine = CatEngine(RecordingInitializer(0.0), selector, RecordingEstimator(0.0), RecordingStopper(False))
  state = CatSessionState(session_id=0, current_theta=0.5)

  selected = engine.select_next(state, item_bank, context)

  assert selected == 2
  assert selector.seen_rng is rng
  assert selector.seen_exposure_rates is None


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


def test_apply_response_records_exposure_and_returns_complete_step_result(item_bank) -> None:
  """Applying a response should record exposure and return a faithful step result."""
  estimator = RecordingEstimator(theta=1.1)
  tracker = ExposureTracker(item_bank.n_items)
  engine = CatEngine(RecordingInitializer(0.0), RecordingSelector(0), estimator, RecordingStopper(False))
  context = RunContext(rng=np.random.default_rng(42), exposure_tracker=tracker, total_sessions=5)
  session = engine.start_session(session_id=1, item_bank=item_bank, context=context, true_theta=0.2)

  result = engine.apply_response(session, item_bank, item_id=3, response=True, context=context)

  assert tracker.counts[3] == 1
  assert tracker.counts.sum() == 1
  assert estimator.seen_item_bank is item_bank
  assert estimator.seen_administered_items is session.administered_item_ids
  assert estimator.seen_response_vector is session.responses
  assert estimator.seen_theta == pytest.approx(0.0)
  assert session.status == SessionStatus.IN_PROGRESS
  assert session.stop_reason is None
  assert result.selected_item_id == 3
  assert result.response is True
  assert result.updated_theta == pytest.approx(1.1)
  assert result.stopped is False
  assert result.stop_reason is None
  assert result.session.status == SessionStatus.IN_PROGRESS
  assert result.session.stop_reason is None
  assert result.session.administered_item_ids == [3]
  assert result.session.responses == [True]
  assert result.session.theta_history == [0.0, pytest.approx(1.1)]


def test_apply_response_sets_stop_fields_when_stopper_trips(item_bank) -> None:
  """Applying a response should mark both session and result as stopped when required."""
  engine = CatEngine(
    RecordingInitializer(0.0),
    RecordingSelector(0),
    RecordingEstimator(theta=0.3),
    RecordingStopper(True),
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  result = engine.apply_response(session, item_bank, item_id=0, response=False, context=context)

  assert session.status == SessionStatus.STOPPED
  assert session.stop_reason == "stop_rule"
  assert result.stopped is True
  assert result.stop_reason == "stop_rule"
  assert result.updated_theta == pytest.approx(0.3)
  assert result.session.status == SessionStatus.STOPPED
  assert result.session.stop_reason == "stop_rule"


def test_apply_response_returns_immutable_session_snapshot(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
  fixed_length_stopper,
) -> None:
  """Step results should preserve the session state observed at that step."""
  engine = CatEngine(fixed_initializer, random_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  first = engine.apply_response(session, item_bank, item_id=0, response=True, context=context)
  session.metadata["phase"] = "mutated"
  second = engine.apply_response(session, item_bank, item_id=1, response=False, context=context)

  assert first.session.administered_item_ids == [0]
  assert first.session.responses == [True]
  assert len(first.session.theta_history) == 2
  assert first.session.metadata == {}
  assert second.session.administered_item_ids == [0, 1]
  assert second.session.responses == [True, False]


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

  with pytest.raises(ValueError, match="true_theta is required") as exc_info:
    provider.answer(state, item_bank, 0, context)

  assert str(exc_info.value) == "true_theta is required to simulate responses"


def test_simulated_response_provider_uses_item_parameters_and_rng_threshold(monkeypatch) -> None:
  """Simulated responses should use the selected item parameters and RNG threshold."""
  provider = SimulatedResponseProvider()
  item_bank = FakeItemBank(item=np.array([1.2, -0.5, 0.2, 0.9]))
  context = RunContext(rng=FakeRng(sample=0.4))  # type: ignore[arg-type]
  state = CatSessionState(session_id=0, current_theta=0.0, true_theta=1.5)
  seen_args: dict[str, object] = {}

  def fake_icc(theta, a, b, c, d) -> float:
    seen_args["theta"] = theta
    seen_args["params"] = (a, b, c, d)
    return 0.8

  monkeypatch.setattr("catsim.engine.icc", fake_icc)

  response = provider.answer(state, item_bank, 7, context)

  assert response is True
  assert item_bank.seen_item_id == 7
  assert seen_args["theta"] == pytest.approx(1.5)
  assert seen_args["params"] == pytest.approx((1.2, -0.5, 0.2, 0.9))


def test_simulated_response_provider_returns_false_below_threshold(monkeypatch) -> None:
  """Simulated responses should be false when the sampled threshold exceeds the probability."""
  provider = SimulatedResponseProvider()
  item_bank = FakeItemBank(item=np.array([0.8, 0.3, 0.1, 1.0]))
  context = RunContext(rng=FakeRng(sample=0.9))  # type: ignore[arg-type]
  state = CatSessionState(session_id=0, current_theta=0.0, true_theta=-0.7)

  monkeypatch.setattr("catsim.engine.icc", lambda *_args: 0.2)

  assert provider.answer(state, item_bank, 2, context) is False


def test_simulated_response_provider_returns_true_when_probability_matches_threshold(monkeypatch) -> None:
  """Simulated responses should treat an exact threshold match as a positive response."""
  provider = SimulatedResponseProvider()
  item_bank = FakeItemBank(item=np.array([1.0, 0.0, 0.0, 1.0]))
  context = RunContext(rng=FakeRng(sample=0.4))  # type: ignore[arg-type]
  state = CatSessionState(session_id=0, current_theta=0.0, true_theta=0.3)

  monkeypatch.setattr("catsim.engine.icc", lambda *_args: 0.4)

  assert provider.answer(state, item_bank, 5, context) is True


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

  with pytest.raises(
    ValueError,
    match=rf"Invalid item index {item_bank.n_items}\. Must be between 0 and {item_bank.n_items - 1}",
  ):
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

  with pytest.raises(ValueError, match=r"Item 0 has already been administered in this session"):
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
  assert result.response is None
  assert result.updated_theta == pytest.approx(session.current_theta)
  assert result.stop_reason == "stop_rule"
  assert result.session.status == SessionStatus.STOPPED
  assert result.session.stop_reason == "stop_rule"
  assert session.status == SessionStatus.STOPPED
  assert session.stop_reason == "stop_rule"


def test_step_stops_when_selector_returns_none(item_bank, fixed_initializer, estimator) -> None:
  """Selectors can explicitly signal that the session should stop."""
  engine = CatEngine(fixed_initializer, NullSelector(), estimator, LengthStopper(max_items=10))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  result = engine.step(session, item_bank, FixedResponseProvider(True), context)

  assert result.stopped is True
  assert result.selected_item_id is None
  assert result.response is None
  assert result.updated_theta == pytest.approx(session.current_theta)
  assert result.stop_reason == "selector_stopped"
  assert result.session.status == SessionStatus.STOPPED
  assert result.session.stop_reason == "selector_stopped"
  assert session.status == SessionStatus.STOPPED


def test_step_stops_when_selector_raises_no_items(item_bank, fixed_initializer, estimator) -> None:
  """Exhausted selectors should map to an item-bank-exhausted stop reason."""
  engine = CatEngine(fixed_initializer, ExhaustedSelector(), estimator, LengthStopper(max_items=10))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  result = engine.step(session, item_bank, FixedResponseProvider(True), context)

  assert result.stopped is True
  assert result.selected_item_id is None
  assert result.response is None
  assert result.updated_theta == pytest.approx(session.current_theta)
  assert result.stop_reason == "item_bank_exhausted"
  assert result.session.status == SessionStatus.STOPPED
  assert result.session.stop_reason == "item_bank_exhausted"
  assert session.status == SessionStatus.STOPPED
