"""Tests for the explicit CAT engine architecture."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from catsim.engine import CatEngine, RunContext, SimulatedResponseProvider
from catsim.estimation import NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import RandomSelector
from catsim.state import CatSessionState, ExposureTracker, SessionStatus
from catsim.stopping import MaxLengthStopper


@dataclass
class FixedResponseProvider:
  """Simple response provider for deterministic engine tests."""

  response: bool

  def answer(self, state, item_bank, item_id, context) -> bool:  # noqa: ARG002
    return self.response


class NullSelector:
  """Selector stub that explicitly stops item selection."""

  def select(
    self,
    item_bank,
    administered_items,
    est_theta,
    rng=None,
    exposure_rates=None,
    response_vector=None,
  ) -> None:
    del item_bank, administered_items, est_theta, rng, exposure_rates, response_vector


class ExhaustedSelector:
  """Selector stub that behaves like an exhausted item bank."""

  def select(
    self,
    item_bank,
    administered_items,
    est_theta,
    rng=None,
    exposure_rates=None,
    response_vector=None,
  ):
    del item_bank, administered_items, est_theta, rng, exposure_rates, response_vector
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
  seen_response_vector: object | None = None

  def select(self, item_bank, administered_items, est_theta, rng=None, exposure_rates=None, response_vector=None):
    self.seen_item_bank = item_bank
    self.seen_administered_items = administered_items
    self.seen_theta = est_theta
    self.seen_rng = rng
    self.seen_exposure_rates = exposure_rates
    self.seen_response_vector = None if response_vector is None else list(response_vector)
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


def test_start_session_passes_item_bank_and_rng_to_initializer(item_bank) -> None:
  """Engine should forward the configured item bank and RNG to the initializer and start a live session."""
  initializer = RecordingInitializer(theta=1.25)
  selector = RandomSelector()
  estimator = NumericalSearchEstimator()
  stopper = MaxLengthStopper(5)
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
  assert session.status == SessionStatus.IN_PROGRESS


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
  stopper = MaxLengthStopper(3)
  engine = CatEngine(fixed_initializer, random_selector, estimator, stopper)
  context = RunContext(rng=np.random.default_rng(42), total_sessions=1)
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  final_session = engine.run_session(session, item_bank, FixedResponseProvider(True), context)

  assert final_session.status == SessionStatus.STOPPED
  assert final_session.stop_reason == "stop_rule"
  assert len(final_session.administered_item_ids) == 3


def test_select_next_passes_response_history(
  item_bank,
  fixed_initializer,
  estimator,
  fixed_length_stopper,
) -> None:
  """Selectors should receive the current response history from the engine."""
  selector = RecordingSelector(item_id=None)
  engine = CatEngine(fixed_initializer, selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42), total_sessions=10)
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)
  session.administered_item_ids.extend([0, 1])
  session.responses.extend([True, False])

  selected = engine.select_next(session, item_bank, context)

  assert selected is None
  assert selector.seen_response_vector == [True, False]


def test_simulated_response_provider_requires_true_theta(item_bank) -> None:
  """Simulated responses require a true theta on the session."""
  provider = SimulatedResponseProvider()
  context = RunContext(rng=np.random.default_rng(42))
  state = CatSessionState(session_id=0, current_theta=0.0)

  with pytest.raises(ValueError, match="true_theta is required") as exc_info:
    provider.answer(state, item_bank, 0, context)

  assert str(exc_info.value) == "true_theta is required to simulate responses"


@pytest.mark.parametrize(
  ("item_id", "draw", "expected"),
  [
    # 2PL item at b = 0: P = 0.82; the estimate (-1.5) or item 0 would both give P below 0.2
    pytest.param(1, 0.5, True, id="2pl-correct"),
    pytest.param(1, 0.9, False, id="2pl-incorrect"),
    # 4PL item with c = 0.2, d = 0.9: P = 0.2 + 0.7 * 0.82 = 0.77; without c and d it would be 0.82
    pytest.param(2, 0.79, False, id="4pl-upper-asymptote"),
    # 3PL item at b = 5 with c = 0.3: P = 0.3 + 0.7 * 0.03 = 0.32; without c it would be 0.03
    pytest.param(3, 0.31, True, id="3pl-guessing"),
  ],
)
def test_simulated_response_is_correct_when_the_draw_is_below_the_true_theta_probability(
  item_id: int, draw: float, expected: bool
) -> None:
  """The response is correct when the uniform draw falls below the chosen item's 4PL probability at the true theta.

  Together the cases fail if the response used the estimate instead of the true theta (1.5), another item,
  or dropped the item's guessing or upper-asymptote parameter.
  """
  item_bank = ItemBank(
    np.array([[1.0, 5.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.2, 0.9], [1.0, 5.0, 0.3, 1.0]])
  )
  state = CatSessionState(session_id=0, current_theta=-1.5, true_theta=1.5)
  context = RunContext(rng=FakeRng(sample=draw))  # type: ignore[arg-type]

  assert SimulatedResponseProvider().answer(state, item_bank, item_id, context) is expected


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
  engine = CatEngine(fixed_initializer, random_selector, estimator, MaxLengthStopper(1))
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
  engine = CatEngine(fixed_initializer, NullSelector(), estimator, MaxLengthStopper(10))
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
  engine = CatEngine(fixed_initializer, ExhaustedSelector(), estimator, MaxLengthStopper(10))
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
