"""Tests for the explicit CAT engine architecture."""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import pytest

from catsim.engine import CatEngine, RunContext, SimulatedResponseProvider
from catsim.estimation import NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import RandomSelector
from catsim.state import CatSessionState, ExposureTracker, StopReason
from catsim.stopping import BaseStopper, MaxLengthStopper


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


class FailingEstimator:
  """Estimator stub that fails on every call."""

  def estimate(self, *, item_bank, administered_items, response_vector, est_theta) -> float:
    del item_bank, administered_items, response_vector, est_theta
    msg = "estimation failed"
    raise ValueError(msg)


class FailingStopper(BaseStopper):
  """Stopper stub that fails on every call."""

  def stop(self, item_bank, administered_items, theta) -> bool:
    del item_bank, administered_items, theta
    msg = "stopping failed"
    raise ValueError(msg)


class CountingLengthStopper(BaseStopper):
  """Stopper that ends the test at a fixed length and counts how often it is asked."""

  def __init__(self, max_items: int) -> None:
    self.max_items = max_items
    self.calls = 0

  def stop(self, item_bank, administered_items, theta) -> bool:
    del item_bank, theta
    self.calls += 1
    return len(administered_items) >= self.max_items


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
  assert session.stop_reason is None


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
  assert estimator.seen_administered_items == [3]
  assert estimator.seen_response_vector == [True]
  assert estimator.seen_theta == pytest.approx(0.0)
  assert session.stop_reason is None
  assert result.selected_item_id == 3
  assert result.response is True
  assert result.updated_theta == pytest.approx(1.1)
  assert result.stopped is False
  assert result.stop_reason is None
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

  assert session.stop_reason is StopReason.STOPPER
  assert result.stopped is True
  assert result.stop_reason is StopReason.STOPPER
  assert result.updated_theta == pytest.approx(0.3)
  assert result.session.stop_reason is StopReason.STOPPER


@pytest.mark.parametrize(
  ("estimator", "stopper", "error"),
  [
    (FailingEstimator(), RecordingStopper(False), "estimation failed"),
    (RecordingEstimator(0.5), FailingStopper(), "stopping failed"),
  ],
  ids=["estimator", "stopper"],
)
def test_apply_response_leaves_the_session_unchanged_when_a_strategy_fails(
  item_bank, estimator, stopper, error
) -> None:
  """A response whose estimate or stop decision fails is not applied at all.

  Fails if the item, the response or its exposure is recorded before the estimator and the
  stopper run: the half-applied step left the session with one more response than estimates, and
  retrying the item was rejected as already administered.
  """
  tracker = ExposureTracker(item_bank.n_items)
  context = RunContext(rng=np.random.default_rng(42), exposure_tracker=tracker)
  engine = CatEngine(RecordingInitializer(0.0), RecordingSelector(0), estimator, stopper)
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)
  before = copy.deepcopy(session)

  with pytest.raises(ValueError, match=error):
    engine.apply_response(session, item_bank, item_id=3, response=True, context=context)

  assert session == before
  assert tracker.counts.sum() == 0


@pytest.mark.parametrize("response", [1, 0, 1.0, "yes", None])
def test_apply_response_rejects_a_response_that_is_not_a_bool(item_bank, response) -> None:
  """Only ``bool`` and ``numpy.bool_`` responses are accepted, and a rejected one changes nothing.

  Fails if 1, 0 or 1.0 are stored as given in the session's list of booleans, or if "yes" or None
  reach the estimator.
  """
  tracker = ExposureTracker(item_bank.n_items)
  context = RunContext(rng=np.random.default_rng(42), exposure_tracker=tracker)
  engine = CatEngine(RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.5), RecordingStopper(False))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)
  before = copy.deepcopy(session)

  with pytest.raises(TypeError, match="response must be a bool"):
    engine.apply_response(session, item_bank, item_id=3, response=response, context=context)

  assert session == before
  assert tracker.counts.sum() == 0


def test_apply_response_rejects_a_stopped_session(item_bank) -> None:
  """A session that has stopped accepts no more responses.

  Fails if apply_response appends an item to a session its stopper has already ended.
  """
  engine = CatEngine(
    RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.5), CountingLengthStopper(max_items=1)
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)
  engine.apply_response(session, item_bank, item_id=0, response=True, context=context)
  before = copy.deepcopy(session)

  with pytest.raises(ValueError, match=r"Session 0 has stopped \(reason: stopper\)"):
    engine.apply_response(session, item_bank, item_id=1, response=True, context=context)

  assert session == before


def test_session_state_holds_builtin_python_values(item_bank) -> None:
  """The engine stores Python scalars whatever numeric types the strategies and the examinee return.

  Fails if an int initial theta, a ``numpy.float64`` estimate (as NumericalSearchEstimator
  returns), a ``numpy.int64`` item index (from the selector, or passed to apply_response by the
  caller), a ``numpy.bool_`` response or a ``numpy.bool_`` stop decision is stored as given.
  """
  engine = CatEngine(
    RecordingInitializer(theta=0),
    RecordingSelector(item_id=np.int64(3)),
    RecordingEstimator(theta=np.float64(0.25)),
    RecordingStopper(np.False_),
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)

  result = engine.step(session, item_bank, FixedResponseProvider(np.True_), context)
  engine.apply_response(session, item_bank, item_id=np.int64(5), response=np.False_, context=context)

  assert session.theta_history == [0.0, 0.25, 0.25]
  assert [type(theta) for theta in session.theta_history] == [float, float, float]
  assert type(session.current_theta) is float
  assert session.administered_item_ids == [3, 5]
  assert [type(item) for item in session.administered_item_ids] == [int, int]
  assert session.responses == [True, False]
  assert [type(response) for response in session.responses] == [bool, bool]
  assert (type(result.selected_item_id), type(result.response), type(result.updated_theta), type(result.stopped)) == (
    int,
    bool,
    float,
    bool,
  )


@pytest.mark.parametrize("item_id", [2.0, "3", None])
def test_apply_response_rejects_an_item_index_that_is_not_an_integer(item_bank, item_id) -> None:
  """An item index that is not an integer is rejected, and changes nothing.

  Fails if 2.0 passes the range check and is stored as a float, or if a string or None reaches the
  comparisons.
  """
  context = RunContext(rng=np.random.default_rng(42))
  engine = CatEngine(RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.5), RecordingStopper(False))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)
  before = copy.deepcopy(session)

  with pytest.raises(TypeError, match="item_id must be an integer"):
    engine.apply_response(session, item_bank, item_id=item_id, response=True, context=context)

  assert session == before


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


def test_step_applies_the_providers_response_to_the_selected_item(item_bank) -> None:
  """One step administers the item the selector chose and records the provider's response to it.

  Fails if step applies a different item or response, or skips the estimate.
  """
  engine = CatEngine(RecordingInitializer(0.0), RecordingSelector(4), RecordingEstimator(0.7), RecordingStopper(False))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)

  result = engine.step(session, item_bank, FixedResponseProvider(False), context)

  assert (result.selected_item_id, result.response, result.updated_theta) == (4, False, pytest.approx(0.7))
  assert session.administered_item_ids == [4]
  assert session.responses == [False]
  assert session.theta_history == [0.0, pytest.approx(0.7)]


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

  assert final_session.stop_reason is StopReason.STOPPER
  assert len(final_session.administered_item_ids) == 3


def test_the_stopper_is_evaluated_once_per_session_state(item_bank, max_info_selector) -> None:
  """A session of n items asks its stopper n + 1 times: about the initial state and after each response.

  Fails if step asks the stopper again before selecting, about the state apply_response has just
  evaluated, which doubled the calls (a 5-item session made 10).
  """
  stopper = CountingLengthStopper(max_items=5)
  engine = CatEngine(RecordingInitializer(0.0), max_info_selector, RecordingEstimator(0.0), stopper)
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)

  engine.run_session(session, item_bank, FixedResponseProvider(True), context)

  assert session.administered_count == 5
  assert stopper.calls == 6


def test_select_next_passes_response_history(
  item_bank,
  fixed_initializer,
  estimator,
  fixed_length_stopper,
) -> None:
  """Selectors should receive the current response history from the engine."""
  selector = RecordingSelector(item_id=2)
  engine = CatEngine(fixed_initializer, selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42), total_sessions=10)
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)
  session.administered_item_ids.extend([0, 1])
  session.responses.extend([True, False])

  selected = engine.select_next(session, item_bank, context)

  assert selected == 2
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


def test_stepping_a_stopped_session_keeps_its_stop_reason(item_bank) -> None:
  """Stepping a session that has already stopped changes nothing about it.

  Fails if the engine decides again why a stopped session ended: a session whose selector ran out
  was relabelled as stopped by its selector on the next step.
  """
  engine = CatEngine(RecordingInitializer(0.0), ExhaustedSelector(), RecordingEstimator(0.0), RecordingStopper(False))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)
  engine.step(session, item_bank, FixedResponseProvider(True), context)

  result = engine.step(session, item_bank, FixedResponseProvider(True), context)

  assert (result.stopped, result.selected_item_id, result.stop_reason) == (True, None, StopReason.NO_ITEMS)
  assert session.stop_reason is StopReason.NO_ITEMS
  assert session.administered_item_ids == []


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


def test_step_ends_a_session_whose_initial_state_satisfies_the_stopper(item_bank) -> None:
  """A stopper satisfied before the first item ends the session without selecting one.

  Fails if the stopper is not asked about the initial state, so the session gets an item it should
  not, or if the terminal step result misreports the session.
  """
  selector = RecordingSelector(item_id=0)
  engine = CatEngine(RecordingInitializer(0.4), selector, RecordingEstimator(0.0), RecordingStopper(True))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)

  result = engine.step(session, item_bank, FixedResponseProvider(True), context)

  assert selector.seen_item_bank is None
  assert (result.stopped, result.selected_item_id, result.response) == (True, None, None)
  assert result.updated_theta == pytest.approx(0.4)
  assert result.stop_reason is StopReason.STOPPER
  assert result.session.stop_reason is StopReason.STOPPER
  assert session.stop_reason is StopReason.STOPPER


@pytest.mark.parametrize("selector", [ExhaustedSelector(), NullSelector()], ids=["raises", "returns-none"])
def test_select_next_ends_the_session_when_the_selector_has_no_item(item_bank, selector) -> None:
  """A selector with no item to offer ends the session with ``NO_ITEMS``, however it says so.

  Fails if select_next lets ``NoItemsAvailableError`` escape, or returns None, while leaving the
  session in progress: a manual loop then had to copy step's handling, and notebook 02's did not.
  """
  engine = CatEngine(RecordingInitializer(0.0), selector, RecordingEstimator(0.0), RecordingStopper(False))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context)

  assert engine.select_next(session, item_bank, context) is None
  assert session.stop_reason is StopReason.NO_ITEMS
