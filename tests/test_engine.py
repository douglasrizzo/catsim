"""Tests for the explicit CAT engine architecture."""

from __future__ import annotations

import copy
import dataclasses
import math
from dataclasses import FrozenInstanceError, dataclass

import numpy as np
import pytest

from catsim import Estimate, StepContext
from catsim.engine import CatEngine, RunContext, SimulatedResponseProvider
from catsim.estimation import BaseEstimator, NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.initialization import BaseInitializer
from catsim.item_bank import ItemBank
from catsim.selection import BaseSelector, LinearSelector, MaxInfoSelector, MEISelector, RandomSelector
from catsim.selection.base import remaining
from catsim.state import CatSessionState, ExposureTracker, StopReason
from catsim.stopping import BaseStopper, MaxLengthStopper
from tests.contexts import make_context


@dataclass
class FixedResponseProvider:
  """Simple response provider for deterministic engine tests."""

  response: bool

  def answer(self, state, item_bank, item_id, context) -> bool:  # noqa: ARG002
    return self.response


@dataclass
class ScriptedResponseProvider:
  """Response provider that gives the n-th scripted response to the n-th item, whichever item it is."""

  responses: list[bool]

  def answer(self, state, item_bank, item_id, context) -> bool:  # noqa: ARG002
    return self.responses[len(state.responses)]


class ExhaustedSelector:
  """Selector stub that behaves like an exhausted item bank."""

  def select(self, context: StepContext) -> int:
    del context
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
  """Selector stub that records the context it is given."""

  item_id: int
  seen: StepContext | None = None

  def select(self, context: StepContext) -> int:
    self.seen = context
    return self.item_id


@dataclass
class RecordingEstimator:
  """Estimator stub that records the context it is given."""

  theta: float
  se: float = 0.5
  evaluations: int | None = None
  seen: StepContext | None = None

  def estimate(self, context: StepContext) -> Estimate:
    self.seen = context
    return Estimate(theta=self.theta, se=self.se, evaluations=self.evaluations)


class FloatEstimator:
  """Estimator stub written for the contract where estimate returned a bare float."""

  def estimate(self, context: StepContext) -> float:
    del context
    return 0.5


class FailingEstimator:
  """Estimator stub that fails on every call."""

  def estimate(self, context: StepContext) -> Estimate:
    del context
    msg = "estimation failed"
    raise ValueError(msg)


class FailingInitializer:
  """Initializer stub that fails on every call."""

  def initialize(self, *, item_bank, rng) -> float:
    del item_bank, rng
    msg = "initialization failed"
    raise ValueError(msg)


class FailingStopper(BaseStopper):
  """Stopper stub that fails on every call."""

  def stop(self, context: StepContext) -> bool:
    del context
    msg = "stopping failed"
    raise ValueError(msg)


class CountingLengthStopper(BaseStopper):
  """Stopper that ends the test at a fixed length and counts how often it is asked."""

  def __init__(self, max_items: int) -> None:
    self.max_items = max_items
    self.calls = 0

  def stop(self, context: StepContext) -> bool:
    self.calls += 1
    return len(context.administered_items) >= self.max_items


@dataclass
class RecordingStopper:
  """Stopper stub that records the context it is given."""

  should_stop: bool
  seen: StepContext | None = None

  def stop(self, context: StepContext) -> bool:
    self.seen = context
    return self.should_stop


@dataclass
class FakeRng:
  """Minimal RNG stub for deterministic response-provider tests."""

  sample: float

  def uniform(self) -> float:
    return self.sample


class LowestIndexSelector(BaseSelector):
  """Custom selector: the remaining item with the lowest index."""

  def select(self, context: StepContext) -> int:
    return int(remaining(context)[0])


class CorrectCountEstimator(BaseEstimator):
  """Custom estimator: the number of correct responses."""

  def estimate(self, context: StepContext) -> Estimate:
    return Estimate(theta=float(sum(context.responses)), se=1.0)


class ThreeItemStopper(BaseStopper):
  """Custom stopper: three administered items."""

  def stop(self, context: StepContext) -> bool:
    return len(context.administered_items) >= 3


class HalfInitializer(BaseInitializer):
  """Custom initializer: every session starts at 0.5."""

  def initialize(self, *, item_bank: ItemBank, rng: np.random.Generator) -> float:
    del item_bank, rng
    return 0.5


def test_start_session_passes_item_bank_and_rng_to_initializer(item_bank) -> None:
  """start_session gives the initializer the bank and the run's generator, and starts a running session at its estimate.

  Fails if the initializer gets another bank or generator, or if the session starts with another estimate or true
  theta, or already stopped.
  """
  initializer = RecordingInitializer(theta=1.25)
  selector = RandomSelector()
  estimator = NumericalSearchEstimator()
  stopper = MaxLengthStopper(5)
  engine = CatEngine(item_bank, initializer, selector, estimator, stopper)
  rng = np.random.default_rng(123)
  context = RunContext(rng=rng)

  session = engine.start_session(session_id=7, context=context, true_theta=-0.5)

  assert initializer.seen_item_bank is item_bank
  assert initializer.seen_rng is rng
  assert session.session_id == 7
  assert session.current_theta == pytest.approx(1.25)
  assert session.true_theta == pytest.approx(-0.5)
  assert session.theta_history == (pytest.approx(1.25),)
  assert session.stop_reason is None


def test_start_session_counts_a_session_in_the_tracker_only_once_it_has_started(item_bank) -> None:
  """A session whose initializer raises is not counted in the exposure tracker; one that starts is.

  Fails if the tracker counts the session before the initializer runs: every later session would see rates divided
  by one session too many.
  """
  tracker = ExposureTracker(item_bank.n_items)
  context = RunContext(rng=np.random.default_rng(0), exposure_tracker=tracker)
  strategies = (RecordingSelector(0), RecordingEstimator(0.0), RecordingStopper(False))

  with pytest.raises(ValueError, match="initialization failed"):
    CatEngine(item_bank, FailingInitializer(), *strategies).start_session(0, context)
  CatEngine(item_bank, RecordingInitializer(0.0), *strategies).start_session(1, context)

  assert tracker.n_sessions == 1


def test_should_stop_passes_runtime_state_to_stopper(item_bank) -> None:
  """The stopper sees the session's bank, items, responses, latest estimate and its SE, and the run's generator.

  Fails if any of them is dropped or taken from elsewhere, such as the initial estimate instead of the current one.
  """
  stopper = RecordingStopper(should_stop=True)
  engine = CatEngine(item_bank, RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.0), stopper)
  state = CatSessionState(
    session_id=3,
    estimates=[Estimate(0.0, math.inf), Estimate(0.5, 0.9), Estimate(1.75, 0.6)],
    administered_item_ids=[2, 4],
    responses=[True, False],
  )
  rng = np.random.default_rng(5)

  stopped = engine.should_stop(state, RunContext(rng=rng))

  assert stopped is True
  assert stopper.seen.item_bank is item_bank
  assert stopper.seen.rng is rng
  assert (stopper.seen.administered_items, stopper.seen.responses) == ((2, 4), (True, False))
  assert (stopper.seen.theta, stopper.seen.se) == (1.75, 0.6)


def test_select_next_passes_complete_context_to_selector(item_bank) -> None:
  """select_next hands the selector the bank, the session's items, responses and current estimate with its SE, the
  generator, the rates and no excluded items.

  Two earlier sessions administered [0, 3] and [0], so the rates are 1 for item 0 and 0.5 for item 3. Fails if any
  input is dropped or taken from elsewhere, such as the initial estimate instead of the current one.
  """
  selector = RecordingSelector(item_id=4)
  tracker = ExposureTracker(item_bank.n_items)
  for items in ([0, 3], [0]):
    tracker.start_session()
    for item_id in items:
      tracker.record(item_id)
  tracker.start_session()
  rng = np.random.default_rng(99)
  context = RunContext(rng=rng, exposure_tracker=tracker)
  engine = CatEngine(item_bank, RecordingInitializer(0.0), selector, RecordingEstimator(0.0), RecordingStopper(False))
  state = CatSessionState(
    session_id=0,
    estimates=[Estimate(0.0, math.inf), Estimate(0.5, 0.9), Estimate(-0.25, 0.6)],
    administered_item_ids=[1, 2],
    responses=[True, False],
  )

  selected = engine.select_next(state, context)

  assert selected == 4
  assert selector.seen.item_bank is item_bank
  assert (selector.seen.administered_items, selector.seen.responses) == ((1, 2), (True, False))
  assert selector.seen.excluded_items == ()
  assert (selector.seen.theta, selector.seen.se) == (-0.25, 0.6)
  assert selector.seen.rng is rng
  expected_rates = np.zeros(item_bank.n_items)
  expected_rates[[0, 3]] = [1.0, 0.5]
  assert np.array_equal(selector.seen.exposure_rates, expected_rates)


def test_a_tracker_caps_exposure_across_sessions_with_no_other_setting() -> None:
  """A tracker in the context is all r_max needs: an item the earlier examinees reached is held back.

  At theta 0 the items rank 0, 1, 2, 3 by information, and each session takes one item. The second session sees
  item 0 at 1/1, the third sees items 0 and 1 at 1/2, both at the cap of 0.5, and the fourth sees items 0, 1 and 2
  at 1/3, below it. Fails if the selector gets no rates from a context that holds only a tracker (every session
  takes item 0), or if the rates divide by anything but the number of sessions started before the current one.
  """
  item_bank = ItemBank(np.array([[a, 0.0, 0.0, 1.0] for a in (2.0, 1.5, 1.0, 0.5)]))
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), MaxInfoSelector(r_max=0.5), RecordingEstimator(0.0), RecordingStopper(False)
  )
  context = RunContext(rng=np.random.default_rng(0), exposure_tracker=ExposureTracker(item_bank.n_items))

  first_items = []
  for session_id in range(4):
    session = engine.start_session(session_id, context)
    item_id = engine.select_next(session, context)
    engine.apply_response(session, item_id, True, context)
    first_items.append(item_id)

  assert first_items == [0, 1, 2, 0]


def test_select_next_passes_no_exposure_rates_without_a_tracker(item_bank) -> None:
  """Without an exposure tracker in the context, the selector gets no rates, and so applies no cap.

  Fails if the engine makes up rates for a run that does not track exposure, or fails on the missing tracker.
  """
  selector = RecordingSelector(item_id=2)
  engine = CatEngine(item_bank, RecordingInitializer(0.0), selector, RecordingEstimator(0.0), RecordingStopper(False))
  context = RunContext(rng=np.random.default_rng(11))
  session = engine.start_session(0, context)

  assert engine.select_next(session, context) == 2
  assert selector.seen.exposure_rates is None


def test_apply_response_records_exposure_and_returns_complete_step_result(item_bank) -> None:
  """apply_response counts the item once in the tracker and reports the step the session took.

  Fails if the administration is not recorded or recorded twice, if the estimator gets other items, responses or
  estimate than the session's, or if the result reports another item, response, estimate or stop state.
  """
  estimator = RecordingEstimator(theta=1.1)
  tracker = ExposureTracker(item_bank.n_items)
  # A non-zero start tells an estimator handed StepContext's default theta of 0 apart from one handed the session's.
  engine = CatEngine(item_bank, RecordingInitializer(0.7), RecordingSelector(0), estimator, RecordingStopper(False))
  context = RunContext(rng=np.random.default_rng(42), exposure_tracker=tracker)
  session = engine.start_session(session_id=1, context=context, true_theta=0.2)

  result = engine.apply_response(session, item_id=3, response=True, context=context)

  assert tracker.counts[3] == 1
  assert tracker.counts.sum() == 1
  assert estimator.seen.item_bank is item_bank
  assert (estimator.seen.administered_items, estimator.seen.responses) == ((3,), (True,))
  assert estimator.seen.theta == pytest.approx(0.7)
  assert session.stop_reason is None
  assert result.selected_item_id == 3
  assert result.response is True
  assert result.updated_theta == pytest.approx(1.1)
  assert result.stopped is False
  assert result.stop_reason is None
  assert result.session.stop_reason is None
  assert result.session.administered_item_ids == (3,)
  assert result.session.responses == (True,)
  assert result.session.theta_history == pytest.approx((0.7, 1.1))


def test_apply_response_starts_the_estimator_from_the_current_estimate_and_its_se(item_bank) -> None:
  """The estimator gets the session's current theta and SE, and the session then reports the ones it returned.

  The session's current estimate is 0.5 with an SE of 0.9, unlike StepContext's defaults of 0 and infinity. Fails if
  the estimator starts from the defaults or the initial estimate, or if current_se is not the SE the estimator
  returned.
  """
  estimator = RecordingEstimator(theta=1.1, se=0.42)
  engine = CatEngine(item_bank, RecordingInitializer(0.0), RecordingSelector(0), estimator, RecordingStopper(False))
  session = CatSessionState(
    session_id=0, estimates=[Estimate(0.0, math.inf), Estimate(0.5, 0.9)], administered_item_ids=[1], responses=[True]
  )

  engine.apply_response(session, item_id=3, response=False, context=RunContext(rng=np.random.default_rng(0)))

  assert (estimator.seen.theta, estimator.seen.se) == (0.5, 0.9)
  assert (session.current_theta, session.current_se) == (1.1, 0.42)


def test_apply_response_sets_stop_fields_when_stopper_trips(item_bank) -> None:
  """A stopper satisfied by the new state ends the session, and the step result reports it.

  Fails if the stopper judges a stale state (without the new item and response, or with the previous estimate or its
  SE), if the decision is dropped, so the session runs past its criterion, or if the result or its snapshot reports
  the session still running.
  """
  stopper = RecordingStopper(True)
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(theta=0.3, se=0.42), stopper
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context, true_theta=0.0)

  result = engine.apply_response(session, item_id=0, response=False, context=context)

  assert (stopper.seen.administered_items, stopper.seen.responses) == ((0,), (False,))
  assert (stopper.seen.theta, stopper.seen.se) == (0.3, 0.42)
  assert session.stop_reason is StopReason.STOPPER
  assert result.stopped is True
  assert result.stop_reason is StopReason.STOPPER
  assert result.updated_theta == pytest.approx(0.3)
  assert result.session.stop_reason is StopReason.STOPPER


@pytest.mark.parametrize(
  ("estimator", "stopper", "exception", "error"),
  [
    (FailingEstimator(), RecordingStopper(False), ValueError, "estimation failed"),
    (
      FloatEstimator(),
      RecordingStopper(False),
      TypeError,
      "FloatEstimator.estimate must return an Estimate, got float",
    ),
    (RecordingEstimator(0.5), FailingStopper(), ValueError, "stopping failed"),
  ],
  ids=["estimator", "estimator-returning-a-float", "stopper"],
)
def test_apply_response_leaves_the_session_unchanged_when_a_strategy_fails(
  item_bank, estimator, stopper, exception, error
) -> None:
  """A response whose estimate or stop decision fails is not applied at all.

  Fails if the item, the response or its exposure is recorded before the estimator and the
  stopper run: the half-applied step left the session with one more response than estimates, and
  retrying the item was rejected as already administered. An estimator that returns a float
  instead of an Estimate must fail with a TypeError that says so, not an AttributeError.
  """
  tracker = ExposureTracker(item_bank.n_items)
  context = RunContext(rng=np.random.default_rng(42), exposure_tracker=tracker)
  engine = CatEngine(item_bank, RecordingInitializer(0.0), RecordingSelector(0), estimator, stopper)
  session = engine.start_session(session_id=0, context=context)
  before = copy.deepcopy(session)

  with pytest.raises(exception, match=error):
    engine.apply_response(session, item_id=3, response=True, context=context)

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
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.5), RecordingStopper(False)
  )
  session = engine.start_session(session_id=0, context=context)
  before = copy.deepcopy(session)

  with pytest.raises(TypeError, match="response must be a bool"):
    engine.apply_response(session, item_id=3, response=response, context=context)

  assert session == before
  assert tracker.counts.sum() == 0


def test_apply_response_rejects_a_stopped_session(item_bank) -> None:
  """A session that has stopped accepts no more responses.

  Fails if apply_response appends an item to a session its stopper has already ended.
  """
  engine = CatEngine(
    item_bank,
    RecordingInitializer(0.0),
    RecordingSelector(0),
    RecordingEstimator(0.5),
    CountingLengthStopper(max_items=1),
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)
  engine.apply_response(session, item_id=0, response=True, context=context)
  before = copy.deepcopy(session)

  with pytest.raises(ValueError, match=r"Session 0 has stopped \(reason: stopper\)"):
    engine.apply_response(session, item_id=1, response=True, context=context)

  assert session == before


def test_session_state_holds_builtin_python_values(item_bank) -> None:
  """The engine stores Python scalars whatever numeric types the strategies and the examinee return.

  Fails if an int initial theta, an estimate whose theta, standard error or evaluation count is a NumPy scalar, a
  ``numpy.int64`` item index (from the selector, or passed to apply_response by the caller), a ``numpy.bool_``
  response or a ``numpy.bool_`` stop decision is stored as given.
  """
  engine = CatEngine(
    item_bank,
    RecordingInitializer(theta=0),
    RecordingSelector(item_id=np.int64(3)),
    RecordingEstimator(theta=np.float64(0.25), se=np.float64(0.4), evaluations=np.int64(7)),
    RecordingStopper(np.False_),
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)

  result = engine.step(session, FixedResponseProvider(np.True_), context)
  engine.apply_response(session, item_id=np.int64(5), response=np.False_, context=context)

  assert session.theta_history == (0.0, 0.25, 0.25)
  assert [type(theta) for theta in session.theta_history] == [float, float, float]
  assert type(session.current_theta) is float
  assert session.estimates[1:] == [Estimate(0.25, 0.4, 7)] * 2
  assert [(type(e.theta), type(e.se), type(e.evaluations)) for e in session.estimates[1:]] == [(float, float, int)] * 2
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
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.5), RecordingStopper(False)
  )
  session = engine.start_session(session_id=0, context=context)
  before = copy.deepcopy(session)

  with pytest.raises(TypeError, match="item_id must be an integer"):
    engine.apply_response(session, item_id=item_id, response=True, context=context)

  assert session == before


def test_a_session_rebuilt_from_a_snapshot_takes_the_next_step(item_bank) -> None:
  """A session rebuilt from a snapshot's fields continues like the original, and the snapshot stays as it was.

  Fails if the rebuilt state keeps the snapshot's tuples: apply_response appended the item and the response,
  then failed on the history tuple, leaving a step half-applied and the item impossible to retry.
  """
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.5), RecordingStopper(False)
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)
  engine.apply_response(session, item_id=3, response=True, context=context)
  snapshot = session.snapshot()
  rebuilt = CatSessionState(**{field.name: getattr(snapshot, field.name) for field in dataclasses.fields(snapshot)})

  engine.apply_response(rebuilt, item_id=4, response=False, context=context)

  assert (rebuilt.administered_item_ids, rebuilt.responses, rebuilt.theta_history) == (
    [3, 4],
    [True, False],
    (0.0, 0.5, 0.5),
  )
  assert (snapshot.administered_item_ids, snapshot.responses, snapshot.theta_history) == ((3,), (True,), (0.0, 0.5))


def test_a_step_result_keeps_the_session_as_the_step_left_it(item_bank) -> None:
  """A step result's snapshot does not change when the session takes further steps.

  Fails if the snapshot shares the session's lists, so the next response also shows up in the
  earlier result.
  """
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.5), RecordingStopper(False)
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)

  first = engine.apply_response(session, item_id=0, response=True, context=context)
  second = engine.apply_response(session, item_id=1, response=False, context=context)

  assert (first.session.administered_item_ids, first.session.responses) == ((0,), (True,))
  assert first.session.theta_history == (0.0, 0.5)
  assert (second.session.administered_item_ids, second.session.responses) == ((0, 1), (True, False))
  assert second.session.theta_history == (0.0, 0.5, 0.5)


def test_a_step_result_cannot_be_changed(item_bank) -> None:
  """Neither a step result's fields nor the sequences in its session snapshot can be changed.

  Fails if CatStepResult is not frozen, or if the snapshot holds lists: ADR-0005 promises immutable
  step results, but ``result.response = False`` and ``result.session.responses.append(True)`` both
  worked.
  """
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.5), RecordingStopper(False)
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)
  result = engine.apply_response(session, item_id=3, response=True, context=context)

  with pytest.raises(FrozenInstanceError, match="cannot assign to field 'response'"):
    result.response = False  # type: ignore[misc]
  for sequence in (result.session.administered_item_ids, result.session.responses, result.session.estimates):
    with pytest.raises(AttributeError, match="object has no attribute 'append'"):
      sequence.append(sequence[0])  # type: ignore[attr-defined]
  assert (result.session.administered_item_ids, result.session.responses) == ((3,), (True,))


def test_snapshots_of_the_same_session_state_are_equal_and_hash_alike(item_bank) -> None:
  """A snapshot is hashable: two snapshots of one state collapse into one set member, and a later state does not.

  Fails if a snapshot field is a list or a dict, which made ``hash(snapshot)`` raise TypeError.
  """
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), RecordingSelector(0), RecordingEstimator(0.5), RecordingStopper(False)
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)
  first = engine.apply_response(session, item_id=3, response=True, context=context)
  same_state = session.snapshot()
  engine.apply_response(session, item_id=4, response=False, context=context)

  assert hash(first.session) == hash(same_state)
  assert len({first.session, same_state, session.snapshot()}) == 2


def test_step_applies_the_providers_response_to_the_selected_item(item_bank) -> None:
  """One step administers the item the selector chose and records the provider's response to it.

  Fails if step applies a different item or response, or skips the estimate.
  """
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), RecordingSelector(4), RecordingEstimator(0.7), RecordingStopper(False)
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)

  result = engine.step(session, FixedResponseProvider(False), context)

  assert (result.selected_item_id, result.response, result.updated_theta) == (4, False, pytest.approx(0.7))
  assert session.administered_item_ids == [4]
  assert session.responses == [False]
  assert session.theta_history == (0.0, pytest.approx(0.7))


def test_run_session_stops_at_fixed_length(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
) -> None:
  """run_session runs until the stopper says stop: three items for a three-item test.

  Fails if it returns before the session ends or keeps administering items after the stopper is satisfied.
  """
  stopper = MaxLengthStopper(3)
  engine = CatEngine(item_bank, fixed_initializer, random_selector, estimator, stopper)
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context, true_theta=0.0)

  final_session = engine.run_session(session, FixedResponseProvider(True), context)

  assert final_session.stop_reason is StopReason.STOPPER
  assert len(final_session.administered_item_ids) == 3


def test_the_stopper_is_evaluated_once_per_session_state(item_bank, max_info_selector) -> None:
  """A session of n items asks its stopper n + 1 times: about the initial state and after each response.

  Fails if step asks the stopper again before selecting, about the state apply_response has just
  evaluated, which doubled the calls (a 5-item session made 10).
  """
  stopper = CountingLengthStopper(max_items=5)
  engine = CatEngine(item_bank, RecordingInitializer(0.0), max_info_selector, RecordingEstimator(0.0), stopper)
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)

  engine.run_session(session, FixedResponseProvider(True), context)

  assert session.administered_count == 5
  assert stopper.calls == 6


def test_simulated_response_provider_requires_true_theta(item_bank) -> None:
  """Simulated responses require a true theta on the session."""
  provider = SimulatedResponseProvider()
  context = RunContext(rng=np.random.default_rng(42))
  state = CatSessionState(session_id=0, estimates=[Estimate(0.0, math.inf)])

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
  state = CatSessionState(session_id=0, estimates=[Estimate(-1.5, math.inf)], true_theta=1.5)
  context = RunContext(rng=FakeRng(sample=draw))  # type: ignore[arg-type]

  assert SimulatedResponseProvider().answer(state, item_bank, item_id, context) is expected


def test_stepping_a_stopped_session_keeps_its_stop_reason(item_bank) -> None:
  """Stepping a session that has already stopped changes nothing about it.

  Fails if the engine decides again why a stopped session ended: a session whose selector ran out
  was relabelled as stopped by its selector on the next step.
  """
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), ExhaustedSelector(), RecordingEstimator(0.0), RecordingStopper(False)
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)
  engine.step(session, FixedResponseProvider(True), context)

  result = engine.step(session, FixedResponseProvider(True), context)

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
  engine = CatEngine(item_bank, fixed_initializer, random_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context, true_theta=0.0)

  with pytest.raises(
    ValueError,
    match=rf"Invalid item index {item_bank.n_items}\. Must be between 0 and {item_bank.n_items - 1}",
  ):
    engine.apply_response(session, item_bank.n_items, True, context)


def test_apply_response_rejects_duplicate_items(
  item_bank,
  fixed_initializer,
  random_selector,
  estimator,
  fixed_length_stopper,
) -> None:
  """A session cannot administer the same item twice."""
  engine = CatEngine(item_bank, fixed_initializer, random_selector, estimator, fixed_length_stopper)
  context = RunContext(rng=np.random.default_rng(42))
  session = CatSessionState(
    session_id=0, estimates=[Estimate(0.0, math.inf), Estimate(0.5, 1.0)], administered_item_ids=[0], responses=[True]
  )

  with pytest.raises(ValueError, match=r"Item 0 has already been administered in this session"):
    engine.apply_response(session, 0, False, context)


def test_step_ends_a_session_whose_initial_state_satisfies_the_stopper(item_bank) -> None:
  """A stopper satisfied before the first item ends the session without selecting one.

  Fails if the stopper is not asked about the initial state, so the session gets an item it should
  not, or if the terminal step result misreports the session.
  """
  selector = RecordingSelector(item_id=0)
  engine = CatEngine(item_bank, RecordingInitializer(0.4), selector, RecordingEstimator(0.0), RecordingStopper(True))
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)

  result = engine.step(session, FixedResponseProvider(True), context)

  assert selector.seen is None
  assert (result.stopped, result.selected_item_id, result.response) == (True, None, None)
  assert result.updated_theta == pytest.approx(0.4)
  assert result.stop_reason is StopReason.STOPPER
  assert result.session.stop_reason is StopReason.STOPPER
  assert session.stop_reason is StopReason.STOPPER


def test_select_next_ends_the_session_when_the_selector_has_no_item(item_bank) -> None:
  """A selector that raises ``NoItemsAvailableError`` ends the session with ``NO_ITEMS``.

  Fails if select_next lets the error escape, or returns None, while leaving the session in progress: a manual loop
  then had to copy step's handling, and notebook 02's did not.
  """
  engine = CatEngine(
    item_bank, RecordingInitializer(0.0), ExhaustedSelector(), RecordingEstimator(0.0), RecordingStopper(False)
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, context=context)

  assert engine.select_next(session, context) is None
  assert session.stop_reason is StopReason.NO_ITEMS


def test_custom_strategies_written_against_the_base_classes_run_a_session(item_bank) -> None:
  """Strategies that subclass the base classes and read only their documented inputs run a whole session.

  With every response correct, the estimates count them, 1, 2 and 3, after the initial 0.5, and the session stops at
  three items. Fails if the engine calls a strategy with anything but its documented signature, or gives it a context
  without the session's items and responses.
  """
  engine = CatEngine(item_bank, HalfInitializer(), LowestIndexSelector(), CorrectCountEstimator(), ThreeItemStopper())
  context = RunContext(rng=np.random.default_rng(0))
  session = engine.start_session(0, context)

  engine.run_session(session, FixedResponseProvider(True), context)

  assert session.administered_item_ids == [0, 1, 2]
  assert session.theta_history == (0.5, 1.0, 2.0, 3.0)
  assert session.stop_reason is StopReason.STOPPER


MIXED_PATTERN = [True, False, False, True, True, False]


def _run_scripted_session(item_bank: ItemBank, selector: BaseSelector, estimator: BaseEstimator) -> CatSessionState:
  """Return a session that answered MIXED_PATTERN to the items the selector chose, from an initial estimate of 0."""
  engine = CatEngine(item_bank, RecordingInitializer(0.0), selector, estimator, MaxLengthStopper(len(MIXED_PATTERN)))
  context = RunContext(rng=np.random.default_rng(0))
  session = engine.start_session(0, context)
  return engine.run_session(session, ScriptedResponseProvider(MIXED_PATTERN), context)


def test_a_session_records_the_estimate_the_estimator_returns_for_each_step(item_bank) -> None:
  """Each recorded Estimate, evaluation count included, is what the estimator returns for that step's context.

  The context of step k is rebuilt here from the session's first k items and responses and its estimate k - 1, and
  the estimator called on it directly. Fails if the engine records a count of its own, such as a running total over
  the session or the estimator's lifetime, or another theta or SE than the estimator's. The first answer alone takes
  Dodd's step, which evaluates nothing; every later step searches, so its count is not 0.
  """
  estimator = NumericalSearchEstimator()
  session = _run_scripted_session(item_bank, MaxInfoSelector(), estimator)
  items, responses, estimates = session.administered_item_ids, session.responses, session.estimates

  direct = [
    estimator.estimate(
      make_context(item_bank, items[:k], responses[:k], theta=estimates[k - 1].theta, se=estimates[k - 1].se)
    )
    for k in range(1, len(responses) + 1)
  ]

  assert estimates[0] == Estimate(0.0, math.inf, 0)
  assert estimates[1:] == direct
  assert all(estimate.evaluations for estimate in direct[1:])


def test_mei_look_ahead_adds_nothing_to_the_recorded_estimates(item_bank) -> None:
  """A session selected by MEISelector records what a session given the same items without look-ahead records.

  MEISelector estimates twice per candidate item with the engine's own estimator here. Fails if those look-ahead
  estimates add to the evaluation counts the session records, as they did to the estimator's shared counters, or
  change the recorded estimates.
  """
  estimator = NumericalSearchEstimator()
  mei = _run_scripted_session(item_bank, MEISelector(estimator=estimator), estimator)

  replay = _run_scripted_session(item_bank, LinearSelector(mei.administered_item_ids), estimator)

  assert replay.administered_item_ids == mei.administered_item_ids
  assert replay.estimates == mei.estimates
