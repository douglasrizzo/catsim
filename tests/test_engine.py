"""Tests for CatEngine: a test's configuration and the start, select and answer steps (ADR-0006, ADR-0010, ADR-0011)."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import override

import numpy as np
import pytest
from irtlib import ItemBank
from irtlib.typing import FloatArray

from catsim import (
  CatEngine,
  Estimate,
  ExcludedItemError,
  NoItemsAvailableError,
  Session,
  StepContext,
  StopReason,
)
from catsim.estimation import BaseEstimator, MLEstimator
from catsim.exposure import BaseExposureControl
from catsim.initialization import BaseInitializer
from catsim.selection import BaseSelector, LinearSelector, MaxInfoSelector, MEISelector, RandomSelector, remaining
from catsim.stopping import BaseStopper, MaxLengthStopper, StandardErrorStopper, StopRule
from tests.contexts import make_context

INITIAL = Estimate(0.0, math.inf, 0)


class ExhaustedSelector(BaseSelector):
  """Selector stub that behaves like an exhausted item bank."""

  @override
  def select(self, context: StepContext) -> int:
    """Raise NoItemsAvailableError, as a selector does on an exhausted bank."""
    del context
    msg = "No items remain"
    raise NoItemsAvailableError(msg)


@dataclass
class RecordingInitializer(BaseInitializer):
  """Initializer stub that records the arguments it receives."""

  theta: float
  seen_item_bank: ItemBank | None = None
  seen_rng: np.random.Generator | None = None

  @override
  def initialize(self, *, item_bank: ItemBank, rng: np.random.Generator) -> float:
    """Record the item bank and generator, and return the fixed theta."""
    self.seen_item_bank = item_bank
    self.seen_rng = rng
    return self.theta


@dataclass
class RecordingSelector(BaseSelector):
  """Selector stub that records the contexts it is given and proposes a fixed item."""

  item: int | np.integer | float | str | None
  seen: list[StepContext] = dataclasses.field(default_factory=list)

  @override
  def select(self, context: StepContext) -> int:
    """Record the context and return the fixed item."""
    self.seen.append(context)
    return self.item  # ty: ignore[invalid-explicit-return]


@dataclass
class RecordingEstimator(BaseEstimator):
  """Estimator stub that records the context it is given and returns a fixed estimate."""

  theta: float
  se: float = 0.5
  evaluations: int | np.integer | None = None
  seen: StepContext | None = None

  @override
  def estimate(self, context: StepContext) -> Estimate:
    """Record the context and return the fixed estimate."""
    self.seen = context
    return Estimate(theta=self.theta, se=self.se, evaluations=self.evaluations)  # ty: ignore[invalid-argument-type]


@dataclass
class ScriptedSEEstimator(BaseEstimator):
  """Estimator stub whose estimate after the n-th response has the n-th scripted standard error."""

  ses: list[float]

  @override
  def estimate(self, context: StepContext) -> Estimate:
    """Return theta 0 with the scripted standard error for the number of responses."""
    return Estimate(theta=0.0, se=self.ses[len(context.responses) - 1])


class FloatEstimator(BaseEstimator):
  """Estimator stub written for the contract where estimate returned a bare float."""

  @override
  def estimate(self, context: StepContext) -> float:  # ty: ignore[incompatible-method-override]
    """Return a bare float instead of an Estimate."""
    del context
    return 0.5


class DrawingEstimator(BaseEstimator):
  """Estimator stub whose estimate is a draw from its context's generator."""

  @override
  def estimate(self, context: StepContext) -> Estimate:
    """Return a uniform draw as the estimate."""
    return Estimate(theta=float(context.rng.uniform()), se=1.0)


class CountingLengthStopper(BaseStopper):
  """Stopper that ends the test at a fixed length and counts how often it is asked."""

  def __init__(self, max_items: int) -> None:
    self.max_items = max_items
    self.calls = 0

  @override
  def stop(self, context: StepContext) -> bool:
    self.calls += 1
    return len(context.administered_items) >= self.max_items


@dataclass
class RecordingStopper(BaseStopper):
  """Stopper stub that records the context it is given and returns a fixed decision."""

  decision: bool | np.bool_
  seen: StepContext | None = None

  @override
  def stop(self, context: StepContext) -> bool:
    """Record the context and return the fixed decision."""
    self.seen = context
    return self.decision  # ty: ignore[invalid-explicit-return]


class LowestIndexSelector(BaseSelector):
  """Custom selector: the remaining item with the lowest index."""

  @override
  def select(self, context: StepContext) -> int:
    return int(remaining(context)[0])


@dataclass
class DecliningControl(BaseExposureControl):
  """Exposure control that declines a fixed set of items, and records what it is asked about each proposal."""

  declines: frozenset[int]
  calls: list[tuple[int, tuple[int, ...], tuple[int, ...]]] = dataclasses.field(default_factory=list)

  @override
  def accept(self, context: StepContext, item: int) -> bool:
    self.calls.append((item, context.excluded_items, context.administered_items))
    # a selector proposing a declined item again would otherwise loop forever in a broken engine
    assert len(self.calls) <= 50, "the control was asked about too many proposals"
    return item not in self.declines


class CorrectCountEstimator(BaseEstimator):
  """Custom estimator: the number of correct responses."""

  @override
  def estimate(self, context: StepContext) -> Estimate:
    return Estimate(theta=float(sum(context.responses)), se=1.0)


class ThreeItemStopper(BaseStopper):
  """Custom stopper: three administered items."""

  @override
  def stop(self, context: StepContext) -> bool:
    return len(context.administered_items) >= 3


class HalfInitializer(BaseInitializer):
  """Custom initializer: every session starts at 0.5."""

  @override
  def initialize(self, *, item_bank: ItemBank, rng: np.random.Generator) -> float:
    del item_bank, rng
    return 0.5


def stub_engine(
  item_bank: ItemBank,
  *,
  selector: BaseSelector | None = None,
  estimator: BaseEstimator | None = None,
  stopper: BaseStopper | None = None,
  exposure_control: BaseExposureControl | None = None,
) -> CatEngine:
  """Return an engine of stubs: start at 0, propose item 0, estimate 0.5 and never stop, unless told otherwise."""
  return CatEngine(
    item_bank,
    RecordingInitializer(0.0),
    selector or RecordingSelector(0),
    estimator or RecordingEstimator(0.5),
    stopper or RecordingStopper(decision=False),
    exposure_control=exposure_control,
  )


def counting_engine(item_bank: ItemBank, exposure_control: BaseExposureControl | None = None) -> CatEngine:
  """Return an engine that starts at 0.5, gives the lowest remaining item, counts correct answers and stops at 3."""
  return CatEngine(
    item_bank,
    HalfInitializer(),
    LowestIndexSelector(),
    CorrectCountEstimator(),
    ThreeItemStopper(),
    exposure_control=exposure_control,
  )


def run(engine: CatEngine, respond: Callable[[int, Session], bool], *, rng: int = 0, start: Session | None = None):
  """Run a session to its end through select and answer, as ADR-0010's loop does, and return it."""
  generator = np.random.default_rng(rng)
  session = engine.start(rng=generator) if start is None else start
  while True:
    selection = engine.select(session, rng=generator)
    session = selection.session
    if selection.item is None:
      return session
    session = engine.answer(session, selection.item, respond(selection.item, session), rng=generator)


def session_after(items: tuple[int, ...], responses: tuple[bool, ...], *, excluded: tuple[int, ...] = ()) -> Session:
  """Return a running session that gave ``items`` (at least one) with ``responses``, at -0.25 with an SE of 0.6."""
  estimates = (INITIAL, *(Estimate(0.1 * k, 0.9) for k in range(1, len(items))), Estimate(-0.25, 0.6))
  return Session(administered_items=items, responses=responses, estimates=estimates, excluded_items=excluded)


def test_engine_rejects_item_parameters_that_are_not_an_item_bank(item_bank: ItemBank) -> None:
  """A bare parameter array was accepted and failed only at the first session, with an AttributeError on n_items."""
  with pytest.raises(TypeError, match=r"item_bank must be an ItemBank, got ndarray; wrap item parameters in ItemBank"):
    stub_engine(item_bank.items)  # ty: ignore[invalid-argument-type]


def test_start_gives_the_initializer_the_bank_and_generator_and_begins_at_its_estimate(item_bank: ItemBank) -> None:
  """start hands the initializer the bank and the generator it is given, and records its estimate as the initial one.

  Fails if the initializer gets another bank or a copy of the generator, or if the session starts with items, another
  estimate, a finite standard error (precision stoppers would judge it) or a stop reason.
  """
  initializer = RecordingInitializer(theta=1.25)
  engine = CatEngine(item_bank, initializer, RandomSelector(), MLEstimator(), MaxLengthStopper(5))
  rng = np.random.default_rng(123)

  session = engine.start(rng=rng)

  assert initializer.seen_item_bank is item_bank
  assert initializer.seen_rng is rng
  assert session == Session(estimates=(Estimate(1.25, math.inf, 0),))


def test_start_keeps_the_excluded_items_it_is_given_in_order(item_bank: ItemBank) -> None:
  """Items withheld from the start, such as those an examinee saw before, are the session's first exclusions.

  Fails if start drops them (the selector would offer them), sorts them, or takes NumPy integers as given.
  """
  session = stub_engine(item_bank).start(rng=0, excluded_items=[np.int64(9), 2])  # ty: ignore[invalid-argument-type]

  assert session.excluded_items == (9, 2)
  assert [type(item) for item in session.excluded_items] == [int, int]


@pytest.mark.parametrize(
  ("excluded", "error", "message"),
  [
    pytest.param(
      [50], ValueError, "excluded_items[0] is 50; item indices must be between 0 and 49", id="past-the-bank"
    ),
    pytest.param([3, 3], ValueError, "excluded_items[1] is 3 ('item_03'); item indices must not repeat", id="repeated"),
    pytest.param([1, 3.7], TypeError, "excluded_items[1] is 3.7; item indices must be integers", id="fraction"),
  ],
)
def test_start_rejects_excluded_items_that_are_not_distinct_items_of_the_bank(
  item_bank: ItemBank, excluded: list[object], error: type[Exception], message: str
) -> None:
  """An exclusion outside the bank, repeated or fractional fails at start, before the initializer draws.

  Fails if start accepts it: NumPy would read 3.7 as item 3, and an item past the bank would fail later in a selector.
  A repeated item is named by its index and its name (ADR-0004).
  """
  initializer = RecordingInitializer(0.0)
  engine = CatEngine(item_bank, initializer, RecordingSelector(0), RecordingEstimator(0.5), RecordingStopper(False))

  with pytest.raises(error, match=re.escape(message)):
    engine.start(rng=0, excluded_items=excluded)  # ty: ignore[invalid-argument-type]
  assert initializer.seen_rng is None


def test_start_ends_a_session_whose_initial_state_satisfies_the_stopper(item_bank: ItemBank) -> None:
  """A stopper satisfied before the first item ends the session in start, and select then proposes nothing.

  Fails if the stopper is not asked about the initial state (the session would get an item it should not), if it is
  asked about another state, or if select still asks the selector.
  """
  selector, stopper = RecordingSelector(0), RecordingStopper(decision=True)
  engine = stub_engine(item_bank, selector=selector, stopper=stopper)
  rng = np.random.default_rng(4)

  session = engine.start(rng=rng, excluded_items=(7,))
  selection = engine.select(session, rng=rng)

  assert session.stop_reason is StopReason.STOPPER
  assert stopper.seen is not None
  assert (stopper.seen.administered_items, stopper.seen.excluded_items, stopper.seen.theta) == ((), (7,), 0.0)
  assert stopper.seen.rng is rng
  assert (selection.item, selection.declined) == (None, ())
  assert selection.session is session
  assert selector.seen == []


def test_select_gives_the_selector_the_session_the_generator_and_the_rates(item_bank: ItemBank) -> None:
  """select hands the selector the bank, the session's items, responses, exclusions and current estimate with its SE,
  the generator it is given and the exposure rates, and engine.context builds that same context.

  Fails if any input is dropped or taken from elsewhere, such as the initial estimate instead of the current one, if
  the generator is copied, or if engine.context differs from what the selector is given.
  """
  selector = RecordingSelector(item=4)
  engine = stub_engine(item_bank, selector=selector)
  session = session_after((1, 2), (True, False), excluded=(5, 0))
  rates = np.zeros(item_bank.n_items)
  rates[[0, 3]] = [1.0, 0.5]
  rng = np.random.default_rng(99)

  selection = engine.select(session, rng=rng, exposure_rates=rates)

  (seen,) = selector.seen
  assert (selection.item, selection.session, selection.declined) == (4, session, ())
  assert seen.item_bank is item_bank
  assert seen.rng is rng
  assert (seen.administered_items, seen.responses, seen.excluded_items) == ((1, 2), (True, False), (5, 0))
  assert (seen.theta, seen.se) == (-0.25, 0.6)
  assert seen.exposure_rates is not None
  assert seen.exposure_rates.tolist() == rates.tolist()
  context = engine.context(session, rng=rng, exposure_rates=rates)
  assert _fields(context) == _fields(seen)


def _fields(context: StepContext) -> tuple[object, ...]:
  """Return a context's fields, with its exposure rates as a list, for comparison."""
  rates = None if context.exposure_rates is None else context.exposure_rates.tolist()
  return (
    context.item_bank,
    context.rng,
    context.administered_items,
    context.responses,
    context.excluded_items,
    context.theta,
    context.se,
    rates,
  )


def test_select_without_rates_gives_the_selector_none(item_bank: ItemBank) -> None:
  """Without exposure rates the selector gets None, under which no cap applies (ADR-0012).

  Fails if the engine makes up rates, such as all zeros, for a caller that keeps no exposure counts.
  """
  selector = RecordingSelector(item=2)

  stub_engine(item_bank, selector=selector).select(Session(estimates=(INITIAL,)), rng=0)

  assert selector.seen[0].exposure_rates is None


def test_select_with_the_same_seed_proposes_the_same_item(item_bank: ItemBank) -> None:
  """An int rng seeds a new generator for the call, so a random selector proposes the same item for the same seed.

  Fails if select ignores the seed, or draws from a generator kept between calls.
  """
  engine = stub_engine(item_bank, selector=RandomSelector())
  session = Session(estimates=(INITIAL,))

  proposals = {seed: [engine.select(session, rng=seed).item for _ in range(3)] for seed in (1, 2)}

  assert all(len(set(items)) == 1 for items in proposals.values())
  assert proposals[1] != proposals[2]


@pytest.mark.parametrize(
  ("rates", "message"),
  [
    pytest.param(
      np.array([4] + [0] * 49), "exposure_rates[0] is 4.0; exposure rates must be between 0 and 1", id="counts"
    ),
    pytest.param(
      np.full(30, 0.1),
      "exposure_rates has shape (30,); there must be one exposure rate per item of the bank (50)",
      id="wrong-length",
    ),
  ],
)
def test_select_rejects_exposure_rates_that_are_not_rates_of_the_bank(item_bank: ItemBank, rates, message: str) -> None:
  """Counts passed as rates turned the cap off, and rates of another bank failed deep in NumPy (D29).

  Fails if select accepts them and the selector caps against them.
  """
  selector = RecordingSelector(item=0)

  with pytest.raises(ValueError, match=re.escape(message)):
    stub_engine(item_bank, selector=selector).select(Session(estimates=(INITIAL,)), rng=0, exposure_rates=rates)
  assert selector.seen == []


@pytest.mark.parametrize("step", ["select", "answer", "context"])
def test_the_steps_reject_a_session_with_items_outside_the_bank(item_bank: ItemBank, step: str) -> None:
  """A session built without the bank, such as one stored for another bank, is checked against the engine's bank.

  Fails if an index past the bank reaches a strategy, where NumPy raises a bare IndexError, or wraps around.
  """
  engine = stub_engine(item_bank)
  session = session_after((3, 50), (True, False))
  calls = {
    "select": lambda: engine.select(session, rng=0),
    "answer": lambda: engine.answer(session, 4, True, rng=0),
    "context": lambda: engine.context(session),
  }

  with pytest.raises(ValueError, match=re.escape("administered_items[1] is 50; item indices must be between 0 and 49")):
    calls[step]()


@pytest.mark.parametrize("step", ["select", "answer", "context"])
def test_the_steps_reject_what_is_not_a_session(item_bank: ItemBank, step: str) -> None:
  """A session's stored fields passed as a dict would fail inside the engine with an AttributeError on its items."""
  engine = stub_engine(item_bank)
  stored = {"administered_items": (), "responses": (), "estimates": (INITIAL,)}
  calls = {
    "select": lambda: engine.select(stored, rng=0),  # ty: ignore[invalid-argument-type]
    "answer": lambda: engine.answer(stored, 4, True, rng=0),  # ty: ignore[invalid-argument-type]
    "context": lambda: engine.context(stored),  # ty: ignore[invalid-argument-type]
  }

  with pytest.raises(TypeError, match="session is a dict; it must be a Session"):
    calls[step]()


def test_context_without_a_generator_draws_fresh_entropy_and_has_no_rates(item_bank: ItemBank) -> None:
  """The context of a session asked about on its own has the session's fields, no rates and an unseeded generator.

  Fails if the engine makes up rates, such as all zeros, or seeds the generator, so that every call draws the same
  numbers.
  """
  engine = stub_engine(item_bank)
  session = session_after((6,), (True,))

  context = engine.context(session)

  assert _fields(context)[2:] == ((6,), (True,), (), -0.25, 0.6, None)
  assert engine.context(session).rng.integers(2**62) != engine.context(session).rng.integers(2**62)


@pytest.mark.parametrize(
  ("max_items", "ses", "ended_by"),
  [(3, [0.9, 0.7, 0.5], "cap"), (5, [0.9, 0.6, 0.35], "precision")],
)
def test_a_stop_rule_names_the_criterion_that_ended_a_finished_session(
  item_bank: ItemBank, max_items: int, ses: list[float], ended_by: str
) -> None:
  """Both sessions end after 3 items against a target SE of 0.4.

  With a cap of 3 and SEs 0.9, 0.7, 0.5, the cap holds and the SE target does not; with a cap of 5 and SEs 0.9, 0.6,
  0.35, the SE target holds after the third item and the cap does not. Fails if the context is not the session as it
  ended: no SE target holds at the initial infinite SE, and no cap with the items before the last response.
  """
  cap, precision = MaxLengthStopper(max_items), StandardErrorStopper(0.4)
  rule = StopRule(cap, precision)
  engine = CatEngine(item_bank, HalfInitializer(), LowestIndexSelector(), ScriptedSEEstimator(ses), rule)

  session = run(engine, lambda item, session: True)

  assert len(session.administered_items) == 3
  assert rule.satisfied(engine.context(session)) == (cap if ended_by == "cap" else precision,)


def test_answer_estimates_from_the_current_estimate_and_returns_the_next_session(item_bank: ItemBank) -> None:
  """answer gives the estimator the session with the new item and response, starting from the current estimate and
  SE, with the generator it is given, and returns a new session with the item, the response and the estimator's
  estimate added.

  The current estimate is -0.25 with an SE of 0.6, unlike StepContext's defaults of 0 and infinity. Fails if the
  estimator starts from the defaults or the initial estimate, gets another bank, items, responses or generator, if the
  new session records another item, response or estimate, or if the given session changes.
  """
  estimator = RecordingEstimator(theta=1.1, se=0.42, evaluations=7)
  engine = stub_engine(item_bank, estimator=estimator)
  session = session_after((1,), (True,), excluded=(8,))
  rng = np.random.default_rng(3)

  answered = engine.answer(session, 3, False, rng=rng)

  assert estimator.seen is not None
  assert estimator.seen.item_bank is item_bank
  assert estimator.seen.rng is rng
  assert (estimator.seen.administered_items, estimator.seen.responses) == ((1, 3), (True, False))
  assert (estimator.seen.theta, estimator.seen.se) == (-0.25, 0.6)
  assert answered == Session(
    administered_items=(1, 3),
    responses=(True, False),
    estimates=(*session.estimates, Estimate(1.1, 0.42, 7)),
    excluded_items=(8,),
  )
  assert session == session_after((1,), (True,), excluded=(8,))


def test_answer_asks_the_stopper_about_the_new_state(item_bank: ItemBank) -> None:
  """A stopper satisfied by the state after the response ends the session that answer returns.

  Fails if the stopper judges a stale state (without the new item and response, or with the previous estimate or its
  SE) or gets another generator than answer's, or if its decision is dropped, so the session runs past its criterion.
  """
  stopper = RecordingStopper(decision=True)
  engine = stub_engine(item_bank, estimator=RecordingEstimator(theta=0.3, se=0.42), stopper=stopper)
  rng = np.random.default_rng(3)

  answered = engine.answer(Session(estimates=(INITIAL,)), 0, False, rng=rng)

  assert stopper.seen is not None
  assert stopper.seen.rng is rng
  assert (stopper.seen.administered_items, stopper.seen.responses) == ((0,), (False,))
  assert (stopper.seen.theta, stopper.seen.se) == (0.3, 0.42)
  assert answered.stop_reason is StopReason.STOPPER


def test_answer_with_the_same_seed_gives_the_same_session(item_bank: ItemBank) -> None:
  """An int rng seeds a new generator for the call, so an estimator that draws records the same estimate for a seed.

  Fails if answer ignores its rng: with fresh entropy the same seed gives other estimates, and with a fixed generator
  of its own two seeds give the same one.
  """
  engine = stub_engine(item_bank, estimator=DrawingEstimator())
  session = Session(estimates=(INITIAL,))

  estimates = {seed: [engine.answer(session, 2, True, rng=seed).current_theta for _ in range(2)] for seed in (5, 6)}

  assert all(len(set(thetas)) == 1 for thetas in estimates.values())
  assert estimates[5] != estimates[6]


def test_answer_rejects_an_estimator_that_returns_a_bare_float(item_bank: ItemBank) -> None:
  """An estimator written for the contract that returned a float fails with a TypeError saying so.

  Fails if the engine reads ``.theta`` off the float and raises an AttributeError that names no estimator.
  """
  engine = stub_engine(item_bank, estimator=FloatEstimator())

  with pytest.raises(TypeError, match=re.escape("FloatEstimator.estimate must return an Estimate, got float")):
    engine.answer(Session(estimates=(INITIAL,)), 3, True, rng=0)


@pytest.mark.parametrize(("response", "stored"), [(1, True), (0, False), (np.True_, True), (np.int64(0), False)])
def test_answer_stores_a_response_of_1_or_0_as_a_bool(item_bank: ItemBank, response: object, stored: bool) -> None:
  """An application that stores answers as 0 and 1 passes them as they are, and the session holds bools (ADR-0009).

  Fails if 1 and 0 are rejected, or stored as given: the session's responses would mix ints and bools.
  """
  answered = stub_engine(item_bank).answer(Session(estimates=(INITIAL,)), 3, response, rng=0)  # ty: ignore[invalid-argument-type]

  assert answered.responses == (stored,)
  assert type(answered.responses[0]) is bool


@pytest.mark.parametrize(
  ("response", "error", "message"),
  [
    pytest.param(1.0, TypeError, "response is 1.0; responses must be bools or the integers 0 and 1", id="float"),
    pytest.param("yes", TypeError, "response is 'yes'; responses must be bools or the integers 0 and 1", id="str"),
    pytest.param(None, TypeError, "response is None; responses must be bools or the integers 0 and 1", id="None"),
    pytest.param(2, ValueError, "response is 2; an integer response must be 0 or 1", id="two"),
  ],
)
def test_answer_rejects_a_response_that_is_not_a_boolean_or_0_or_1(
  item_bank: ItemBank, response: object, error: type[Exception], message: str
) -> None:
  """bool(1.0), bool("yes") and bool(2) are all True, so a converted response would score an answer never given.

  Fails too if the bad response reaches the estimator.
  """
  estimator = RecordingEstimator(0.5)

  with pytest.raises(error, match=re.escape(message)):
    stub_engine(item_bank, estimator=estimator).answer(Session(estimates=(INITIAL,)), 3, response, rng=0)  # ty: ignore[invalid-argument-type]
  assert estimator.seen is None


def test_answer_rejects_a_stopped_session(item_bank: ItemBank) -> None:
  """A session that has stopped accepts no more responses (ADR-0010).

  Fails if answer appends an item to a session its stopper has already ended.
  """
  stopped = Session(estimates=(INITIAL,), stop_reason=StopReason.NO_ITEMS)

  with pytest.raises(
    ValueError, match=re.escape("the session has stopped (stop_reason 'no_items') and accepts no more responses")
  ):
    stub_engine(item_bank).answer(stopped, 1, True, rng=0)


@pytest.mark.parametrize(
  ("item", "error", "message"),
  [
    pytest.param(2.0, TypeError, "item is 2.0; item indices must be integers", id="float"),
    pytest.param("3", TypeError, "item is '3'; item indices must be integers", id="str"),
    pytest.param(True, TypeError, "item is True; item indices must be integers, not bools", id="bool"),
    pytest.param(50, ValueError, "item is 50; item indices must be between 0 and 49", id="past-the-bank"),
    pytest.param(-1, ValueError, "item is -1; item indices must be between 0 and 49", id="negative"),
    pytest.param(
      1, ValueError, "item is 1 ('item_01'); the session has already been given that item", id="administered"
    ),
    pytest.param(
      8,
      ExcludedItemError,
      "item is 8 ('item_08'); the session excluded that item, which it must not be given",
      id="excluded",
    ),
  ],
)
def test_answer_rejects_an_item_select_could_not_have_proposed(
  item_bank: ItemBank, item: object, error: type[Exception], message: str
) -> None:
  """An item that is not an index, outside the bank, already given or excluded fails before the estimator runs.

  Fails if 2.0 is stored as a float, a bank-sized index reads past the item parameters, -1 reads the last item, a
  response counts twice in the estimate, or an item the exposure control declined is administered (ExcludedItemError,
  so an application can tell it from a bad index). An item of the bank is named by its index and its name (ADR-0004).
  """
  estimator = RecordingEstimator(0.5)

  with pytest.raises(error, match=re.escape(message)):
    stub_engine(item_bank, estimator=estimator).answer(session_after((1,), (True,), excluded=(8,)), item, True, rng=0)  # ty: ignore[invalid-argument-type]
  assert estimator.seen is None


def test_a_session_holds_builtin_python_values(item_bank: ItemBank) -> None:
  """The engine stores Python scalars whatever numeric types the strategies and the caller give it.

  Fails if a NumPy initial theta, an estimate whose theta, standard error or evaluation count is a NumPy scalar, a
  ``numpy.int64`` item index (from the selector, or passed to answer) or a ``numpy.bool_`` response or stop decision
  is stored as given.
  """
  engine = CatEngine(
    item_bank,
    RecordingInitializer(theta=np.float64(0.0)),
    RecordingSelector(item=np.int64(3)),
    RecordingEstimator(theta=np.float64(0.25), se=np.float64(0.4), evaluations=np.int64(7)),
    RecordingStopper(np.False_),
  )
  session = engine.start(rng=0)
  selection = engine.select(session, rng=0)
  assert selection.item is not None

  session = engine.answer(selection.session, selection.item, np.True_, rng=0)  # ty: ignore[invalid-argument-type]
  session = engine.answer(session, np.int64(5), np.False_, rng=0)  # ty: ignore[invalid-argument-type]

  assert type(selection.item) is int
  assert session.theta_history == (0.0, 0.25, 0.25)
  assert session.estimates[1:] == (Estimate(0.25, 0.4, 7),) * 2
  scalars = [session.current_theta, *session.administered_items, *session.responses, session.estimates[1].evaluations]
  assert [type(value) for value in scalars] == [float, int, int, bool, bool, int]
  assert session.stop_reason is None


def test_the_stopper_is_asked_once_per_session_state(item_bank: ItemBank, max_info_selector: MaxInfoSelector) -> None:
  """A session of n items asks its stopper n + 1 times: about the initial state and after each response (ADR-0010).

  Fails if select also asks the stopper, about the state answer has just judged, which doubled the calls (a 5-item
  session made 10).
  """
  stopper = CountingLengthStopper(max_items=5)
  engine = CatEngine(item_bank, RecordingInitializer(0.0), max_info_selector, RecordingEstimator(0.0), stopper)

  session = run(engine, lambda item, session: True)

  assert len(session.administered_items) == 5
  assert stopper.calls == 6


def test_selecting_for_a_stopped_session_returns_it_unchanged(item_bank: ItemBank) -> None:
  """select on a session that has ended proposes nothing and returns the same session, with its stop reason.

  Fails if the engine decides again why a stopped session ended: a session whose selector ran out was relabelled as
  stopped by its stopper on the next step, or asked the selector again.
  """
  selector = RecordingSelector(0)
  ended = Session(estimates=(INITIAL,), stop_reason=StopReason.NO_ITEMS)

  selection = stub_engine(item_bank, selector=selector, stopper=RecordingStopper(True)).select(ended, rng=0)

  assert (selection.item, selection.declined) == (None, ())
  assert selection.session is ended
  assert selector.seen == []


def test_select_ends_the_session_when_the_selector_has_no_item(item_bank: ItemBank) -> None:
  """A selector that raises NoItemsAvailableError ends the session with NO_ITEMS, and select proposes nothing.

  Fails if select lets the error escape, or returns no item while leaving the session running: every loop then had
  to copy the handling, and notebook 02's did not.
  """
  session = session_after((2,), (True,))

  selection = stub_engine(item_bank, selector=ExhaustedSelector()).select(session, rng=0)

  assert (selection.item, selection.declined) == (None, ())
  assert selection.session == dataclasses.replace(session, stop_reason=StopReason.NO_ITEMS)


@pytest.mark.parametrize(
  ("proposal", "error", "message"),
  [
    pytest.param(-1, ValueError, "the item RecordingSelector proposed is -1; item indices must be between", id="neg"),
    pytest.param(50, ValueError, "the item RecordingSelector proposed is 50; item indices must be between", id="past"),
    pytest.param(
      1,
      ValueError,
      "the item RecordingSelector proposed is 1 ('item_01'); the session has already been given that item",
      id="had",
    ),
    pytest.param(
      7,
      ExcludedItemError,
      "the item RecordingSelector proposed is 7 ('item_07'); the session excluded that item",
      id="excl",
    ),
  ],
)
def test_select_checks_the_proposal_before_the_control_sees_it(
  item_bank: ItemBank, proposal: int, error: type[Exception], message: str
) -> None:
  """An invalid item from the selector raises in select, naming the selector, before the exposure control is asked.

  Fails if the engine hands it on: a negative index would show the bank's last item, an item given again would count
  twice, and an excluded item would be shown to an examinee it was withheld from. An item of the bank is named by its
  index and its name (ADR-0004).
  """
  control = DecliningControl(declines=frozenset())
  engine = stub_engine(item_bank, selector=RecordingSelector(proposal), exposure_control=control)

  with pytest.raises(error, match=re.escape(message)):
    engine.select(session_after((1,), (True,), excluded=(7,)), rng=0)
  assert control.calls == []


def test_a_declined_item_is_excluded_for_the_rest_of_the_test(item_bank: ItemBank) -> None:
  """An item the exposure control declines joins the session's excluded items and is never proposed again.

  The lowest-index selector proposes item 0 first at every step; the control declines it once, and the session gives
  items 1, 2 and 3. Fails if the decline lasts only for the selection it happened in: the selector would propose item
  0 at every step, the control would be asked about it three times, and the session would keep no record of it.
  """
  control = DecliningControl(declines=frozenset({0}))

  session = run(counting_engine(item_bank, control), lambda item, session: True)

  assert session.administered_items == (1, 2, 3)
  assert session.excluded_items == (0,)
  assert [item for item, _, _ in control.calls] == [0, 1, 2, 3]


def test_the_control_sees_each_proposal_with_the_declines_so_far_at_the_same_position(item_bank: ItemBank) -> None:
  """Each proposal reaches the control in a context that adds the selection's declines to the session's exclusions.

  The session gave item 5 and had excluded item 7; the control declines items 0 and 1, so the selector proposes 0, 1
  and 2. Fails if a later proposal's context drops the earlier declines (the selector would propose 0 again), replaces
  the session's own exclusions, or moves the test position, or if select does not return the declines.
  """
  control = DecliningControl(declines=frozenset({0, 1}))
  session = session_after((5,), (True,), excluded=(7,))

  selection = counting_engine(item_bank, control).select(session, rng=0)

  assert control.calls == [(0, (7,), (5,)), (1, (7, 0), (5,)), (2, (7, 0, 1), (5,))]
  assert (selection.item, selection.declined) == (2, (0, 1))
  assert selection.session == dataclasses.replace(session, excluded_items=(7, 0, 1))


def test_a_session_whose_remaining_items_are_all_declined_ends_with_no_items() -> None:
  """When the control declines every item left, the session ends with NO_ITEMS and keeps the declines.

  On a 4-item bank where item 2 was given, the control declines 0, 1 and 3 in turn. Fails if the engine loops
  forever, lets NoItemsAvailableError escape, ends the session without recording or returning the items it declined,
  or reports it stopped by the stopper.
  """
  bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0]] * 4))
  engine = counting_engine(bank, DecliningControl(declines=frozenset(range(4))))

  selection = engine.select(session_after((2,), (True,)), rng=0)

  assert (selection.item, selection.declined) == (None, (0, 1, 3))
  assert selection.session.stop_reason is StopReason.NO_ITEMS
  assert selection.session.excluded_items == (0, 1, 3)


class ExhaustedControl(BaseExposureControl):
  """Exposure control that declines item 0 and then has nothing left to accept."""

  @override
  def accept(self, context: StepContext, item: int) -> bool:
    if item == 0:
      return False
    msg = "every remaining item is over-exposed"
    raise NoItemsAvailableError(msg)


def test_a_control_with_no_item_to_accept_ends_the_session_with_no_items(item_bank: ItemBank) -> None:
  """An exposure control that raises NoItemsAvailableError ends the session with NO_ITEMS, keeping its declines.

  Fails if the error escapes select: a simulation run would then abort at that examinee, as it did not before.
  """
  selection = counting_engine(item_bank, ExhaustedControl()).select(Session(estimates=(INITIAL,)), rng=0)

  assert (selection.item, selection.declined) == (None, (0,))
  assert (selection.session.excluded_items, selection.session.stop_reason) == ((0,), StopReason.NO_ITEMS)


def test_a_selector_proposing_a_declined_item_again_raises_instead_of_looping(item_bank: ItemBank) -> None:
  """A selector that ignores excluded_items and proposes the item just declined raises ExcludedItemError.

  The selector always proposes item 3, which the control declines. Fails if the engine asks the control about item 3
  again (forever, but for the stub's limit), or hands a declined item to the examinee.
  """
  control = DecliningControl(declines=frozenset({3}))
  engine = stub_engine(item_bank, selector=RecordingSelector(3), exposure_control=control)

  with pytest.raises(ExcludedItemError, match=re.escape("is 3 ('item_03'); the session excluded that item")):
    engine.select(Session(estimates=(INITIAL,)), rng=0)

  assert control.calls == [(3, (), ())]


def test_a_session_rebuilt_from_its_stored_fields_continues_like_the_original(item_bank: ItemBank) -> None:
  """A session rebuilt from stored fields, exclusions included, takes the same next steps as the one it was stored from.

  The session had item 1 and excluded items 0 and 2; the lowest-index selector then gives 3 and 4. Fails if the
  rebuilt session drops the exclusions, or the engine does not pass them to the selector: the session would get
  items 0 and 2.
  """
  engine = counting_engine(item_bank)
  original = engine.answer(engine.start(rng=0, excluded_items=(0, 2)), 1, True, rng=0)
  stored = {field.name: list(getattr(original, field.name)) for field in dataclasses.fields(Session)[:4]}

  rebuilt = Session(**stored)  # ty: ignore[invalid-argument-type]
  finished = run(engine, lambda item, session: True, start=rebuilt)

  assert rebuilt == original
  assert finished.administered_items == (1, 3, 4)
  assert finished.excluded_items == (0, 2)


def test_custom_strategies_written_against_the_base_classes_run_a_session(item_bank: ItemBank) -> None:
  """Strategies that subclass the base classes and read only their documented inputs run a whole session.

  With every response correct, the estimates count them, 1, 2 and 3, after the initial 0.5, and the session stops at
  three items. Fails if the engine calls a strategy with anything but its documented signature, or gives it a context
  without the session's items and responses.
  """
  session = run(counting_engine(item_bank), lambda item, session: True)

  assert session.administered_items == (0, 1, 2)
  assert session.theta_history == (0.5, 1.0, 2.0, 3.0)
  assert session.stop_reason is StopReason.STOPPER


MIXED_PATTERN = (True, False, False, True, True, False)


def _scripted_session(item_bank: ItemBank, selector: BaseSelector, estimator: BaseEstimator) -> Session:
  """Return a session that answered MIXED_PATTERN to the items the selector chose, from an initial estimate of 0."""
  engine = CatEngine(item_bank, RecordingInitializer(0.0), selector, estimator, MaxLengthStopper(len(MIXED_PATTERN)))
  return run(engine, lambda item, session: MIXED_PATTERN[len(session.responses)])


def test_a_session_records_the_estimate_the_estimator_returns_for_each_step(item_bank: ItemBank) -> None:
  """Each recorded Estimate, evaluation count included, is what the estimator returns for that step's context.

  The context of step k is rebuilt here from the session's first k items and responses and its estimate k - 1, and
  the estimator called on it directly. Fails if the engine records a count of its own, such as a running total over
  the session or the estimator's lifetime, or another theta or SE than the estimator's. The first answer alone takes
  Dodd's step, which evaluates nothing; every later step searches, so its count is not 0.
  """
  estimator = MLEstimator()
  session = _scripted_session(item_bank, MaxInfoSelector(), estimator)
  items, responses, estimates = session.administered_items, session.responses, session.estimates

  direct = [
    estimator.estimate(
      make_context(item_bank, items[:k], responses[:k], theta=estimates[k - 1].theta, se=estimates[k - 1].se)
    )
    for k in range(1, len(responses) + 1)
  ]

  assert estimates[0] == INITIAL
  assert list(estimates[1:]) == direct
  assert all(estimate.evaluations for estimate in direct[1:])


def test_mei_look_ahead_adds_nothing_to_the_recorded_estimates(item_bank: ItemBank) -> None:
  """A session selected by MEISelector records what a session given the same items without look-ahead records.

  MEISelector estimates twice per candidate item with the engine's own estimator here. Fails if those look-ahead
  estimates add to the evaluation counts the session records, as they did to the estimator's shared counters, or
  change the recorded estimates.
  """
  estimator = MLEstimator()
  mei = _scripted_session(item_bank, MEISelector(estimator=estimator), estimator)

  replay = _scripted_session(item_bank, LinearSelector(list(mei.administered_items)), estimator)

  assert replay.administered_items == mei.administered_items
  assert replay.estimates == mei.estimates


def test_the_rates_select_is_given_are_the_rates_the_selector_caps_against() -> None:
  """With r_max 0.5 and rates of 0.5 for items 0 and 1, MaxInfoSelector passes over them for item 2 at theta 0.

  At theta 0 the items rank 0, 1, 2, 3 by information. Fails if select drops the rates, so the selector takes item 0,
  or hands it rates other than the ones given.
  """
  bank = ItemBank(np.array([[a, 0.0, 0.0, 1.0] for a in (2.0, 1.5, 1.0, 0.5)]))
  engine = CatEngine(
    bank, RecordingInitializer(0.0), MaxInfoSelector(r_max=0.5), RecordingEstimator(0.0), MaxLengthStopper(4)
  )
  rates: FloatArray = np.array([0.5, 0.5, 1 / 3, 0.0])
  session = engine.start(rng=0)

  assert engine.select(session, rng=0).item == 0
  assert engine.select(session, rng=0, exposure_rates=rates).item == 2
