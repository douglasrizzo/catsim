"""Tests for simulate, its generated and recorded examinees and their random streams, and its results."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Callable
from typing import override

import numpy as np
import pytest

from catsim import CatEngine, Estimate, ItemBank, Session, SimulationResult, StepContext, StopReason, simulate
from catsim.estimation import EAPEstimator, NumericalSearchEstimator
from catsim.exposure import BaseExposureControl
from catsim.initialization import BaseInitializer, FixedPointInitializer, UniformInitializer
from catsim.selection import BaseSelector, MaxInfoSelector, ProgressiveSelector, RandomesqueSelector, RandomSelector
from catsim.stopping import BaseStopper, MaxLengthStopper, StandardErrorStopper, StopRule
from catsim.typing import FloatArray


class FirstRemainingSelector(BaseSelector):
  """Select the lowest-index remaining item and record the exposure rates shown at each call."""

  def __init__(self) -> None:
    super().__init__()
    self.seen_exposure_rates: list[FloatArray | None] = []

  @override
  def select(self, context: StepContext) -> int:
    rates = context.exposure_rates
    self.seen_exposure_rates.append(None if rates is None else rates.copy())
    return next(i for i in range(context.item_bank.n_items) if i not in context.administered_items)


class AcceptingControl(BaseExposureControl):
  """Exposure control that accepts every proposal without drawing a random number."""

  @override
  def accept(self, context: StepContext, item: int) -> bool:
    del context, item
    return True


class CoinControl(BaseExposureControl):
  """Exposure control that accepts each proposal with probability one half, drawn from the session's generator."""

  @override
  def accept(self, context: StepContext, item: int) -> bool:
    del item
    return bool(context.rng.random() < 0.5)


def make_engine(
  item_bank: ItemBank,
  selector: BaseSelector,
  stopper: BaseStopper | int,
  exposure_control: BaseExposureControl | None = None,
  initializer: BaseInitializer | None = None,
) -> CatEngine:
  """Return an engine with the given selector and control, starting at 0 unless told otherwise.

  An int stopper stops after that many items.
  """
  return CatEngine(
    item_bank,
    initializer or FixedPointInitializer(0.0),
    selector,
    NumericalSearchEstimator(),
    MaxLengthStopper(stopper) if isinstance(stopper, int) else stopper,
    exposure_control=exposure_control,
  )


@pytest.fixture
def random_engine(item_bank: ItemBank) -> CatEngine:
  """Provide an engine that gives every examinee 10 random items."""
  return make_engine(item_bank, RandomSelector(), 10)


@pytest.fixture
def simulation_result(random_engine: CatEngine) -> SimulationResult:
  """Provide a completed simulation result for five generated examinees."""
  return simulate(random_engine, true_thetas=[-1.0, -0.5, 0.0, 0.5, 1.0], rng=42)


def probabilities(item_bank: ItemBank, theta: float) -> FloatArray:
  """Return each item's 4PL probability of a correct answer at ``theta``, computed here rather than by catsim."""
  a, b, c, d = item_bank.items.T
  return c + (d - c) / (1.0 + np.exp(-a * (theta - b)))


def stream(seed: int, examinee: int, kind: int) -> np.random.Generator:
  """Return the generator ADR-0013 gives examinee ``examinee``: kind 0 for its pattern, 1 for its session."""
  return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=(examinee, kind)))


def administered(result: SimulationResult) -> list[tuple[int, ...]]:
  """Return each session's administered items, in session order."""
  return [session.administered_items for session in result.sessions]


def answers(result: SimulationResult) -> list[tuple[bool, ...]]:
  """Return each session's responses, in session order."""
  return [session.responses for session in result.sessions]


def shared_answers(first: SimulationResult, second: SimulationResult) -> list[tuple[bool, bool]]:
  """Return, for every examinee and every item both runs gave them, the two runs' answers to it."""
  pairs = []
  for one, other in zip(first.sessions, second.sessions, strict=True):
    answers = dict(zip(other.administered_items, other.responses, strict=True))
    pairs += [
      (response, answers[item])
      for item, response in zip(one.administered_items, one.responses, strict=True)
      if item in answers
    ]
  return pairs


class TestGeneratedExaminees:
  """Tests for runs of examinees given by their true abilities."""

  def test_the_result_keeps_the_true_abilities_in_examinee_order(self, random_engine: CatEngine) -> None:
    """Session i simulates true_thetas[i]; fails if the result reorders, rounds or drops the true abilities."""
    thetas = [0.25, -1.5, 1.0]
    result = simulate(random_engine, true_thetas=thetas)

    assert result.true_thetas is not None
    assert result.true_thetas.tolist() == thetas
    assert len(result.sessions) == 3

  def test_every_session_runs_to_its_stopper(self, simulation_result: SimulationResult) -> None:
    """Every session of a run is run to its stopper: 10 items, 10 responses and 11 estimates.

    Fails if simulate returns sessions before they end, or loses part of a session's history.
    """
    assert len(simulation_result.sessions) == 5
    for session in simulation_result.sessions:
      assert session.stop_reason is StopReason.STOPPER
      assert (len(session.administered_items), len(session.responses), len(session.theta_history)) == (10, 10, 11)

  def test_an_examinee_answers_from_a_pattern_drawn_from_its_own_stream(self) -> None:
    """Examinee i answers item j correctly when uniform j of ``SeedSequence(seed, spawn_key=(i, 0))`` is below P_j.

    The probabilities are the 4PL's at the true ability, computed here. Fails if the answers are drawn as the items are
    given, from the session's generator or one shared by the run, if the pattern comes from another stream (the
    session's, or one keyed by anything but the examinee's position), or if the answers use the estimate instead of
    the true ability or leave out an item's guessing or upper asymptote.
    """
    item_bank = ItemBank.generate(40, rng=3)
    thetas = [-1.2, 0.4, 2.0, -0.3]

    result = simulate(make_engine(item_bank, RandomSelector(), 15), true_thetas=thetas, rng=11)

    expected = []
    for i, (theta, session) in enumerate(zip(thetas, result.sessions, strict=True)):
      correct = stream(11, i, 0).random(item_bank.n_items) < probabilities(item_bank, theta)
      expected.append(tuple(bool(correct[item]) for item in session.administered_items))
    assert answers(result) == expected

  def test_two_engines_give_an_examinee_the_same_answer_to_every_item_both_give(self, item_bank: ItemBank) -> None:
    """Common random numbers: on the same seed and examinees, the answer to an item does not depend on the engine.

    Maximum information and random selection give the items in different orders, so fails if an answer is drawn when
    its item is given rather than from a pattern drawn before the test (ADR-0013).
    """
    thetas = np.random.default_rng(2).normal(0, 1, 20)

    by_information = simulate(make_engine(item_bank, MaxInfoSelector(), 20), true_thetas=thetas, rng=9)
    at_random = simulate(make_engine(item_bank, RandomSelector(), 20), true_thetas=thetas, rng=9)

    pairs = shared_answers(by_information, at_random)
    assert len(pairs) > 100
    assert {first for first, _ in pairs} == {True, False}
    assert all(first == second for first, second in pairs)

  def test_without_exposure_feedback_a_session_does_not_depend_on_the_other_examinees(
    self, item_bank: ItemBank
  ) -> None:
    """Examinee 2's session is the same whoever examinees 0 and 1 are, when no session sees exposure rates.

    The initializer and the randomesque selector draw from the session's generator, and the examinees before it give
    tests of other lengths in the two runs, so they make other numbers of draws. Fails if the sessions draw from one
    generator in turn, as the runner before ADR-0013 did, so that each session's draws depend on the sessions before it.
    """
    engine = CatEngine(
      item_bank,
      UniformInitializer(-1.0, 1.0),
      RandomesqueSelector(bin_size=4),
      EAPEstimator(),
      StopRule(StandardErrorStopper(0.6), MaxLengthStopper(15)),
    )

    first = simulate(engine, true_thetas=[-2.5, 2.8, 0.3], rng=4)
    second = simulate(engine, true_thetas=[0.1, -0.2, 0.3], rng=4)

    assert first.examinees()["test_length"][:2].sum() != second.examinees()["test_length"][:2].sum()
    assert first.sessions[2] == second.sessions[2]

  def test_the_examinees_before_a_session_are_all_that_its_exposure_feedback_depends_on(
    self, item_bank: ItemBank
  ) -> None:
    """A run's first four sessions are those of a run of its first four examinees, even under an exposure cap.

    The cap holds back the first session's items from the second (its rates are 1), so the feedback acts. Fails if the
    rates divide the counts by the planned number of examinees (ADR-0012's rejected rate), or if an examinee's streams
    depend on how many examinees the run has.
    """
    engine = make_engine(item_bank, MaxInfoSelector(r_max=0.3), 5)
    thetas = np.random.default_rng(6).normal(0, 1, 8)

    whole = simulate(engine, true_thetas=thetas, rng=1)
    prefix = simulate(engine, true_thetas=thetas[:4], rng=1)

    assert set(whole.sessions[0].administered_items).isdisjoint(whole.sessions[1].administered_items)
    assert prefix.sessions == whole.sessions[:4]


class TestRecordedExaminees:
  """Tests for runs of examinees given by their recorded responses."""

  ROWS = np.random.default_rng(0).random((3, 50)) < 0.5

  @pytest.mark.parametrize(
    "responses",
    [ROWS, ROWS.astype(int), ROWS.astype(float)],
    ids=["bools", "integers", "floats"],
  )
  def test_recorded_examinees_answer_from_their_rows(self, item_bank: ItemBank, responses: np.ndarray) -> None:
    """Every response of session i is row i's entry for that item, whether the rows hold bools, 0 and 1, or 0.0 and 1.0.

    Fails if a session answers from another row or draws its answers, or if the integer or float rows are read
    differently from the Boolean ones.
    """
    result = simulate(make_engine(item_bank, MaxInfoSelector(), 12), responses=responses, rng=0)

    assert result.true_thetas is None
    assert answers(result) == [
      tuple(bool(self.ROWS[i, item]) for item in session.administered_items)
      for i, session in enumerate(result.sessions)
    ]

  def test_items_without_a_recorded_answer_are_excluded_from_the_start(self, item_bank: ItemBank) -> None:
    """A NaN marks an item the examinee did not answer: their session excludes it from the start and never gives it.

    Examinee 0 answered the odd items and examinee 1 the even ones. Fails if a NaN is read as an answer (bool(nan) is
    True), or if the exclusions are not the NaN items in item order.
    """
    responses = self.ROWS[:2].astype(float)
    responses[0, ::2] = np.nan
    responses[1, 1::2] = np.nan

    result = simulate(make_engine(item_bank, MaxInfoSelector(), 12), responses=responses, rng=0)

    assert [session.excluded_items for session in result.sessions] == [tuple(range(0, 50, 2)), tuple(range(1, 50, 2))]
    assert [sorted({item % 2 for item in items}) for items in administered(result)] == [[1], [0]]

  def test_an_examinee_who_runs_out_of_answered_items_ends_with_no_items(self, item_bank: ItemBank) -> None:
    """An examinee who answered three items gets those three, in a test planned for ten, and ends with NO_ITEMS.

    Fails if the session goes on past its answers, or ends with STOPPER as if its planned length were reached.
    """
    responses = np.full((1, item_bank.n_items), np.nan)
    responses[0, [2, 9, 14]] = [1.0, 0.0, 1.0]

    session = simulate(make_engine(item_bank, MaxInfoSelector(), 10), responses=responses, rng=0).sessions[0]

    assert sorted(zip(session.administered_items, session.responses, strict=True)) == [
      (2, True),
      (9, False),
      (14, True),
    ]
    assert session.stop_reason is StopReason.NO_ITEMS

  def test_recorded_examinees_with_known_abilities_answer_from_their_rows(self, item_bank: ItemBank) -> None:
    """With both inputs, the rows give the answers and the true abilities only the metrics.

    An examinee of true ability 3 who answered every item incorrectly, and one of -3 who answered every item correctly,
    keep their recorded answers. The bias is the mean error of the final estimates, worked out by hand. Fails if
    simulate draws the answers from the true abilities when the rows are given.
    """
    responses = np.array([[0] * item_bank.n_items, [1] * item_bank.n_items])

    result = simulate(make_engine(item_bank, MaxInfoSelector(), 6), true_thetas=[3.0, -3.0], responses=responses, rng=0)

    assert answers(result) == [(False,) * 6, (True,) * 6]
    low, high = result.examinees()["theta"].tolist()
    assert result.bias().value == pytest.approx(((low - 3.0) + (high + 3.0)) / 2)


class TestRandomNumbers:
  """Tests for the run's root entropy (ADR-0013)."""

  def test_the_same_seed_gives_the_same_run(self, random_engine: CatEngine) -> None:
    """The streams derive from the seed alone, so the same seed gives the same tests on the same engine.

    Fails if simulate keeps random state across runs, or ignores the seed (a different seed would then give the
    same tests).
    """
    thetas = [-1.0, 0.0, 1.0]
    first = simulate(random_engine, true_thetas=thetas, rng=3)
    second = simulate(random_engine, true_thetas=thetas, rng=3)
    other_seed = simulate(random_engine, true_thetas=thetas, rng=4)

    assert second.sessions == first.sessions
    assert np.array_equal(second.exposure_counts, first.exposure_counts)
    assert administered(other_seed) != administered(first)

  def test_without_a_control_the_selectors_draws_are_the_only_ones_of_a_session(self, item_bank: ItemBank) -> None:
    """Session i's items are random selection's choices from stream (i, 1), drawn here one after another.

    The fixed initializer, the estimator and the stopper draw nothing, so random selection, which picks uniformly among
    the remaining items, makes every draw of the session. Fails if the engine draws from the session's generator when
    there is no exposure control, or passes the selector another set of items to choose from: every later draw, and so
    every item, would move.
    """
    result = simulate(make_engine(item_bank, RandomSelector(), 5), true_thetas=[-1.0, 0.0, 1.0], rng=7)

    expected = []
    for i in range(3):
      session_rng, items = stream(7, i, 1), []
      for _ in range(5):
        items.append(int(session_rng.choice([item for item in range(item_bank.n_items) if item not in items])))
      expected.append(tuple(items))
    assert administered(result) == expected

  def test_two_runs_without_rng_give_other_tests(self, random_engine: CatEngine) -> None:
    """A run without rng takes fresh entropy, so two such runs give other final estimates.

    Each of the three sessions draws 10 of 50 items at random, so equal final estimates for all three are
    practically impossible. Fails if rng defaults to a fixed seed, which makes every unseeded run the same.
    """
    thetas = [-1.0, 0.0, 1.0]

    assert not np.array_equal(
      simulate(random_engine, true_thetas=thetas).examinees()["theta"],
      simulate(random_engine, true_thetas=thetas).examinees()["theta"],
    )

  def test_a_generator_passed_as_rng_is_drawn_from_once_per_run(self, random_engine: CatEngine) -> None:
    """A run takes one number from a generator as its root entropy, so the generator advances and a second run differs.

    Runs of five examinees and of one advance equal generators to the same state, and equal generators give equal runs.
    Fails if the generator is read without being drawn from, which makes replications in a loop identical, or if the
    sessions draw from it directly, which advances it by an amount that grows with the run.
    """
    thetas = [-1.0, 0.0, 1.0, 0.5, -0.5]
    five, one = np.random.default_rng(5), np.random.default_rng(5)

    first = simulate(random_engine, true_thetas=thetas, rng=five)
    simulate(random_engine, true_thetas=thetas[:1], rng=one)

    assert five.bit_generator.state == one.bit_generator.state != np.random.default_rng(5).bit_generator.state
    assert simulate(random_engine, true_thetas=thetas, rng=np.random.default_rng(5)).sessions == first.sessions
    assert administered(simulate(random_engine, true_thetas=thetas, rng=five)) != administered(first)

  @pytest.mark.parametrize(
    ("arguments", "lambda_prior"),
    [
      ({"true_thetas": [0.0, math.nan]}, False),
      ({"true_thetas": [0.0], "parts": 2}, False),
      ({"true_thetas": [0.0, 1.0], "parts": 2, "workers": 2}, True),
    ],
    ids=["nan-true-ability", "more-parts-than-examinees", "an-engine-the-workers-cannot-import"],
  )
  def test_a_call_that_fails_its_checks_leaves_the_generator_as_it_was(
    self, random_engine: CatEngine, arguments: dict[str, object], lambda_prior: bool
  ) -> None:
    """A call rejected for its inputs draws nothing; fails if the entropy is drawn before the inputs are checked.

    An engine with a lambda prior cannot go to the workers, and its check is one of the inputs' checks.
    """
    engine = (
      dataclasses.replace(random_engine, estimator=EAPEstimator(log_prior=lambda theta: -0.5 * theta**2))
      if lambda_prior
      else random_engine
    )
    rng = np.random.default_rng(5)

    with pytest.raises(ValueError, match=r"finite|every part needs at least one examinee|cannot import"):
      simulate(engine, rng=rng, **arguments)  # ty: ignore[invalid-argument-type]

    assert rng.random() == np.random.default_rng(5).random()


class TestExposure:
  """Tests for the exposure feedback and the proposal counts of a run (ADR-0011, ADR-0012)."""

  def test_each_session_sees_the_exposure_of_the_sessions_before_it(self, item_bank: ItemBank) -> None:
    """simulate counts exposure across its sessions, so each sees catR's rates of the examinees before it.

    Every session administers items 0 and 1, so from the second session on both have gone to every earlier
    examinee: rate 1 at both of the session's selections, and 0 for every other item. The first session sees no
    rates (ADR-0012). Fails if each session counts afresh (rates stay 0), or if the rates divide by the planned
    number of examinees (0.25, 0.5, 0.75) or count the session in progress (0.5, 0.67, 0.75).
    """
    selector = FirstRemainingSelector()

    result = simulate(make_engine(item_bank, selector, 2), true_thetas=np.zeros(4))

    first_session, later_sessions = selector.seen_exposure_rates[:2], selector.seen_exposure_rates[2:]
    assert first_session == [None, None]
    expected = np.zeros(item_bank.n_items)
    expected[:2] = 1.0
    assert [None if rates is None else rates.tolist() for rates in later_sessions] == [expected.tolist()] * 6
    assert result.exposure_rates().value[:2].tolist() == [1.0, 1.0]

  def test_a_control_that_accepts_without_drawing_reproduces_the_run_without_one(self, item_bank: ItemBank) -> None:
    """With a control that accepts every proposal and draws nothing, a seeded run gives the run without a control.

    Fails if asking the control changes what the selector is given or draws from the generator, or if accepted
    proposals are counted twice (the proposal counts would exceed the exposure counts).
    """
    thetas = [-1.0, 0.0, 1.0]
    without = simulate(make_engine(item_bank, RandomesqueSelector(bin_size=3), 5), true_thetas=thetas, rng=7)
    accepting = simulate(
      make_engine(item_bank, RandomesqueSelector(bin_size=3), 5, AcceptingControl()), true_thetas=thetas, rng=7
    )

    assert accepting.sessions == without.sessions
    assert [session.excluded_items for session in accepting.sessions] == [(), (), ()]
    assert np.array_equal(accepting.proposal_counts, accepting.exposure_counts)

  @pytest.mark.parametrize("recorded", [False, True], ids=["generated", "recorded-with-missing-answers"])
  def test_a_runs_proposals_are_its_administrations_plus_its_declines(
    self, item_bank: ItemBank, recorded: bool
  ) -> None:
    """Under a control that declines at random, each item's proposals are its administrations plus its declines.

    A session excludes its examinee's missing answers first and adds each decline after them, so the declines are its
    excluded items past the missing ones. Fails if a decline is not counted or is counted as an administration, if the
    missing answers are counted as proposals, which only the selector and the exposure control make, or if
    proposal_rates divides by anything but the number of sessions.
    """
    rows = (np.random.default_rng(8).random((20, item_bank.n_items)) < 0.5).astype(float)
    rows[:, :10] = np.nan
    engine = make_engine(item_bank, RandomSelector(), 5, CoinControl())
    examinees = {"responses": rows} if recorded else {"true_thetas": np.zeros(20)}
    missing = 10 if recorded else 0

    result = simulate(engine, rng=3, **examinees)  # ty: ignore[invalid-argument-type]

    declines = np.bincount(
      [item for session in result.sessions for item in session.excluded_items[missing:]], minlength=item_bank.n_items
    )
    assert [session.excluded_items[:missing] for session in result.sessions] == [tuple(range(missing))] * 20
    assert int(declines.sum()) > 0
    assert result.examinees()["test_length"].tolist() == [5] * 20
    assert np.array_equal(result.proposal_counts, result.exposure_counts + declines)
    assert np.allclose(result.proposal_rates().value, (result.exposure_counts + declines) / 20)


class TestParts:
  """Tests for a run split into parts, each with exposure counts of its own (ADR-0015)."""

  def test_one_part_is_the_sequential_run(self, item_bank: ItemBank) -> None:
    """With parts=1, every session runs in examinee order, capped against catR's rates over all the sessions before it.

    The run is rebuilt here from the engine's public steps: each examinee's pattern and session generator come from
    its streams, and the rates from counts kept over the whole run. The initializer and the progressive selector draw
    from the session's generator, and the selector's cap reads the rates. Fails if the one part runs its sessions out
    of order, resets its counts, or gives an examinee the streams of another position.
    """
    engine = make_engine(
      item_bank, ProgressiveSelector(test_size=5, r_max=0.3), 5, initializer=UniformInitializer(-1.0, 1.0)
    )
    thetas = np.random.default_rng(3).normal(0, 1, 12)

    result = simulate(engine, true_thetas=thetas, rng=8, parts=1)

    counts, sessions = np.zeros(item_bank.n_items, dtype=int), []
    for i, theta in enumerate(thetas.tolist()):
      pattern = stream(8, i, 0).random(item_bank.n_items) < probabilities(item_bank, theta)
      rng = stream(8, i, 1)
      rates = counts / i if i else None
      session = engine.start(rng=rng)
      while (selection := engine.select(session, rng=rng, exposure_rates=rates)).item is not None:
        session = engine.answer(selection.session, selection.item, bool(pattern[selection.item]), rng=rng)
      sessions.append(selection.session)
      counts[list(selection.session.administered_items)] += 1
    assert result.sessions == tuple(sessions)
    assert result.exposure_counts.tolist() == counts.tolist()
    assert result.parts == 1

  def test_each_part_counts_its_own_exposure(self, item_bank: ItemBank) -> None:
    """With exposure feedback, a session sees the exposure of the sessions before it in its part, and of no other.

    Every session administers items 0 and 1. Five examinees in two parts are the first three and the last two, as
    ``numpy.array_split`` sizes them, so the first session of each part sees no rates and each later one rates of 1
    for both items. The run's counts are the parts' sums, 5 for each item. Fails if the parts share their counts (the
    last session would see rates of 4), if a session divides by the sessions before it in the run (the fourth would see
    rates of 0 rather than none, and the fifth 1/4), if the parts are sized 2 and 3, or if the run keeps one part's
    counts.
    """
    selector = FirstRemainingSelector()

    result = simulate(make_engine(item_bank, selector, 2), true_thetas=np.zeros(5), parts=2)

    exposed = [1.0, 1.0] + [0.0] * (item_bank.n_items - 2)
    seen = [None if rates is None else rates.tolist() for rates in selector.seen_exposure_rates]
    assert seen == [None] * 2 + [exposed] * 4 + [None] * 2 + [exposed] * 2
    assert result.exposure_counts.tolist() == [5, 5] + [0] * (item_bank.n_items - 2)
    assert result.parts == 2

  @pytest.mark.parametrize("parts", [2, 3, 7])
  def test_without_exposure_feedback_the_parts_do_not_change_the_output(self, item_bank: ItemBank, parts: int) -> None:
    """Without an r_max cap, a run in parts gives every examinee the session that the run in one part gives them.

    The initializer and the randomesque selector draw from each session's generator, and the answers come from each
    examinee's pattern; seven parts put every examinee in a part of their own. Fails if an examinee of a later part
    draws from the streams of its position in its part rather than in the run (ADR-0013), or if the run's exposure
    counts are not the sum of its parts'.
    """
    engine = CatEngine(
      item_bank,
      UniformInitializer(-1.0, 1.0),
      RandomesqueSelector(bin_size=4),
      EAPEstimator(),
      StopRule(StandardErrorStopper(0.6), MaxLengthStopper(8)),
    )
    thetas = np.random.default_rng(5).normal(0, 1, 7)

    whole = simulate(engine, true_thetas=thetas, rng=2)
    split = simulate(engine, true_thetas=thetas, rng=2, parts=parts)

    assert split.sessions == whole.sessions
    assert split.exposure_counts.tolist() == whole.exposure_counts.tolist()
    assert (whole.parts, split.parts) == (1, parts)


@pytest.mark.parametrize(
  ("parts", "workers"), [(1, 1), (2, 1), (2, 2)], ids=["sequential", "two-parts", "two-parts-on-two-workers"]
)
def test_progress_is_told_after_each_session_how_many_have_finished(
  random_engine: CatEngine, parts: int, workers: int
) -> None:
  """progress is called in the calling process once per session, with the sessions finished and the total.

  Two parts hold two examinees and one. Fails if it is called before the first session ends or not after the last,
  with the examinee's index rather than the count of finished sessions, if each part counts from 1 again, if it is
  called once per part, or in the workers (this list would stay empty, and a lambda cannot reach them), or if passing
  it changes the sessions.
  """
  calls: list[tuple[int, int]] = []
  thetas = [-0.5, 0.0, 0.5]

  shown = simulate(
    random_engine,
    true_thetas=thetas,
    rng=0,
    parts=parts,
    workers=workers,
    progress=lambda finished, total: calls.append((finished, total)),
  )

  assert calls == [(1, 3), (2, 3), (3, 3)]
  assert shown.sessions == simulate(random_engine, true_thetas=thetas, rng=0).sessions


@pytest.mark.parametrize(
  ("arguments", "error", "message"),
  [
    pytest.param({}, TypeError, "simulate needs examinees: true_thetas for generated ones", id="no-examinees"),
    pytest.param(
      {"true_thetas": 5},
      TypeError,
      r"one true ability per examinee, not a single number \(5\).*rng\.normal\(0, 1, n\)",
      id="a-number-of-examinees",
    ),
    pytest.param({"true_thetas": np.int64(5)}, TypeError, r"not a single number", id="a-numpy-number-of-examinees"),
    pytest.param({"true_thetas": []}, ValueError, "true_thetas cannot be empty", id="no-true-abilities"),
    pytest.param(
      {"true_thetas": [[0.0, 1.0]]},
      TypeError,
      r"true_thetas must be one-dimensional, one true ability per examinee, got shape \(1, 2\)",
      id="true-abilities-in-rows",
    ),
    pytest.param(
      {"true_thetas": [0.0, math.nan]},
      ValueError,
      r"true_thetas\[1\] is nan; true abilities must be finite numbers",
      id="nan-true-ability",
    ),
    pytest.param(
      {"true_thetas": [0.0, -math.inf]},
      ValueError,
      r"true_thetas\[1\] is -inf; true abilities must be finite numbers",
      id="infinite-true-ability",
    ),
    pytest.param(
      {"true_thetas": [True, False]},
      TypeError,
      "true_thetas holds bool values; true abilities must be real numbers",
      id="boolean-true-abilities",
    ),
    pytest.param(
      {"responses": [1, 0, 1]},
      TypeError,
      r"responses must be two-dimensional, one row per examinee and one column per item of the bank, got shape \(3,\)",
      id="one-row-of-responses-flat",
    ),
    pytest.param({"responses": np.zeros((0, 50))}, ValueError, "responses cannot be empty", id="no-response-rows"),
    pytest.param({"responses": []}, ValueError, "responses cannot be empty", id="no-responses"),
    pytest.param(
      {"responses": np.zeros((2, 49))},
      ValueError,
      r"responses has shape \(2, 49\); there must be one column per item of the bank \(50\)",
      id="a-missing-column",
    ),
    pytest.param(
      {"responses": np.where(np.arange(100).reshape(2, 50) == 54, 2, 0)},
      ValueError,
      r"responses\[1, 4\] is 2, for item 4 \('item_04'\); an integer response must be 0 or 1",
      id="integer-response-2",
    ),
    pytest.param(
      {"responses": np.where(np.arange(100).reshape(2, 50) == 7, 0.5, 1.0)},
      ValueError,
      r"responses\[0, 7\] is 0.5, for item 7 \('item_07'\); a response must be 0 or 1, or NaN where it is missing",
      id="float-response-one-half",
    ),
    pytest.param(
      {"responses": np.where(np.arange(100).reshape(2, 50) == 60, np.inf, 0.0)},
      ValueError,
      r"responses\[1, 10\] is inf, for item 10 \('item_10'\); a response must be 0 or 1, or NaN where it is missing",
      id="infinite-response",
    ),
    pytest.param(
      {"responses": [[None] * 50]},
      TypeError,
      "responses holds object values; responses must be bools, the integers 0 and 1, or the floats 0.0 and 1.0 with "
      "NaN for a missing response",
      id="none-for-missing",
    ),
    pytest.param(
      {"true_thetas": [0.0, 1.0, 2.0], "responses": np.zeros((2, 50))},
      ValueError,
      "true_thetas has 3 entries and responses 2 rows; with both, there must be one true ability per row of responses",
      id="abilities-for-other-rows",
    ),
    pytest.param(
      {"true_thetas": [0.0], "rng": 1.5},
      TypeError,
      r"rng is 1.5; it must be None, an integer seed or a numpy.random.Generator",
      id="float-seed",
    ),
    pytest.param(
      {"true_thetas": [0.0], "rng": True},
      TypeError,
      r"rng is True; it must be None, an integer seed or a numpy.random.Generator",
      id="boolean-seed",
    ),
    pytest.param(
      {"true_thetas": [0.0], "rng": -1}, ValueError, "rng is -1; a seed must not be negative", id="negative-seed"
    ),
    pytest.param(
      {"true_thetas": [0.0], "parts": 0}, ValueError, "parts is 0; it must be a positive integer", id="no-parts"
    ),
    pytest.param(
      {"true_thetas": [0.0], "parts": 1.5}, TypeError, "parts is 1.5; it must be a positive integer", id="float-parts"
    ),
    pytest.param(
      {"true_thetas": [0.0], "parts": True}, TypeError, "parts is True; it must be a positive integer", id="parts-flag"
    ),
    pytest.param(
      {"true_thetas": [0.0], "workers": -2}, ValueError, "workers is -2; it must be a positive integer", id="no-workers"
    ),
    pytest.param(
      {"true_thetas": [0.0], "workers": "4"},
      TypeError,
      "workers is '4'; it must be a positive integer",
      id="workers-string",
    ),
    pytest.param(
      {"true_thetas": [0.0, 1.0], "parts": 3},
      ValueError,
      "parts is 3 and the run has 2 examinees; every part needs at least one examinee",
      id="more-parts-than-generated-examinees",
    ),
    pytest.param(
      {"responses": np.zeros((2, 50)), "parts": 3},
      ValueError,
      "parts is 3 and the run has 2 examinees",
      id="more-parts-than-recorded-examinees",
    ),
    pytest.param(
      {"true_thetas": [0.0], "progress": True},
      TypeError,
      r"progress is True; it must be None or a callable, called as progress\(finished, total\)",
      id="progress-flag",
    ),
  ],
)
def test_a_bad_input_fails_before_any_session_runs(
  random_engine: CatEngine, arguments: dict[str, object], error: type[Exception], message: str
) -> None:
  """Each malformed input raises, naming the input, the position and value where there is one, and the rule.

  Fails if simulate reads a bad input silently: a single number as a count or as one ability, a NaN ability whose
  answers would all be wrong, booleans as abilities 0 and 1, a response of 0.5 or 2 as correct, a missing column that
  shifts every item, None as a missing answer, true abilities paired with the wrong rows, a float or boolean seed, a
  count of parts or workers that is not a positive integer, more parts than examinees (an empty part), or the old
  ``progress=True`` flag.
  """
  with pytest.raises(error, match=message):
    simulate(random_engine, **arguments)  # ty: ignore[invalid-argument-type]


def test_simulate_rejects_what_is_not_an_engine(item_bank: ItemBank) -> None:
  """An item bank passed as the engine fails with a message naming the type, not an AttributeError deep in the run."""
  with pytest.raises(TypeError, match="engine is a ItemBank; it must be a CatEngine"):
    simulate(item_bank, true_thetas=[0.0])  # ty: ignore[invalid-argument-type]


def finished(
  items: tuple[int, ...] = (),
  theta: float = 0.0,
  se: float = 1.0,
  stop_reason: StopReason = StopReason.STOPPER,
) -> Session:
  """Return an ended session that administered ``items``, all answered correctly, whose final estimate is (theta, se).

  A session of no items keeps its initial estimate, whose standard error is infinite.
  """
  if not items:
    return Session(estimates=(Estimate(theta, math.inf, 0),), stop_reason=stop_reason)
  estimates = (Estimate(0.0, math.inf, 0), *[Estimate(0.0, 1.0, 1)] * (len(items) - 1), Estimate(theta, se, 1))
  return Session(administered_items=items, responses=(True,) * len(items), estimates=estimates, stop_reason=stop_reason)


def result_of(
  sessions: tuple[Session, ...] | list[Session],
  n_items: int = 5,
  true_thetas: tuple[float, ...] | list[float] | FloatArray | None = None,
  *,
  names: tuple[str, ...] | None = None,
) -> SimulationResult:
  """Return a result of the sessions on a bank of n_items items, whose proposals are its administrations."""
  generated = ItemBank.generate(n_items, rng=0)
  counts = np.bincount(np.array([item for session in sessions for item in session.administered_items], dtype=int))
  return SimulationResult(
    item_bank=generated if names is None else ItemBank(generated.items, names=names),
    sessions=tuple(sessions),
    proposal_counts=np.pad(counts, (0, n_items - len(counts))),
    duration=0.0,
    true_thetas=None if true_thetas is None else np.array(true_thetas),
  )


# Two hand-built sessions with known estimates and evaluation counts, and their examinees' true abilities. The first
# gives items 3 and 1 and ends at 0.6 (SE 0.9) for a true 0.5, its estimates costing 7 and 5 evaluations; the second
# gives item 2, runs out of items and ends at -0.8 (SE 1.1) for a true -1.0, its estimate costing 4. The errors of the
# final estimates are 0.1 and 0.2.
TWO_SESSIONS = (
  Session(
    administered_items=(3, 1),
    responses=(True, False),
    estimates=(Estimate(0.0, math.inf, 0), Estimate(0.4, 1.2, 7), Estimate(0.6, 0.9, 5)),
    stop_reason=StopReason.STOPPER,
  ),
  Session(
    administered_items=(2,),
    responses=(False,),
    estimates=(Estimate(0.0, math.inf, 0), Estimate(-0.8, 1.1, 4)),
    stop_reason=StopReason.NO_ITEMS,
  ),
)
TWO_TRUE_THETAS = (0.5, -1.0)
NAMES = ("q0", "q1", "q2", "q3", "q4")


class TestTables:
  """Tests for the per-step and per-session tables of a result (ADR-0016)."""

  def test_steps_has_one_row_per_session_and_administered_item(self) -> None:
    """The steps of the two sessions, worked out by hand, with each item's index and name and the examinee's ability.

    Fails if the rows include the initial estimate, if positions count from 1 or run across sessions, if an item is
    named by its position in the test rather than its index, if a session gets another's true ability, or if a column
    holds Python objects rather than the integers, Booleans, floats and strings pandas reads.
    """
    steps = result_of(TWO_SESSIONS, true_thetas=TWO_TRUE_THETAS, names=NAMES).steps()

    assert {name: column.tolist() for name, column in steps.items()} == {
      "session": [0, 0, 1],
      "position": [0, 1, 0],
      "item": [3, 1, 2],
      "name": ["q3", "q1", "q2"],
      "response": [True, False, False],
      "theta": [0.4, 0.6, -0.8],
      "se": [1.2, 0.9, 1.1],
      "evaluations": [7.0, 5.0, 4.0],
      "true_theta": [0.5, 0.5, -1.0],
    }
    assert [column.dtype.kind for column in steps.values()] == ["i", "i", "i", "U", "b", "f", "f", "f", "f"]

  def test_examinees_has_one_row_per_session(self) -> None:
    """Each session's length, final estimate, standard error, stop reason, true ability and decile, by hand.

    Of two true abilities, -1.0 is the lowest cut and falls in decile 0, and 0.5 is the highest and in decile 9. Fails
    if the estimate is an earlier one than the final, if the length counts the estimates (one more than the items),
    if the stop reason is the enum's repr rather than its value, or if the decile is not that of the session's own
    true ability.
    """
    examinees = result_of(TWO_SESSIONS, true_thetas=TWO_TRUE_THETAS).examinees()

    assert {name: column.tolist() for name, column in examinees.items()} == {
      "session": [0, 1],
      "test_length": [2, 1],
      "theta": [0.6, -0.8],
      "se": [0.9, 1.1],
      "stop_reason": ["stopper", "no_items"],
      "true_theta": [0.5, -1.0],
      "decile": [9, 0],
    }
    assert [column.dtype.kind for column in examinees.values()] == ["i", "i", "f", "f", "U", "f", "i"]

  def test_without_true_abilities_the_tables_have_no_true_ability_columns(self) -> None:
    """Recorded examinees' abilities are unknown, so neither table has a true ability or decile column.

    Fails if an unknown ability is written as NaN, which ADR-0016 rules out as a marker, or as 0.
    """
    result = result_of(TWO_SESSIONS)

    assert list(result.steps()) == ["session", "position", "item", "name", "response", "theta", "se", "evaluations"]
    assert list(result.examinees()) == ["session", "test_length", "theta", "se", "stop_reason"]

  def test_an_estimate_without_an_evaluation_count_has_nan_in_the_steps(self) -> None:
    """A custom estimator may give no count: its step has NaN, so the mean count of the run's estimates is 3.5.

    The first session's estimates cost 3 and an unknown number, the second's 4. Fails if an unknown count is read as
    0, which makes the mean 7 / 3.
    """
    uncounted = Session(
      administered_items=(3, 1),
      responses=(True, False),
      estimates=(Estimate(0.0, math.inf, 0), Estimate(0.4, 1.2, 3), Estimate(0.6, 0.9)),
      stop_reason=StopReason.STOPPER,
    )

    evaluations = result_of([uncounted, TWO_SESSIONS[1]]).steps()["evaluations"]

    assert evaluations[[0, 2]].tolist() == [3.0, 4.0]
    assert math.isnan(evaluations[1])
    assert np.nanmean(evaluations) == pytest.approx(3.5)

  def test_a_run_without_items_has_an_empty_steps_table_of_the_same_types(self) -> None:
    """Sessions that stopped before their first item give a steps table of no rows, with the usual dtypes.

    Fails if the empty columns fall back to NumPy's default float dtype, whose item indices cannot index the bank.
    """
    steps = result_of([finished(), finished()]).steps()

    assert [len(column) for column in steps.values()] == [0] * 8
    assert [column.dtype.kind for column in steps.values()] == ["i", "i", "i", "U", "b", "f", "f", "f"]


class TestResultValue:
  """Tests for what a result holds and checks when it is built (ADR-0016)."""

  def test_a_result_and_its_arrays_cannot_change(self) -> None:
    """A result keeps copies of the arrays it is given, read-only, and its sessions as a tuple.

    Fails if the result keeps the caller's arrays, so that changing them afterwards changes the result, if it keeps
    the list it was given, or if a field or an array can be assigned.
    """
    true_thetas = np.array(TWO_TRUE_THETAS)
    proposal_counts = np.array([0, 1, 1, 1, 0])
    result = SimulationResult(
      item_bank=ItemBank.generate(5, rng=0),
      sessions=list(TWO_SESSIONS),  # ty: ignore[invalid-argument-type]
      proposal_counts=proposal_counts,
      duration=0.0,
      true_thetas=true_thetas,
    )
    true_thetas[0], proposal_counts[1] = 9.0, 2

    assert result.sessions == TWO_SESSIONS
    assert result.true_thetas is not None
    assert (result.true_thetas.tolist(), result.proposal_counts.tolist()) == ([0.5, -1.0], [0, 1, 1, 1, 0])
    with pytest.raises(dataclasses.FrozenInstanceError):
      result.duration = 1.0  # ty: ignore[invalid-assignment]
    for array in (result.true_thetas, result.exposure_counts, result.proposal_counts):
      with pytest.raises(ValueError, match="read-only"):
        array[0] = 1

  @pytest.mark.parametrize(
    ("changes", "error", "message"),
    [
      pytest.param({"sessions": ()}, ValueError, "sessions is empty; a result holds at least one session", id="none"),
      pytest.param(
        {"sessions": (TWO_SESSIONS[0], "session")},
        TypeError,
        r"sessions\[1\] is a str; a result's sessions must be Session objects",
        id="not-a-session",
      ),
      pytest.param(
        {"sessions": (Session(estimates=(Estimate(0.0, math.inf),)),)},
        ValueError,
        r"sessions\[0\] has not ended, its stop_reason is None; a result holds finished sessions",
        id="unfinished",
      ),
      pytest.param(
        {"item_bank": ItemBank.generate(3, rng=0), "proposal_counts": [0, 1, 1]},
        ValueError,
        r"sessions\[0\].administered_items\[0\] is 3; the item bank has 3 items, 0 to 2",
        id="item-outside-the-bank",
      ),
      pytest.param(
        {"true_thetas": [0.5, -1.0, 0.2]},
        ValueError,
        re.escape("true_thetas has shape (3,); there must be one true ability per session (2)"),
        id="true-abilities-of-another-run",
      ),
      pytest.param(
        {"true_thetas": [0.5, math.nan]},
        ValueError,
        r"true_thetas\[1\] is nan; true abilities must be finite, and true_thetas None when they are unknown",
        id="nan-for-unknown",
      ),
      pytest.param(
        {"proposal_counts": [0, 0, 1, 1, 0]},
        ValueError,
        r"proposal_counts\[1\] is 0, for item 1 \('item_1'\); an item's proposals are at least its 1 administrations "
        "and at most the 2 sessions",
        id="fewer-proposals-than-administrations",
      ),
      pytest.param(
        {"proposal_counts": [0, 3, 1, 1, 0]},
        ValueError,
        r"proposal_counts\[1\] is 3, for item 1 \('item_1'\); an item's proposals are at least",
        id="more-proposals-than-sessions",
      ),
      pytest.param(
        {"proposal_counts": [0, 1, 1, 1]},
        ValueError,
        re.escape("proposal_counts has shape (4,); there must be one count per item of the bank (5)"),
        id="a-count-missing",
      ),
      pytest.param(
        {"proposal_counts": [0.0, 0.5, 0.5, 0.5, 0.0]},
        TypeError,
        "proposal_counts holds float64 values; counts must be integers",
        id="rates-for-counts",
      ),
      pytest.param(
        {"item_bank": ItemBank.generate(5, rng=0).items},
        TypeError,
        "item_bank is a ndarray; it must be an ItemBank",
        id="item-matrix-for-bank",
      ),
      pytest.param(
        {"duration": math.nan},
        ValueError,
        "duration is nan; a run's wall-clock time is a finite, non-negative number of seconds",
        id="nan-duration",
      ),
      pytest.param({"duration": "1.5"}, TypeError, "duration is '1.5'; it must be a number of seconds", id="text"),
      pytest.param({"parts": 0}, ValueError, "parts is 0; a run has at least one part", id="no-parts"),
      pytest.param({"parts": True}, TypeError, "parts is True; it must be an integer", id="boolean-parts"),
    ],
  )
  def test_a_result_that_does_not_hold_together_raises(
    self, changes: dict[str, object], error: type[Exception], message: str
  ) -> None:
    """Each inconsistent input raises, naming it, the position and value where there is one, and the rule.

    Fails if a result is built on an item matrix instead of a bank (an AttributeError later), with a duration that is
    not a number of seconds, from no session (every metric would average nothing), from a session still running, with
    an item the bank cannot name, with true abilities of another run or NaN for unknown ones, with proposal counts that
    cannot be its own, or with no part.
    """
    arguments: dict[str, object] = {
      "item_bank": ItemBank.generate(5, rng=0),
      "sessions": TWO_SESSIONS,
      "proposal_counts": [0, 1, 1, 1, 0],
      "duration": 0.0,
    }

    with pytest.raises(error, match=message):
      SimulationResult(**(arguments | changes))  # ty: ignore[invalid-argument-type]


class TestResultMetrics:
  """Tests for the metrics a result computes from its sessions, overall (ADR-0016)."""

  def test_accuracy_metrics_compare_final_estimates_with_true_abilities(self) -> None:
    """Errors 0.1 and 0.2 give the bias 0.15 and the MSE 0.025, each with its error worked out by hand.

    Bias: deviations -0.05 and 0.05, error sqrt(0.005 / 2) = 0.05. MSE: squared errors 0.01 and 0.04, deviations
    -0.015 and 0.015, error sqrt(0.00045 / 2) = 0.015. RMSE: sqrt(0.025), error 0.015 / (2 sqrt(0.025)). Fails if a
    metric reads an earlier estimate than the final one, swaps the sign of the bias, or pairs an estimate with another
    session's true ability.
    """
    result = result_of(TWO_SESSIONS, true_thetas=TWO_TRUE_THETAS)

    assert result.bias() == pytest.approx((0.15, 0.05))
    assert result.mse() == pytest.approx((0.025, 0.015))
    assert result.rmse() == pytest.approx((math.sqrt(0.025), 0.015 / (2 * math.sqrt(0.025))))

  def test_test_length_and_standard_error_are_means_over_the_sessions(self) -> None:
    """Lengths 2 and 1 give 1.5 with an error of 0.5, and standard errors 0.9 and 1.1 give 1.0 with an error of 0.1.

    Neither needs the true abilities, which this result lacks. Fails if the standard error is that of the initial,
    infinite, estimate, if the length counts the estimates (2.5), or if these metrics ask for true abilities.
    """
    result = result_of(TWO_SESSIONS)

    assert result.test_length() == pytest.approx((1.5, 0.5))
    assert result.se() == pytest.approx((1.0, 0.1))

  def test_exposure_and_proposal_rates_are_shares_of_the_sessions(self) -> None:
    """Items 3 and 1 go to the first session and item 2 to the second, and the selector proposed item 0 once too.

    The exposure counts are [0, 1, 1, 1, 0], counted from the sessions, so the rates are halves with a binomial error
    of sqrt(0.25 / 2); proposals [1, 1, 2, 1, 0] give rates [0.5, 0.5, 1, 0.5, 0]. Fails if the rates divide by the
    number of items or of steps rather than sessions, or if the proposal rates read the exposure counts.
    """
    result = SimulationResult(
      item_bank=ItemBank.generate(5, rng=0),
      sessions=TWO_SESSIONS,
      proposal_counts=np.array([1, 1, 2, 1, 0]),
      duration=0.0,
    )
    half = math.sqrt(0.25 / 2)

    exposure, proposal = result.exposure_rates(), result.proposal_rates()

    assert result.exposure_counts.tolist() == [0, 1, 1, 1, 0]
    assert exposure.value.tolist() == [0.0, 0.5, 0.5, 0.5, 0.0]
    assert exposure.mc_se == pytest.approx([0.0, half, half, half, 0.0])
    assert proposal.value.tolist() == [0.5, 0.5, 1.0, 0.5, 0.0]
    assert proposal.mc_se == pytest.approx([half, half, 0.0, half, 0.0])

  def test_the_overlap_rate_needs_tests_of_one_length_of_at_least_one_item(self) -> None:
    """Tests [0, 1] and [0, 2] of a 5-item bank have rates 1, 0.5, 0.5, 0, 0: overlap (1 + 0.25 + 0.25) / 2 = 0.75.

    Both tests' rates sum to 1.5, so the error is 0. Tests of 1 and 2 items, or of none, have no overlap rate, whose
    formula assumes one fixed length. Fails if a number is computed from a length the sessions do not share, or from a
    length of 0.
    """
    assert result_of([finished((0, 1)), finished((0, 2))]).overlap_rate() == pytest.approx((0.75, 0.0))
    assert result_of([finished((0,)), finished((0, 1))]).overlap_rate() is None
    assert result_of([finished(), finished()]).overlap_rate() is None

  @pytest.mark.parametrize(
    ("metric", "needed_by"),
    [
      pytest.param(lambda result: result.bias(), "bias", id="bias"),
      pytest.param(lambda result: result.mse(), "mse", id="mse"),
      pytest.param(lambda result: result.rmse(), "rmse", id="rmse"),
      pytest.param(lambda result: result.deciles(), "deciles", id="deciles"),
      pytest.param(lambda result: result.ability_levels(), "ability_levels", id="ability-levels"),
      pytest.param(lambda result: result.test_length(decile=0), "a metric by decile", id="length-by-decile"),
      pytest.param(lambda result: result.se(decile=0), "a metric by decile", id="se-by-decile"),
      pytest.param(lambda result: result.exposure_rates(decile=0), "a metric by decile", id="exposure-by-decile"),
      pytest.param(lambda result: result.overlap_rate(decile=0), "a metric by decile", id="overlap-by-decile"),
    ],
  )
  def test_metrics_that_need_true_abilities_raise_without_them(
    self, metric: Callable[[SimulationResult], object], needed_by: str
  ) -> None:
    """Recorded examinees' abilities are unknown, so the accuracy metrics and every metric by ability raise (ADR-0014).

    Fails if a metric compares the estimates with zeros or NaN, or groups the sessions by their estimates instead.
    """
    with pytest.raises(
      ValueError,
      match=f"{needed_by} needs the examinees' true abilities, and this result has none; pass true_thetas to simulate",
    ):
      metric(result_of(TWO_SESSIONS))


class TestAbilityLevels:
  """Tests for the metrics at each distinct true ability (ADR-0016)."""

  def test_the_ability_levels_table_of_a_hand_worked_run(self) -> None:
    """Five examinees at abilities 1, -1, 1, 0 and -1, in that order, give three rows, from -1 to 1.

    At -1, errors 0.1 and 0.3 give a bias of 0.2 with an error of 0.1, an MSE of 0.05 with an error of 0.04 (squared
    errors 0.01 and 0.09), and lengths 1 and 3 a mean of 2 with an error of 1. At 0, one examinee gives an error of -0.2
    and NaN Monte Carlo errors, which need two sessions. At 1, both take two items, (0, 1) and (0, 2), of a 4-item
    bank: rates 1, 0.5, 0.5 and 0 give an overlap of (1 + 0.25 + 0.25) / 2 = 0.75. Fails if rows follow the session
    order rather than the ability, if a row mixes two abilities' sessions, or if a single session's errors are 0.
    """
    sessions = [
      finished((0, 1), theta=1.5, se=0.4),
      finished((2,), theta=-0.9, se=0.5),
      finished((0, 2), theta=0.5, se=0.6),
      finished((3,), theta=-0.2, se=0.8),
      finished((1, 2, 3), theta=-0.7, se=0.7),
    ]

    table = result_of(sessions, n_items=4, true_thetas=[1.0, -1.0, 1.0, 0.0, -1.0]).ability_levels()

    assert table["true_theta"].tolist() == [-1.0, 0.0, 1.0]
    assert table["n_sessions"].tolist() == [2, 1, 2]
    assert table["bias"] == pytest.approx([0.2, -0.2, 0.0])
    assert table["bias_mc_se"][[0, 2]] == pytest.approx([0.1, 0.5])
    assert table["mse"][0] == pytest.approx(0.05)
    assert table["mse_mc_se"][0] == pytest.approx(0.04)
    assert table["test_length"] == pytest.approx([2.0, 1.0, 2.0])
    assert table["test_length_mc_se"][0] == pytest.approx(1.0)
    assert table["se"] == pytest.approx([0.6, 0.8, 0.5])
    assert all(np.isnan(table[name][1]) for name in ("bias_mc_se", "mse_mc_se", "rmse_mc_se", "se_mc_se"))
    assert np.isnan(table["overlap_rate"][0])
    assert table["overlap_rate"][2] == pytest.approx(0.75)


def ranks(values: FloatArray) -> np.ndarray:
  """Return each value's rank among distinct values, 0 for the lowest."""
  return np.argsort(np.argsort(values))


class TestDeciles:
  """Tests for the deciles of the true ability and the metrics by decile (ADR-0016)."""

  @pytest.mark.parametrize(
    ("n_examinees", "decile_of_rank"),
    [
      pytest.param(20, lambda rank: rank // 2, id="20"),
      pytest.param(100, lambda rank: rank // 10, id="100"),
      pytest.param(11, lambda rank: min(rank, 9), id="11"),
    ],
  )
  def test_distinct_abilities_fill_the_deciles_by_rank(
    self, n_examinees: int, decile_of_rank: Callable[[int], int]
  ) -> None:
    """Of 20 distinct abilities, the ranks 2k and 2k + 1 form decile k; of 11, ranks 0 to 8 are deciles 0 to 8.

    With 11 abilities, each cut but the last equals an ability, which belongs to the decile above it, and the highest
    two share decile 9. Fails if the cuts are numpy.linspace's tenths, which at 11 examinees leave deciles 3 and 7
    empty, as catR's seq(0, 1, 0.1) does; if a decile's interval is closed on the right instead of the left, so an
    ability equal to a cut falls a decile lower; or if the highest ability falls outside decile 9.
    """
    true_thetas = np.random.default_rng(n_examinees).permutation(n_examinees) * 0.37 - 2.0

    deciles = result_of([finished()] * n_examinees, true_thetas=true_thetas).examinees()["decile"]

    assert deciles.tolist() == [decile_of_rank(rank) for rank in ranks(true_thetas)]

  def test_tied_abilities_share_a_decile_and_can_leave_deciles_empty(self) -> None:
    """Abilities 0, 0, 0, 0, 0, 1, 2, 3, 4 and 5: the five zeros all fall in decile 4, and deciles 0 to 3 are empty.

    The cuts at 0.1 to 0.4 are interpolated between zeros, so they are 0, and the cut at 0.5 lies between 0 and 1, at
    0.5; the cuts at 0.6 to 0.9 are 1.4, 2.3, 3.2 and 4.1. Fails if tied abilities are split between deciles by their
    order, as grouping by rank would split them, or if the zeros fall in the lowest decile whose cut is 0.
    """
    result = result_of([finished()] * 10, true_thetas=[0, 0, 0, 0, 0, 1, 2, 3, 4, 5])

    assert result.examinees()["decile"].tolist() == [4, 4, 4, 4, 4, 5, 6, 7, 8, 9]
    assert result.deciles()["n_sessions"].tolist() == [0, 0, 0, 0, 5, 1, 1, 1, 1, 1]

  def test_an_empty_decile_gives_nan_for_every_metric(self) -> None:
    """Three examinees of one ability are all in decile 9, so decile 0 holds no session and every metric of it is NaN.

    Fails if an empty decile averages an empty array (NaN with a warning, which the test settings turn into an error),
    or reports 0 as if it had been measured.
    """
    result = result_of([finished((0, 1), theta=0.2)] * 3, true_thetas=[1.0, 1.0, 1.0])

    scalars = [
      result.bias(decile=0),
      result.mse(decile=0),
      result.rmse(decile=0),
      result.test_length(decile=0),
      result.se(decile=0),
      result.overlap_rate(decile=0),
    ]
    table = result.deciles()

    assert all(metric is not None and np.isnan(metric).all() for metric in scalars)
    assert np.isnan(result.exposure_rates(decile=0)).all()
    assert table["n_sessions"].tolist() == [0] * 9 + [3]
    assert all(np.isnan(column[:9]).all() for name, column in table.items() if name not in {"decile", "n_sessions"})
    assert table["bias"][9] == pytest.approx(-0.8)

  def test_the_deciles_table_of_a_hand_worked_run(self) -> None:
    """Twenty examinees of abilities 0 to 19, two per decile, with each decile's metrics worked out by hand.

    In decile k, the examinee of ability 2k takes one item and ends at an error of 0.1k + 0.05 with a standard error of
    0.5, and the one of 2k + 1 takes three and ends at 0.1k - 0.05 with 0.7. So the bias is 0.1k with an error of 0.05;
    the MSE 0.01k^2 + 0.0025, whose squared errors differ by 0.02k, with an error of 0.01k; the length 2 with an error
    of 1; and the standard error 0.6 with an error of 0.1. The tests differ in length, so there is no overlap rate.
    Fails if a row covers another decile's sessions, or a metric or its error is put in another column or row.
    """
    sessions = [
      finished((k,), theta=2 * k + 0.1 * k + 0.05, se=0.5)
      if examinee % 2 == 0
      else finished((k, 10, 11), theta=2 * k + 1 + 0.1 * k - 0.05, se=0.7, stop_reason=StopReason.NO_ITEMS)
      for examinee in range(20)
      for k in [examinee // 2]
    ]
    k = np.arange(10)
    mse = 0.01 * k**2 + 0.0025

    table = result_of(sessions, n_items=12, true_thetas=np.arange(20.0)).deciles()

    assert table["decile"].tolist() == k.tolist()
    assert table["n_sessions"].tolist() == [2] * 10
    assert table["true_theta"] == pytest.approx(2 * k + 0.5)
    assert table["bias"] == pytest.approx(0.1 * k)
    assert table["bias_mc_se"] == pytest.approx([0.05] * 10)
    assert table["mse"] == pytest.approx(mse)
    assert table["mse_mc_se"] == pytest.approx(0.01 * k)
    assert table["rmse"] == pytest.approx(np.sqrt(mse))
    assert table["rmse_mc_se"] == pytest.approx(0.01 * k / (2 * np.sqrt(mse)))
    assert table["test_length"] == pytest.approx([2.0] * 10)
    assert table["test_length_mc_se"] == pytest.approx([1.0] * 10)
    assert table["se"] == pytest.approx([0.6] * 10)
    assert table["se_mc_se"] == pytest.approx([0.1] * 10)
    assert np.isnan(table["overlap_rate"]).all()
    assert np.isnan(table["overlap_rate_mc_se"]).all()

  def test_a_deciles_exposure_and_overlap_are_those_of_its_own_tests(self) -> None:
    """In decile k, the two examinees take items (k, 10) and (k, 11), so similar examinees share item k.

    Within decile 3, item 3 has rate 1 and items 10 and 11 rate 0.5, so the overlap is (1 + 0.25 + 0.25) / 2 = 0.75.
    Over the run, items 0 to 9 have rate 0.1 and items 10 and 11 rate 0.5, so it is (10 * 0.01 + 2 * 0.25) / 2 = 0.3.
    Every test's rates sum to the same value, so the errors are 0. Fails if a decile's metric is computed from the
    whole run's rates.
    """
    sessions = [finished((examinee // 2, 10 + examinee % 2)) for examinee in range(20)]
    result = result_of(sessions, n_items=12, true_thetas=np.arange(20.0))

    rates, errors = result.exposure_rates(decile=3)

    assert rates.tolist() == [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5]
    assert errors[[3, 10]] == pytest.approx([0.0, math.sqrt(0.25 / 2)])
    assert result.overlap_rate(decile=3) == pytest.approx((0.75, 0.0))
    assert result.overlap_rate() == pytest.approx((0.3, 0.0))

  @pytest.mark.parametrize(
    ("decile", "error", "message"),
    [
      pytest.param(10, ValueError, "decile is 10; the deciles are 0, of the lowest true abilities, to 9", id="ten"),
      pytest.param(-1, ValueError, "decile is -1; the deciles are 0", id="negative"),
      pytest.param(2.0, TypeError, r"decile is 2.0; it must be None or an integer from 0 to 9", id="float"),
      pytest.param(True, TypeError, "decile is True; it must be None or an integer", id="boolean"),
    ],
  )
  def test_a_decile_is_one_of_0_to_9(self, decile: object, error: type[Exception], message: str) -> None:
    """Fails if decile 10 gives an empty decile's NaN, as if it existed, or True reads as decile 1."""
    result = result_of(TWO_SESSIONS, true_thetas=TWO_TRUE_THETAS)

    with pytest.raises(error, match=message):
      result.bias(decile=decile)  # ty: ignore[invalid-argument-type]
