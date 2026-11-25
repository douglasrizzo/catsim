"""Tests for simulate, its generated and recorded examinees and their random streams, and its results."""

from __future__ import annotations

import dataclasses
import math
import re

import numpy as np
import pytest

from catsim import CatEngine, Estimate, ItemBank, Session, SimulationResult, StepContext, StopReason, simulate
from catsim.compat import override
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
    assert result.responses == expected

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

    assert first.test_lengths[:2].sum() != second.test_lengths[:2].sum()
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
    assert result.responses == [
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
    assert [sorted({item % 2 for item in items}) for items in result.administered_items] == [[1], [0]]

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

    assert result.responses == [(False,) * 6, (True,) * 6]
    low, high = result.final_thetas.tolist()
    assert result.bias == pytest.approx(((low - 3.0) + (high + 3.0)) / 2)

  @pytest.mark.parametrize("metric", ["bias", "mse", "rmse"])
  def test_the_metrics_of_examinees_with_unknown_abilities_raise(self, item_bank: ItemBank, metric: str) -> None:
    """Recorded examinees alone have no true abilities, so the metrics that need them raise and say how to give them.

    Fails if a metric compares the estimates with zeros or NaN instead.
    """
    result = simulate(make_engine(item_bank, MaxInfoSelector(), 5), responses=self.ROWS, rng=0)

    with pytest.raises(
      ValueError, match="need the examinees' true abilities, and this result has none; pass true_thetas"
    ):
      getattr(result, metric)


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
    assert other_seed.administered_items != first.administered_items

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
    assert result.administered_items == expected

  def test_two_runs_without_rng_give_other_tests(self, random_engine: CatEngine) -> None:
    """A run without rng takes fresh entropy, so two such runs give other final estimates.

    Each of the three sessions draws 10 of 50 items at random, so equal final estimates for all three are
    practically impossible. Fails if rng defaults to a fixed seed, which makes every unseeded run the same.
    """
    thetas = [-1.0, 0.0, 1.0]

    assert not np.array_equal(
      simulate(random_engine, true_thetas=thetas).final_thetas, simulate(random_engine, true_thetas=thetas).final_thetas
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
    assert simulate(random_engine, true_thetas=thetas, rng=five).administered_items != first.administered_items

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
    assert result.exposure_rates[:2].tolist() == [1.0, 1.0]

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
    assert accepting.proposal_counts is not None
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
    assert result.test_lengths.tolist() == [5] * 20
    assert result.proposal_counts is not None
    assert result.proposal_rates is not None
    assert np.array_equal(result.proposal_counts, result.exposure_counts + declines)
    assert np.allclose(result.proposal_rates, (result.exposure_counts + declines) / 20)


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
    assert result.sessions == sessions
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
      r"responses\[1, 4\] is 2; an integer response must be 0 or 1",
      id="integer-response-2",
    ),
    pytest.param(
      {"responses": np.where(np.arange(100).reshape(2, 50) == 7, 0.5, 1.0)},
      ValueError,
      r"responses\[0, 7\] is 0.5; a response must be 0 or 1, or NaN where it is missing",
      id="float-response-one-half",
    ),
    pytest.param(
      {"responses": np.where(np.arange(100).reshape(2, 50) == 60, np.inf, 0.0)},
      ValueError,
      r"responses\[1, 10\] is inf; a response must be 0 or 1, or NaN where it is missing",
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


def finished_session(items: tuple[int, ...]) -> Session:
  """Return a session that administered ``items``, all answered correctly, with a constant estimate."""
  return Session(
    administered_items=items,
    responses=(True,) * len(items),
    estimates=(Estimate(0.0, math.inf),) * (len(items) + 1),
  )


def result_of(sessions: list[Session], n_items: int = 5, true_thetas: list[float] | None = None) -> SimulationResult:
  """Return a result holding the given sessions, on a bank of n_items items no session's exposure is counted in."""
  return SimulationResult(
    item_bank=ItemBank.generate(n_items, rng=0),
    sessions=sessions,
    exposure_counts=np.zeros(n_items, dtype=int),
    duration=0.0,
    true_thetas=None if true_thetas is None else np.array(true_thetas),
  )


# Two hand-built sessions with known estimates and evaluation counts, and their examinees' true abilities. The first
# ends at 0.6 (SE 0.9) for a true 0.5 after two items, its estimates costing 7 and 5 evaluations; the second at -0.8
# (SE 1.1) for a true -1.0 after one item costing 4. The second's initial estimate has no count, as a hand-built one
# need not.
TWO_SESSIONS = [
  Session(
    administered_items=(3, 1),
    responses=(True, False),
    estimates=(Estimate(0.0, math.inf, 0), Estimate(0.4, 1.2, 7), Estimate(0.6, 0.9, 5)),
  ),
  Session(administered_items=(2,), responses=(False,), estimates=(Estimate(0.0, math.inf), Estimate(-0.8, 1.1, 4))),
]
TWO_TRUE_THETAS = [0.5, -1.0]


class TestSimulationResult:
  """Tests for SimulationResult derived properties."""

  def test_per_session_properties_read_each_session_in_order(self) -> None:
    """Each per-session property has one entry per session, in session order.

    Fails if final_thetas or final_ses read an earlier estimate (such as the initial one), if
    theta_histories drops the initial estimate, or if test_lengths counts the estimates (one more
    than the items). Fails too if final_thetas, final_ses or test_lengths is a list rather than an
    array (no ``.mean()``, no elementwise ``result.final_ses < 0.4``), or if test_lengths holds
    floats, which ``numpy.bincount`` rejects.
    """
    result = result_of(TWO_SESSIONS)

    assert result.final_thetas.tolist() == [0.6, -0.8]
    assert result.final_ses.tolist() == [0.9, 1.1]
    assert result.test_lengths.tolist() == [2, 1]
    assert np.bincount(result.test_lengths).tolist() == [0, 1, 1]
    assert result.theta_histories == [[0.0, 0.4, 0.6], [0.0, -0.8]]

  def test_evaluations_sum_each_sessions_counts_over_its_responses(self) -> None:
    """Session totals are 7 + 5 = 12 and 4; the mean per estimate is 16 / 3.

    Fails if the initial estimate takes part (the second session would be None for its missing
    count), or if mean_evaluations averages the session totals (8) instead of the estimates.
    """
    result = result_of(TWO_SESSIONS)

    assert result.evaluations == [12, 4]
    assert result.mean_evaluations == pytest.approx(16 / 3)

  def test_a_session_with_an_estimate_without_a_count_has_no_evaluation_total(self) -> None:
    """A session with an uncounted estimate has no total; the mean is over the estimates that have a count.

    The first session's estimates cost 3 and an unknown number, the second's 4, so the totals are
    [None, 4] and the mean is (3 + 4) / 2. Fails if an unknown count is read as 0 (a total of 3 and a
    mean of 7 / 3).
    """
    sessions = [
      Session(
        administered_items=(3, 1),
        responses=(True, False),
        estimates=(Estimate(0.0, math.inf, 0), Estimate(0.4, 1.2, 3), Estimate(0.6, 0.9)),
      ),
      Session(
        administered_items=(2,), responses=(False,), estimates=(Estimate(0.0, math.inf, 0), Estimate(-0.8, 1.1, 4))
      ),
    ]
    result = result_of(sessions)

    assert result.evaluations == [None, 4]
    assert result.mean_evaluations == pytest.approx(3.5)

  def test_mean_evaluations_is_none_without_a_counted_estimate(self) -> None:
    """A run whose sessions have no response has no mean count; fails if it reports 0 or NaN instead."""
    result = result_of([finished_session(()), finished_session(())])

    assert result.evaluations == [0, 0]
    assert result.mean_evaluations is None

  def test_exposure_rates_sum_to_the_test_length(self, simulation_result: SimulationResult) -> None:
    """With one rate per item, and every examinee given 10 items, the rates sum to 10."""
    assert simulation_result.exposure_rates.shape == (simulation_result.item_bank.n_items,)
    assert simulation_result.exposure_rates.sum() == pytest.approx(10.0)

  def test_metrics_compare_final_estimates_with_true_abilities(self) -> None:
    """Errors 0.6 - 0.5 and -0.8 + 1.0 give bias 0.15, MSE (0.01 + 0.04) / 2 = 0.025 and RMSE its root.

    Fails if a metric compares another estimate than the final one, or swaps the sign of the bias.
    """
    result = result_of(TWO_SESSIONS, true_thetas=TWO_TRUE_THETAS)

    assert result.bias == pytest.approx(0.15)
    assert result.mse == pytest.approx(0.025)
    assert result.rmse == pytest.approx(math.sqrt(0.025))

  @pytest.mark.parametrize("metric", ["bias", "mse", "rmse"])
  def test_metrics_of_a_result_without_true_thetas_raise(self, metric: str) -> None:
    """A result built without true abilities cannot give its accuracy metrics.

    Fails if a metric compares the estimates with zeros or NaN instead of saying that the true abilities are missing.
    """
    with pytest.raises(ValueError, match="need the examinees' true abilities, and this result has none"):
      getattr(result_of(TWO_SESSIONS), metric)

  def test_a_result_rejects_true_thetas_that_do_not_match_its_sessions(self) -> None:
    """One true ability per session: a list for another run would pair estimates with the wrong examinees."""
    message = "true_thetas has shape (3,); there must be one true ability per session (2)"
    with pytest.raises(ValueError, match=re.escape(message)):
      result_of(TWO_SESSIONS, true_thetas=[0.5, -1.0, 0.2])

  def test_a_result_rejects_a_true_ability_that_is_not_finite(self) -> None:
    """The true abilities are known for every session or for none (ADR-0016), so NaN does not mark an unknown one.

    Fails if a result accepts a NaN true ability, whose bias, MSE and RMSE would then be NaN.
    """
    with pytest.raises(
      ValueError, match=r"true_thetas\[1\] is nan; true abilities must be finite, and true_thetas None"
    ):
      result_of(TWO_SESSIONS, true_thetas=[0.5, math.nan])


def test_simulation_result_empty_sessions_exposure_and_overlap(item_bank) -> None:
  """A result with no sessions has zero exposure and proposal rates and no overlap rate.

  Fails if exposure_rates or proposal_rates divides the counts by zero sessions (NaN, with a warning that the test
  settings turn into an error), or if overlap_rate computes a number for a run without tests.
  """
  result = SimulationResult(
    item_bank=item_bank,
    sessions=[],
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.0,
    proposal_counts=np.zeros(item_bank.n_items, dtype=int),
  )

  assert result.proposal_rates is not None
  assert np.array_equal(result.exposure_rates, np.zeros(item_bank.n_items))
  assert np.array_equal(result.proposal_rates, np.zeros(item_bank.n_items))
  assert result.overlap_rate is None


@pytest.mark.parametrize("metric", ["bias", "mse", "rmse"])
def test_metrics_of_a_result_without_sessions_raise(metric: str) -> None:
  """With no session, the metrics averaged an empty array: NaN and a RuntimeWarning instead of a clear error."""
  with pytest.raises(ValueError, match="need at least one session, and the result has none"):
    getattr(result_of([]), metric)


def test_simulation_result_overlap_rate_for_a_hand_worked_run() -> None:
  """Two 2-item tests, [0, 1] and [0, 2], on a 5-item bank give exposure rates [1, 0.5, 0.5, 0, 0].

  The rates average Q / N, so (N / Q) var(r) + Q / N reduces to sum(r^2) / Q = 1.5 / 2 = 0.75. A wrong test
  length of 1 would give 0.9.
  """
  sessions = [finished_session((0, 1)), finished_session((0, 2))]
  result = SimulationResult(
    item_bank=ItemBank.generate(5, rng=0),
    sessions=sessions,
    exposure_counts=np.array([2, 1, 1, 0, 0]),
    duration=0.0,
  )

  assert result.exposure_rates.tolist() == [1.0, 0.5, 0.5, 0.0, 0.0]
  assert result.overlap_rate == pytest.approx(0.75)


def test_simulation_result_overlap_requires_uniform_non_zero_test_lengths() -> None:
  """Overlap is None for sessions of different lengths or of no items.

  Fails if overlap_rate computes a number from a test length the sessions do not share, or divides
  by a test length of zero.
  """
  heterogeneous = result_of([finished_session((0,)), finished_session((0, 1))])
  zero_length = result_of([finished_session(()), finished_session(())])

  assert heterogeneous.overlap_rate is None
  assert zero_length.overlap_rate is None
