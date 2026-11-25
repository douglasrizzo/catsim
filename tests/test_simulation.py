"""Tests for the simulation runner, its simulated responses and its results."""

from __future__ import annotations

import math
import re
import subprocess
import sys

import numpy as np
import pytest

from catsim import (
  CatEngine,
  Estimate,
  ItemBank,
  Session,
  SimulatedResponseProvider,
  SimulationResult,
  SimulationRunner,
  StepContext,
  StopReason,
)
from catsim.compat import override
from catsim.estimation import NumericalSearchEstimator
from catsim.exposure import BaseExposureControl
from catsim.initialization import FixedPointInitializer
from catsim.selection import BaseSelector, RandomesqueSelector, RandomSelector
from catsim.stopping import MaxLengthStopper
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


class RngRecordingSelector(BaseSelector):
  """Select the lowest-index remaining item and record the generator each call was given."""

  def __init__(self) -> None:
    super().__init__()
    self.seen_rngs: list[np.random.Generator] = []

  @override
  def select(self, context: StepContext) -> int:
    self.seen_rngs.append(context.rng)
    return next(i for i in range(context.item_bank.n_items) if i not in context.administered_items)


class AcceptingControl(BaseExposureControl):
  """Exposure control that accepts every proposal without drawing a random number."""

  @override
  def accept(self, context: StepContext, item: int) -> bool:
    del context, item
    return True


class CoinControl(BaseExposureControl):
  """Exposure control that accepts each proposal with probability one half, drawn from the run's generator."""

  @override
  def accept(self, context: StepContext, item: int) -> bool:
    del item
    return bool(context.rng.random() < 0.5)


def make_runner(
  item_bank: ItemBank, selector: BaseSelector, max_items: int, exposure_control: BaseExposureControl | None = None
) -> SimulationRunner:
  """Return a runner whose sessions start at 0, use the given selector and control and stop after max_items items."""
  return SimulationRunner(
    CatEngine(
      item_bank,
      FixedPointInitializer(0.0),
      selector,
      NumericalSearchEstimator(),
      MaxLengthStopper(max_items),
      exposure_control=exposure_control,
    )
  )


class TestSimulationRunnerRun:
  """Tests for SimulationRunner.run()."""

  def test_run_keeps_the_true_abilities_in_session_order(self, simulation_runner: SimulationRunner) -> None:
    """Session i simulates true_thetas[i]; fails if the result reorders, rounds or drops the true abilities."""
    thetas = [0.25, -1.5, 1.0]
    result = simulation_runner.run(thetas)

    assert result.true_thetas is not None
    assert result.true_thetas.tolist() == thetas
    assert len(result.sessions) == 3

  @pytest.mark.parametrize("count", [5, np.int64(5)], ids=["int", "numpy_int"])
  def test_run_rejects_a_number_of_examinees(self, simulation_runner: SimulationRunner, count: int) -> None:
    """run takes the true abilities, not how many examinees to draw.

    Fails if a single number is accepted, either as a count of examinees to draw or as the one
    true ability of a one-examinee run.
    """
    with pytest.raises(TypeError, match=r"one true ability per examinee, not a single number .*rng\.normal\(0, 1, n\)"):
      simulation_runner.run(count)

  def test_run_with_empty_examinee_array_raises(self, simulation_runner: SimulationRunner) -> None:
    """A run with no examinee fails rather than returning an empty result."""
    with pytest.raises(ValueError, match="true_thetas cannot be empty"):
      simulation_runner.run([])

  def test_run_with_multidimensional_examinees_raises(self, simulation_runner: SimulationRunner) -> None:
    """A two-dimensional array of abilities fails rather than being read as one examinee per row."""
    with pytest.raises(
      TypeError, match=r"true_thetas must be one-dimensional, one true ability per examinee, got shape \(1, 2\)"
    ):
      simulation_runner.run(np.array([[0.0, 1.0]]))

  @pytest.mark.parametrize("bad", [np.nan, np.inf], ids=["nan", "inf"])
  def test_run_rejects_a_true_ability_that_is_not_finite(self, simulation_runner: SimulationRunner, bad: float) -> None:
    """A NaN or infinite true ability fails before any session runs.

    Fails if the runner simulates it: the response probability at NaN is NaN, so every simulated
    response would silently be wrong.
    """
    with pytest.raises(ValueError, match="true_thetas must be finite numbers"):
      simulation_runner.run([0.0, bad])

  def test_two_runs_with_the_same_seed_give_the_same_sessions(self, simulation_runner: SimulationRunner) -> None:
    """Every run builds its generator from its seed, so the same seed gives the same tests on the same runner.

    Fails if the runner keeps one generator across runs, so the second run continues the first
    one's stream, or ignores the seed (a different seed would then give the same tests).
    """
    thetas = [-1.0, 0.0, 1.0]
    first = simulation_runner.run(thetas, rng=3)
    second = simulation_runner.run(thetas, rng=3)
    other_seed = simulation_runner.run(thetas, rng=4)

    assert second.administered_items == first.administered_items
    assert second.responses == first.responses
    assert second.theta_histories == first.theta_histories
    assert np.array_equal(second.exposure_counts, first.exposure_counts)
    assert other_seed.administered_items != first.administered_items

  def test_a_seeded_run_without_a_control_gives_the_tests_it_always_gave(self, item_bank: ItemBank) -> None:
    """A seeded run of random selection without an exposure control gives fixed, pinned items and responses.

    Fails if the engine draws from the run's generator when there is no exposure control, or passes the selector
    another set of items to choose from: every later draw, and so every item and response, would move.
    """
    result = make_runner(item_bank, RandomSelector(), max_items=5).run([-1.0, 0.0, 1.0], rng=7)

    assert result.administered_items == [(47, 30, 41, 10, 44), (0, 6, 24, 36, 13), (29, 27, 35, 31, 42)]
    assert result.responses == [
      (False, False, True, False, False),
      (False, True, True, True, False),
      (False, True, False, True, True),
    ]

  def test_a_control_that_accepts_without_drawing_reproduces_the_run_without_one(self, item_bank: ItemBank) -> None:
    """With a control that accepts every proposal and draws nothing, a seeded run gives the run without a control.

    Fails if asking the control changes what the selector is given or draws from the generator, or if accepted
    proposals are counted twice (the proposal counts would exceed the exposure counts).
    """
    thetas = [-1.0, 0.0, 1.0]
    without = make_runner(item_bank, RandomesqueSelector(bin_size=3), max_items=5).run(thetas, rng=7)
    accepting = make_runner(item_bank, RandomesqueSelector(bin_size=3), 5, AcceptingControl()).run(thetas, rng=7)

    assert accepting.administered_items == without.administered_items
    assert accepting.responses == without.responses
    assert accepting.theta_histories == without.theta_histories
    assert [session.excluded_items for session in accepting.sessions] == [(), (), ()]
    assert accepting.proposal_counts is not None
    assert np.array_equal(accepting.proposal_counts, accepting.exposure_counts)

  def test_a_runs_proposals_are_its_administrations_plus_its_declines(self, item_bank: ItemBank) -> None:
    """Under a control that declines at random, each item's proposals are its administrations plus its declines.

    The declines are read from the sessions' excluded items. Fails if the runner reports the tracker's counts as the
    proposals, if a decline is not recorded in the tracker or is recorded as an administration, or if proposal_rates
    divides by anything but the number of sessions.
    """
    result = make_runner(item_bank, RandomSelector(), 5, CoinControl()).run(np.zeros(20), rng=3)

    declines = np.bincount(
      [item for session in result.sessions for item in session.excluded_items], minlength=item_bank.n_items
    )
    assert int(declines.sum()) > 0
    assert result.test_lengths.tolist() == [5] * 20
    assert result.proposal_counts is not None
    assert result.proposal_rates is not None
    assert np.array_equal(result.proposal_counts, result.exposure_counts + declines)
    assert np.allclose(result.proposal_rates, (result.exposure_counts + declines) / 20)

  def test_two_runs_without_rng_give_other_tests(self, simulation_runner: SimulationRunner) -> None:
    """A run without rng draws fresh randomness, so two such runs give other final estimates.

    Each of the three sessions draws 10 of 50 items at random, so equal final estimates for all three are
    practically impossible. Fails if rng defaults to a fixed seed, which makes every unseeded run the same.
    """
    thetas = [-1.0, 0.0, 1.0]

    assert not np.array_equal(simulation_runner.run(thetas).final_thetas, simulation_runner.run(thetas).final_thetas)

  def test_a_generator_passed_as_rng_is_the_one_every_session_draws_from(self, item_bank: ItemBank) -> None:
    """A Generator passed as rng is used as is, and every session of the run gets that same one.

    Fails if the runner reseeds or copies the generator, or gives each session a generator of its own.
    """
    selector = RngRecordingSelector()
    rng = np.random.default_rng(5)

    make_runner(item_bank, selector, max_items=2).run([0.0, 1.0], rng=rng)

    assert len(selector.seen_rngs) == 4
    assert all(seen is rng for seen in selector.seen_rngs)

  def test_a_run_with_a_progress_bar_matches_a_quiet_run(self, simulation_runner: SimulationRunner) -> None:
    """progress=True only adds a progress bar: fails if it changes the sessions or the run does not complete."""
    thetas = [-0.5, 0.0, 0.5]
    quiet = simulation_runner.run(thetas, rng=0, progress=False)
    shown = simulation_runner.run(thetas, rng=0, progress=True)

    assert shown.administered_items == quiet.administered_items
    assert shown.theta_histories == quiet.theta_histories

  def test_importing_the_simulation_module_does_not_import_tqdm(self) -> None:
    """tqdm is imported only for a run that shows a progress bar; fails if catsim imports it at module level."""
    code = "import sys, catsim, catsim.simulation; print('tqdm' in sys.modules)"
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)

    assert completed.stdout.strip() == "False"

  def test_run_populates_sessions(self, simulation_result: SimulationResult) -> None:
    """Every session of a run is run to its stopper: 10 items, 10 responses and 11 estimates.

    Fails if run returns sessions before they end, or loses part of a session's history.
    """
    assert len(simulation_result.sessions) == 5
    for session in simulation_result.sessions:
      assert session.stop_reason is StopReason.STOPPER
      assert (len(session.administered_items), len(session.responses), len(session.theta_history)) == (10, 10, 11)

  def test_each_session_sees_the_exposure_of_the_sessions_before_it(self, item_bank: ItemBank) -> None:
    """The runner counts exposure across its sessions, so each sees catR's rates of the examinees before it.

    Every session administers items 0 and 1, so from the second session on both have gone to every earlier
    examinee: rate 1 at both of the session's selections, and 0 for every other item. The first session sees no
    rates (ADR-0012). Fails if each session counts afresh (rates stay 0), or if the rates divide by the planned
    number of examinees (0.25, 0.5, 0.75) or count the session in progress (0.5, 0.67, 0.75).
    """
    selector = FirstRemainingSelector()

    result = make_runner(item_bank, selector, max_items=2).run(np.zeros(4))

    first_session, later_sessions = selector.seen_exposure_rates[:2], selector.seen_exposure_rates[2:]
    assert first_session == [None, None]
    expected = np.zeros(item_bank.n_items)
    expected[:2] = 1.0
    assert [None if rates is None else rates.tolist() for rates in later_sessions] == [expected.tolist()] * 6
    assert result.exposure_rates[:2].tolist() == [1.0, 1.0]


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
  def test_metrics_need_every_true_theta(self, metric: str) -> None:
    """Bias, MSE and RMSE raise for a result with a session whose true theta is NaN, naming how many lack one.

    Fails if a metric silently compares the estimate with a stand-in such as 0 (bias 0.5 for estimates [1, 1] against
    true [unknown, 1]) or ignores the session.
    """
    result = result_of(TWO_SESSIONS, true_thetas=[0.5, math.nan])

    with pytest.raises(ValueError, match="need every session's true theta, but 1 of 2 sessions have none"):
      getattr(result, metric)

  @pytest.mark.parametrize("metric", ["bias", "mse", "rmse"])
  def test_metrics_of_a_result_without_true_thetas_raise(self, metric: str) -> None:
    """A result of recorded examinees has no true abilities, so its accuracy metrics cannot be computed.

    Fails if a metric compares the estimates with zeros or NaN instead of saying that the true abilities are missing.
    """
    with pytest.raises(ValueError, match="need the sessions' true abilities, and the result has none"):
      getattr(result_of(TWO_SESSIONS), metric)

  def test_a_result_rejects_true_thetas_that_do_not_match_its_sessions(self) -> None:
    """One true ability per session: a list for another run would pair estimates with the wrong examinees."""
    message = "true_thetas has shape (3,); there must be one true ability per session (2)"
    with pytest.raises(ValueError, match=re.escape(message)):
      result_of(TWO_SESSIONS, true_thetas=[0.5, -1.0, 0.2])


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


class FakeRng(np.random.Generator):
  """Generator whose uniform draw is always the given sample, for deterministic response-provider tests."""

  def __init__(self, sample: float) -> None:
    super().__init__(np.random.PCG64(0))
    self.sample = sample

  # Unannotated: Generator.uniform's overloads return an array for a size, which this stub never takes.
  @override
  def uniform(self, low=0.0, high=1.0, size=None):
    """Return the fixed sample."""
    del low, high, size
    return self.sample


@pytest.mark.parametrize(
  ("item", "draw", "expected"),
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
  item: int, draw: float, expected: bool
) -> None:
  """The response is correct when the uniform draw falls below the chosen item's 4PL probability at the true theta.

  Together the cases fail if the response used the estimate instead of the true theta (1.5), another item,
  or dropped the item's guessing or upper-asymptote parameter.
  """
  item_bank = ItemBank(
    np.array([[1.0, 5.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.2, 0.9], [1.0, 5.0, 0.3, 1.0]])
  )
  session = Session(estimates=(Estimate(-1.5, math.inf),))
  rng = FakeRng(sample=draw)

  assert SimulatedResponseProvider(1.5).answer(item_bank, item, session, rng) is expected
