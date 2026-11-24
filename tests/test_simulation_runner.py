"""Tests for the simulation runner, its simulated responses and its results."""

from __future__ import annotations

import math
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
import pytest

from catsim import Estimate, StepContext
from catsim.engine import CatEngine
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import BaseSelector
from catsim.simulation import SimulatedResponseProvider, SimulationResult, SimulationRunner
from catsim.state import CatSessionState, StopReason
from catsim.stopping import MaxLengthStopper


class FirstRemainingSelector(BaseSelector):
  """Select the lowest-index remaining item and record the exposure rates shown at each call."""

  def __init__(self) -> None:
    super().__init__()
    self.seen_exposure_rates: list[np.ndarray] = []

  def select(self, context: StepContext) -> int:
    assert context.exposure_rates is not None, "SimulationRunner should always pass exposure rates"
    self.seen_exposure_rates.append(context.exposure_rates.copy())
    return next(i for i in range(context.item_bank.n_items) if i not in context.administered_items)


class RngRecordingSelector(BaseSelector):
  """Select the lowest-index remaining item and record the generator each call was given."""

  def __init__(self) -> None:
    super().__init__()
    self.seen_rngs: list[np.random.Generator] = []

  def select(self, context: StepContext) -> int:
    self.seen_rngs.append(context.rng)
    return next(i for i in range(context.item_bank.n_items) if i not in context.administered_items)


def make_runner(item_bank: ItemBank, selector: BaseSelector, max_items: int) -> SimulationRunner:
  """Return a runner whose sessions start at 0, use the given selector and stop after max_items items."""
  return SimulationRunner(
    CatEngine(item_bank, FixedPointInitializer(0.0), selector, NumericalSearchEstimator(), MaxLengthStopper(max_items))
  )


class TestSimulationRunnerRun:
  """Tests for SimulationRunner.run()."""

  def test_run_keeps_the_true_abilities_in_session_order(self, simulation_runner: SimulationRunner) -> None:
    """Session i simulates true_thetas[i]; fails if the runner reorders, rounds or replaces the true abilities."""
    thetas = [0.25, -1.5, 1.0]
    result = simulation_runner.run(thetas)

    assert [session.true_theta for session in result.sessions] == thetas
    assert [session.session_id for session in result.sessions] == [0, 1, 2]

  @pytest.mark.parametrize("count", [5, np.int64(5)], ids=["int", "numpy_int"])
  def test_run_rejects_a_number_of_examinees(self, simulation_runner: SimulationRunner, count: int) -> None:
    """run takes the true abilities, not how many examinees to draw.

    Fails if a single number is accepted, either as a count of examinees drawn from the bank, as
    run(int) did, or as the one true ability of a one-examinee run.
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
    assert second.response_vectors == first.response_vectors
    assert second.theta_histories == first.theta_histories
    assert np.array_equal(second.exposure_counts, first.exposure_counts)
    assert other_seed.administered_items != first.administered_items

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
    quiet, shown = simulation_runner.run(thetas, progress=False), simulation_runner.run(thetas, progress=True)

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
      assert (len(session.administered_item_ids), len(session.responses), len(session.theta_history)) == (10, 10, 11)

  def test_each_session_sees_the_exposure_of_the_sessions_before_it(self, item_bank: ItemBank) -> None:
    """The runner shares one tracker across its sessions, so each sees the exposure of the examinees before it.

    Every session administers items 0 and 1, so from the second session on both have gone to every earlier
    examinee: rate 1 whenever the session can still be offered them. Fails if each session gets a fresh tracker
    (rates stay 0), or if the rates divide by the planned number of examinees (0.25, 0.5, 0.75) or count the
    session in progress (0.5, 0.67, 0.75).
    """
    selector = FirstRemainingSelector()

    result = make_runner(item_bank, selector, max_items=2).run(np.zeros(4))

    # The selector runs twice per session; at its second run the session has given item 0, which it is not
    # offered again, so only item 1's rate counts there.
    first_runs, second_runs = selector.seen_exposure_rates[0::2], selector.seen_exposure_rates[1::2]
    assert [rates[:2].tolist() for rates in first_runs] == [[0.0, 0.0]] + [[1.0, 1.0]] * 3
    assert [float(rates[1]) for rates in second_runs] == [0.0, 1.0, 1.0, 1.0]
    assert not any(rates[2:].any() for rates in selector.seen_exposure_rates)
    assert result.exposure_rates[:2].tolist() == [1.0, 1.0]


def finished_session(session_id: int, items: list[int]) -> CatSessionState:
  """Return a session that administered ``items``, all answered correctly, with a constant estimate."""
  return CatSessionState(
    session_id=session_id,
    estimates=[Estimate(0.0, math.inf)] * (len(items) + 1),
    administered_item_ids=items,
    responses=[True] * len(items),
  )


def result_of(sessions: list[CatSessionState], n_items: int = 5) -> SimulationResult:
  """Return a result holding the given sessions, on a bank of n_items items no session's exposure is counted in."""
  return SimulationResult(
    item_bank=ItemBank.generate_item_bank(n_items, seed=0),
    sessions=sessions,
    exposure_counts=np.zeros(n_items, dtype=int),
    duration=0.0,
  )


def two_finished_sessions() -> list[CatSessionState]:
  """Return two hand-built sessions with known true abilities, estimates and evaluation counts.

  The first ends at 0.6 (SE 0.9) for a true 0.5 after two items, its estimates costing 7 and 5
  evaluations; the second at -0.8 (SE 1.1) for a true -1.0 after one item costing 4. The second's
  initial estimate has no count, as a hand-built one need not.
  """
  return [
    CatSessionState(
      session_id=0,
      estimates=[Estimate(0.0, math.inf, 0), Estimate(0.4, 1.2, 7), Estimate(0.6, 0.9, 5)],
      true_theta=0.5,
      administered_item_ids=[3, 1],
      responses=[True, False],
    ),
    CatSessionState(
      session_id=1,
      estimates=[Estimate(0.0, math.inf), Estimate(-0.8, 1.1, 4)],
      true_theta=-1.0,
      administered_item_ids=[2],
      responses=[False],
    ),
  ]


class TestSimulationResult:
  """Tests for SimulationResult derived properties."""

  def test_per_session_properties_read_each_session_in_order(self) -> None:
    """Each per-session property has one entry per session, in session order.

    Fails if final_thetas or final_ses read an earlier estimate (such as the initial one), if
    theta_histories drops the initial estimate, or if test_lengths counts the estimates (one more
    than the items).
    """
    result = result_of(two_finished_sessions())

    assert result.final_thetas == [0.6, -0.8]
    assert result.final_ses == [0.9, 1.1]
    assert result.test_lengths == [2, 1]
    assert result.theta_histories == [[0.0, 0.4, 0.6], [0.0, -0.8]]
    assert result.true_thetas.tolist() == [0.5, -1.0]

  def test_evaluations_sum_each_sessions_counts_over_its_responses(self) -> None:
    """Session totals are 7 + 5 = 12 and 4; the mean per estimate is 16 / 3.

    Fails if the initial estimate takes part (the second session would be None for its missing
    count), or if mean_evaluations averages the session totals (8) instead of the estimates.
    """
    result = result_of(two_finished_sessions())

    assert result.evaluations == [12, 4]
    assert result.mean_evaluations == pytest.approx(16 / 3)

  def test_a_session_with_an_estimate_without_a_count_has_no_evaluation_total(self) -> None:
    """A session with an uncounted estimate has no total; the mean is over the estimates that have a count.

    The first session's estimates cost 3 and an unknown number, the second's 4, so the totals are
    [None, 4] and the mean is (3 + 4) / 2. Fails if an unknown count is read as 0 (a total of 3 and a
    mean of 7 / 3).
    """
    sessions = [
      CatSessionState(
        session_id=0,
        estimates=[Estimate(0.0, math.inf, 0), Estimate(0.4, 1.2, 3), Estimate(0.6, 0.9)],
        administered_item_ids=[3, 1],
        responses=[True, False],
      ),
      CatSessionState(
        session_id=1,
        estimates=[Estimate(0.0, math.inf, 0), Estimate(-0.8, 1.1, 4)],
        administered_item_ids=[2],
        responses=[False],
      ),
    ]
    result = result_of(sessions)

    assert result.evaluations == [None, 4]
    assert result.mean_evaluations == pytest.approx(3.5)

  def test_mean_evaluations_is_none_without_a_counted_estimate(self) -> None:
    """A run whose sessions have no response has no mean count; fails if it reports 0 or NaN instead."""
    result = result_of([finished_session(0, []), finished_session(1, [])])

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
    result = result_of(two_finished_sessions())

    assert result.bias == pytest.approx(0.15)
    assert result.mse == pytest.approx(0.025)
    assert result.rmse == pytest.approx(math.sqrt(0.025))

  def test_true_thetas_are_nan_for_a_session_without_one(self) -> None:
    """A session with no true theta, such as one run by hand, has NaN in true_thetas.

    Fails if the result substitutes a number (0.0, as it once did) or drops the session, so the
    true abilities no longer line up with the sessions.
    """
    sessions = two_finished_sessions()
    sessions[0].true_theta = None
    true_thetas = result_of(sessions).true_thetas

    assert true_thetas.shape == (2,)
    assert np.isnan(true_thetas[0])
    assert true_thetas[1] == -1.0

  @pytest.mark.parametrize("metric", ["bias", "mse", "rmse"])
  def test_metrics_need_every_true_theta(self, metric: str) -> None:
    """Bias, MSE and RMSE raise for a result with a session without a true theta, naming how many lack one.

    Fails if a metric silently compares the estimate with a stand-in such as 0, as it once did
    (bias 0.5 for estimates [1, 1] against true [None, 1]), or ignores the session.
    """
    sessions = two_finished_sessions()
    sessions[1].true_theta = None

    with pytest.raises(ValueError, match="need every session's true theta, but 1 of 2 sessions have none"):
      getattr(result_of(sessions), metric)


def test_simulation_result_empty_sessions_exposure_and_overlap(item_bank) -> None:
  """A result with no sessions has zero exposure rates and no overlap rate.

  Fails if exposure_rates divides the counts by zero sessions (NaN, with a warning that the test settings turn into
  an error), or if overlap_rate computes a number for a run without tests.
  """
  result = SimulationResult(
    item_bank=item_bank,
    sessions=[],
    exposure_counts=np.zeros(item_bank.n_items, dtype=int),
    duration=0.0,
  )

  assert np.array_equal(result.exposure_rates, np.zeros(item_bank.n_items))
  assert result.overlap_rate is None


def test_simulation_result_overlap_rate_for_a_hand_worked_run() -> None:
  """Two 2-item tests, [0, 1] and [0, 2], on a 5-item bank give exposure rates [1, 0.5, 0.5, 0, 0].

  The rates average Q / N, so (N / Q) var(r) + Q / N reduces to sum(r^2) / Q = 1.5 / 2 = 0.75. A wrong test
  length of 1 would give 0.9.
  """
  sessions = [
    finished_session(0, [0, 1]),
    finished_session(1, [0, 2]),
  ]
  result = SimulationResult(
    item_bank=ItemBank.generate_item_bank(5, seed=0),
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
  heterogeneous = result_of([finished_session(0, [0]), finished_session(1, [0, 1])])
  zero_length = result_of([finished_session(0, []), finished_session(1, [])])

  assert heterogeneous.overlap_rate is None
  assert zero_length.overlap_rate is None


@dataclass
class FakeRng:
  """Minimal RNG stub for deterministic response-provider tests."""

  sample: float

  def uniform(self) -> float:
    return self.sample


def test_simulated_response_provider_requires_true_theta(item_bank) -> None:
  """Simulated responses require a true theta on the session."""
  session = CatSessionState(session_id=0, estimates=[Estimate(0.0, math.inf)]).snapshot()

  with pytest.raises(ValueError, match="true_theta is required") as exc_info:
    SimulatedResponseProvider().answer(item_bank, 0, session, np.random.default_rng(42))

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
  session = CatSessionState(session_id=0, estimates=[Estimate(-1.5, math.inf)], true_theta=1.5).snapshot()
  rng = FakeRng(sample=draw)

  assert SimulatedResponseProvider().answer(item_bank, item_id, session, rng) is expected  # type: ignore[arg-type]
