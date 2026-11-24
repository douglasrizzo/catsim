"""Tests for the refactored simulation architecture."""

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
from catsim.simulation import SimulatedResponseProvider, SimulationRunner
from catsim.state import CatSessionState, SimulationResult, StopReason
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
    assert second.estimations == first.estimations
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
    assert shown.latest_estimations == quiet.latest_estimations

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


class TestSimulationResult:
  """Tests for SimulationResult derived properties."""

  def test_latest_estimations_matches_session_thetas(self, simulation_result: SimulationResult) -> None:
    """latest_estimations holds the last estimate of each session's history.

    Fails if it reports an earlier estimate, such as the initial one.
    """
    assert simulation_result.latest_estimations == [session.theta_history[-1] for session in simulation_result.sessions]

  def test_estimations_returns_theta_histories(self, simulation_result: SimulationResult) -> None:
    """Estimations should expose one theta-history per session."""
    assert len(simulation_result.estimations) == len(simulation_result.sessions)
    assert all(len(history) >= 2 for history in simulation_result.estimations)

  def test_response_vectors_have_expected_length(self, simulation_result: SimulationResult) -> None:
    """Response vectors should match administered item counts."""
    for responses, items in zip(simulation_result.response_vectors, simulation_result.administered_items, strict=False):
      assert len(responses) == len(items)

  def test_exposure_rates_sum_to_the_test_length(self, simulation_result: SimulationResult) -> None:
    """With one rate per item, and every examinee given 10 items, the rates sum to 10."""
    assert simulation_result.exposure_rates.shape == (simulation_result.item_bank.n_items,)
    assert simulation_result.exposure_rates.sum() == pytest.approx(10.0)

  def test_metrics_compare_final_estimates_with_true_abilities(self, simulation_result: SimulationResult) -> None:
    """Bias is the mean of estimate minus true ability, MSE the mean squared error, RMSE its square root."""
    errors = np.asarray(simulation_result.latest_estimations) - simulation_result.examinees

    assert simulation_result.bias == pytest.approx(errors.mean())
    assert simulation_result.mse == pytest.approx(np.mean(errors**2))
    assert simulation_result.rmse == pytest.approx(np.sqrt(np.mean(errors**2)))


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
