"""Integration tests for the CAT engine and simulation pipeline."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from catsim.engine import CatEngine, RunContext, SimulatedResponseProvider
from catsim.estimation import BaseEstimator, EAPEstimator, MAPEstimator, NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer, UniformInitializer
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  ClusterSelector,
  LinearSelector,
  MaxInfoBBlockSelector,
  MaxInfoSelector,
  MaxInfoStratSelector,
  MEISelector,
  RandomesqueSelector,
  RandomSelector,
  The54321Selector,
  UrrySelector,
)
from catsim.selection.base import BaseSelector
from catsim.simulation import SimulationRunner
from catsim.state import ExposureTracker, SimulationResult, StopReason
from catsim.stopping import BaseStopper, ConfidenceIntervalStopper, MaxLengthStopper, StandardErrorStopper, StopRule


def run_simulation(
  item_bank: ItemBank,
  selector: BaseSelector,
  stopper: BaseStopper,
  *,
  estimator: BaseEstimator | None = None,
  examinees: int = 12,
  seed: int = 42,
) -> SimulationResult:
  """Run a small deterministic simulation and return its aggregate result."""
  runner = SimulationRunner(
    item_bank,
    UniformInitializer(low=-2.0, high=2.0),
    selector,
    estimator or NumericalSearchEstimator(),
    stopper,
    seed=seed,
  )
  result = runner.run(examinees)
  assert len(result.sessions) == examinees
  assert all(session.is_stopped for session in result.sessions)
  assert result.exposure_counts.sum() == sum(len(session.administered_item_ids) for session in result.sessions)
  return result


@pytest.mark.integration
@pytest.mark.parametrize("factory", [pytest.param(EAPEstimator, id="eap"), pytest.param(MAPEstimator, id="map")])
def test_bayesian_estimators_complete_simulation(factory: Callable[[], BaseEstimator]) -> None:
  """Bayesian estimators should complete end-to-end runs, also as MEI's look-ahead estimator.

  One estimator instance serves every examinee, and MEI calls it again with hypothetical
  responses, so sharing one instance between the engine and MEI must not change any estimate.
  """
  item_bank = ItemBank.generate_item_bank(80, seed=42)
  stopper = MaxLengthStopper(6)
  shared = factory()
  reused = run_simulation(item_bank, MEISelector(estimator=shared), stopper, estimator=shared, examinees=4)
  fresh = run_simulation(item_bank, MEISelector(estimator=factory()), stopper, estimator=factory(), examinees=4)

  assert all(len(session.administered_item_ids) == 6 for session in reused.sessions)
  assert [session.theta_history for session in reused.sessions] == [session.theta_history for session in fresh.sessions]


@pytest.mark.integration
@pytest.mark.parametrize(
  "factory",
  [
    pytest.param(lambda: LinearSelector(list(range(8))), id="linear"),
    pytest.param(lambda: AStratSelector(8), id="astrat"),
    pytest.param(lambda: AStratBBlockSelector(8), id="astrat_bblock"),
    pytest.param(lambda: MaxInfoStratSelector(8), id="max_info_strat"),
    pytest.param(lambda: MaxInfoBBlockSelector(8), id="max_info_bblock"),
    pytest.param(The54321Selector, id="54321"),
    pytest.param(lambda: RandomesqueSelector(3), id="randomesque"),
  ],
)
def test_finite_selectors_complete_fixed_length_runs(factory: Callable[[], BaseSelector]) -> None:
  """Finite selectors should drive the shared simulation path without legacy adapters."""
  item_bank = ItemBank.generate_item_bank(80, seed=42)
  result = run_simulation(item_bank, factory(), MaxLengthStopper(8))

  assert all(len(session.administered_item_ids) == 8 for session in result.sessions)
  assert result.overlap_rate is not None


@pytest.mark.integration
@pytest.mark.parametrize(
  "selector",
  [
    pytest.param(MaxInfoSelector(), id="max_info"),
    pytest.param(RandomSelector(), id="random"),
    pytest.param(UrrySelector(), id="urry"),
  ],
)
def test_infinite_selectors_complete_variable_length_runs(selector: BaseSelector) -> None:
  """An infinite selector that raises, or ends a session outside its stopping rule's 3 to 10 items, fails the run."""
  item_bank = ItemBank.generate_item_bank(100, seed=42)
  result = run_simulation(item_bank, selector, StopRule(StandardErrorStopper(0.6), MaxLengthStopper(10), min_items=3))

  assert all(3 <= len(session.administered_item_ids) <= 10 for session in result.sessions)


@pytest.mark.integration
@pytest.mark.parametrize("method", ["item_info", "cluster_info", "weighted_info"])
def test_cluster_selector_completes_runs(method: str) -> None:
  """A cluster-selection method that raises or stops before six items under SimulationRunner fails the run.

  The selector treats cluster ids as labels, and this test checks only that each method completes a run, so four
  fixed groups of 15 items are enough. Four groups for six items also make each session reuse a cluster.
  """
  item_bank = ItemBank.generate_item_bank(60, seed=42)
  clusters = (np.arange(item_bank.n_items) % 4).tolist()
  result = run_simulation(
    item_bank,
    ClusterSelector(clusters=clusters, method=method, r_max=0.3),
    MaxLengthStopper(6),
  )

  assert all(len(session.administered_item_ids) == 6 for session in result.sessions)


@pytest.mark.integration
@pytest.mark.parametrize(
  ("stopper", "expected_min", "expected_max"),
  [
    pytest.param(MaxLengthStopper(5), 5, 5, id="fixed_length"),
    pytest.param(StopRule(StandardErrorStopper(0.6), MaxLengthStopper(9), min_items=3), 3, 9, id="standard_error"),
    pytest.param(
      StopRule(ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.8), MaxLengthStopper(9), min_items=3),
      3,
      9,
      id="confidence_interval",
    ),
  ],
)
def test_stoppers_complete_runs(stopper: BaseStopper, expected_min: int, expected_max: int) -> None:
  """A stopper that ends a session before its minimum or after its maximum number of items fails the run."""
  item_bank = ItemBank.generate_item_bank(100, seed=42)
  result = run_simulation(item_bank, MaxInfoSelector(), stopper)

  assert all(expected_min <= len(session.administered_item_ids) <= expected_max for session in result.sessions)


@pytest.mark.integration
def test_a_session_whose_stopper_never_holds_ends_when_the_bank_runs_out() -> None:
  """No stopper checks bank exhaustion, so an engine that mishandles an exhausted selector raises or ends early.

  A standard error of 0.001 needs an information of a million, which five items cannot give.
  """
  item_bank = ItemBank.generate_item_bank(5, seed=42)
  result = run_simulation(item_bank, MaxInfoSelector(), StandardErrorStopper(0.001))

  assert all(sorted(session.administered_item_ids) == [0, 1, 2, 3, 4] for session in result.sessions)


@pytest.mark.integration
def test_stepping_the_engine_by_hand_reproduces_a_simulation_runner_session() -> None:
  """A session stepped through CatEngine matches the one SimulationRunner produces from the same seed.

  The randomesque selector and the simulated responses draw from the same generator in turn, so the two
  paths agree only if they make the same draws in the same order.
  """
  item_bank = ItemBank.generate_item_bank(40, seed=42)

  def components() -> tuple[FixedPointInitializer, RandomesqueSelector, NumericalSearchEstimator, MaxLengthStopper]:
    return FixedPointInitializer(0.0), RandomesqueSelector(3), NumericalSearchEstimator(), MaxLengthStopper(6)

  batched = SimulationRunner(item_bank, *components(), seed=7).run([0.3]).sessions[0]

  engine = CatEngine(*components())
  context = RunContext(
    rng=np.random.default_rng(7), exposure_tracker=ExposureTracker(item_bank.n_items), total_sessions=1
  )
  manual = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.3)
  # step one item at a time, as an interactive caller would, until a step reports the session stopped
  while not engine.step(manual, item_bank, SimulatedResponseProvider(), context).stopped:
    pass

  assert manual.administered_item_ids == batched.administered_item_ids
  assert manual.responses == batched.responses
  assert manual.theta_history == pytest.approx(batched.theta_history)
  assert manual.stop_reason == batched.stop_reason == StopReason.STOPPER
