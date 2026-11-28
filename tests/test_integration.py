"""Integration tests for the CAT engine and simulation pipeline."""

from __future__ import annotations

from collections.abc import Callable
from typing import override

import numpy as np
import pytest
from irtlib import ItemBank

from catsim import CatEngine, Estimate, SimulationResult, StepContext, StopReason, simulate
from catsim.estimation import BaseEstimator, EAPEstimator, MAPEstimator, NumericalSearchEstimator
from catsim.initialization import UniformInitializer
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  BaseSelector,
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
  """Run a small deterministic simulation of examinees drawn from N(0, 1) and return its aggregate result."""
  engine = CatEngine(
    item_bank, UniformInitializer(low=-2.0, high=2.0), selector, estimator or NumericalSearchEstimator(), stopper
  )
  true_thetas = np.random.default_rng(seed).normal(0, 1, examinees)
  result = simulate(engine, true_thetas=true_thetas, rng=seed)
  assert len(result.sessions) == examinees
  assert all(session.stop_reason is not None for session in result.sessions)
  assert result.exposure_counts.sum() == sum(len(session.administered_items) for session in result.sessions)
  return result


@pytest.mark.integration
@pytest.mark.parametrize("factory", [pytest.param(EAPEstimator, id="eap"), pytest.param(MAPEstimator, id="map")])
def test_bayesian_estimators_complete_simulation(factory: Callable[[], BaseEstimator]) -> None:
  """Bayesian estimators should complete end-to-end runs, also as MEI's look-ahead estimator.

  One estimator instance serves every examinee, and MEI calls it again with hypothetical
  responses, so sharing one instance between the engine and MEI must not change any estimate.
  """
  item_bank = ItemBank.generate(80, rng=42)
  stopper = MaxLengthStopper(6)
  shared = factory()
  reused = run_simulation(item_bank, MEISelector(estimator=shared), stopper, estimator=shared, examinees=4)
  fresh = run_simulation(item_bank, MEISelector(estimator=factory()), stopper, estimator=factory(), examinees=4)

  assert all(len(session.administered_items) == 6 for session in reused.sessions)
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
def test_selectors_complete_fixed_length_runs(factory: Callable[[], BaseSelector]) -> None:
  """Each selector drives simulate through eight-item sessions.

  A selector that raises mid-run, such as a stratified selector built for eight items that runs out of strata, or a
  run that stops a session before its eighth item, fails.
  """
  item_bank = ItemBank.generate(80, rng=42)
  result = run_simulation(item_bank, factory(), MaxLengthStopper(8))

  assert all(len(session.administered_items) == 8 for session in result.sessions)
  assert result.overlap_rate() is not None


@pytest.mark.integration
@pytest.mark.parametrize(
  "selector",
  [
    pytest.param(MaxInfoSelector(), id="max_info"),
    pytest.param(RandomSelector(), id="random"),
    pytest.param(UrrySelector(), id="urry"),
  ],
)
def test_selectors_complete_variable_length_runs(selector: BaseSelector) -> None:
  """A selector that raises, or a session that ends outside its stopping rule's 3 to 10 items, fails the run."""
  item_bank = ItemBank.generate(100, rng=42)
  result = run_simulation(item_bank, selector, StopRule(StandardErrorStopper(0.6), MaxLengthStopper(10), min_items=3))

  assert all(3 <= len(session.administered_items) <= 10 for session in result.sessions)


@pytest.mark.integration
@pytest.mark.parametrize("method", ["item_info", "cluster_info", "weighted_info"])
def test_cluster_selector_completes_runs(method: str) -> None:
  """A cluster-selection method that raises or stops before six items under simulate fails the run.

  The selector treats cluster ids as labels, and this test checks only that each method completes a run, so four
  fixed groups of 15 items are enough. Four groups for six items also make each session reuse a cluster.
  """
  item_bank = ItemBank.generate(60, rng=42)
  clusters = (np.arange(item_bank.n_items) % 4).tolist()
  result = run_simulation(
    item_bank,
    ClusterSelector(clusters=clusters, method=method, r_max=0.3),
    MaxLengthStopper(6),
  )

  assert all(len(session.administered_items) == 6 for session in result.sessions)


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
  item_bank = ItemBank.generate(100, rng=42)
  result = run_simulation(item_bank, MaxInfoSelector(), stopper)

  assert all(expected_min <= len(session.administered_items) <= expected_max for session in result.sessions)


@pytest.mark.integration
def test_a_session_whose_stopper_never_holds_ends_when_the_bank_runs_out() -> None:
  """No stopper checks bank exhaustion, so an engine that mishandles an exhausted selector raises or ends early.

  A standard error of 0.001 needs an information of a million, which five items cannot give.
  """
  item_bank = ItemBank.generate(5, rng=42)
  result = run_simulation(item_bank, MaxInfoSelector(), StandardErrorStopper(0.001))

  assert all(sorted(session.administered_items) == [0, 1, 2, 3, 4] for session in result.sessions)


class DrawingEstimator(BaseEstimator):
  """Estimator stub whose estimate is a draw from its context's generator, and which records every generator."""

  def __init__(self) -> None:
    self.seen: list[np.random.Generator] = []

  @override
  def estimate(self, context: StepContext) -> Estimate:
    """Record the context's generator and return a normal draw from it as the estimate."""
    self.seen.append(context.rng)
    return Estimate(theta=float(context.rng.normal()), se=1.0)


@pytest.mark.integration
def test_stepping_the_engine_by_hand_reproduces_a_simulated_session() -> None:
  """A session stepped through start, select and answer on stream (i, 1) is the session simulate gives examinee i.

  The uniform initializer, the randomesque selector and an estimator that draws all draw from the session's generator,
  in start, select and answer, so the two paths agree only if simulate gives examinee 2 the generator of
  ``SeedSequence(seed, spawn_key=(2, 1))`` for all three steps (ADR-0013) and its unchecked steps are the public ones.
  The examinees are recorded, so the answers come from their rows and draw nothing. Fails if a step of simulate draws
  from another generator, or if answer ignores the generator it is given.
  """
  item_bank = ItemBank.generate(40, rng=42)
  responses = (np.random.default_rng(1).random((3, item_bank.n_items)) < 0.5).astype(float)
  responses[2, ::3] = np.nan

  def engine(estimator: BaseEstimator) -> CatEngine:
    return CatEngine(item_bank, UniformInitializer(), RandomesqueSelector(3), estimator, MaxLengthStopper(6))

  simulated = simulate(engine(DrawingEstimator()), responses=responses, rng=7).sessions[2]

  estimator = DrawingEstimator()
  manual_engine = engine(estimator)
  rng = np.random.default_rng(np.random.SeedSequence(7, spawn_key=(2, 1)))
  manual = manual_engine.start(rng=rng, excluded_items=range(0, item_bank.n_items, 3))
  # one item at a time, as an interactive caller would, until select proposes none
  while (selection := manual_engine.select(manual, rng=rng)).item is not None:
    manual = manual_engine.answer(selection.session, selection.item, bool(responses[2, selection.item]), rng=rng)
  manual = selection.session

  assert manual == simulated
  assert manual.stop_reason is StopReason.STOPPER
  assert len(estimator.seen) == 6
  assert all(seen is rng for seen in estimator.seen)
