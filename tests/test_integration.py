"""Integration tests for the CAT engine and simulation pipeline."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from sklearn.cluster import KMeans

from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer, InitializationDistribution, RandomInitializer
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  ClusterSelector,
  LinearSelector,
  MaxInfoBBlockSelector,
  MaxInfoSelector,
  MaxInfoStratSelector,
  RandomesqueSelector,
  RandomSelector,
  The54321Selector,
  UrrySelector,
)
from catsim.selection.base import BaseSelector
from catsim.simulation import SimulationRunner
from catsim.state import SessionStatus, SimulationResult
from catsim.stopping import BaseStopper, ConfidenceIntervalStopper, MinErrorStopper
from catsim.stopping import TestLengthStopper as LengthStopper


def run_simulation(
  item_bank: ItemBank,
  selector: BaseSelector,
  stopper: BaseStopper,
  *,
  estimator: NumericalSearchEstimator | None = None,
  examinees: int = 12,
  seed: int = 42,
) -> SimulationResult:
  """Run a small deterministic simulation and return its aggregate result."""
  runner = SimulationRunner(
    item_bank,
    RandomInitializer(InitializationDistribution.UNIFORM, (-2, 2)),
    selector,
    estimator or NumericalSearchEstimator(),
    stopper,
    seed=seed,
  )
  result = runner.run(examinees)
  assert len(result.sessions) == examinees
  assert all(session.status == SessionStatus.STOPPED for session in result.sessions)
  assert result.exposure_counts.sum() == sum(len(session.administered_item_ids) for session in result.sessions)
  return result


@pytest.mark.integration
@pytest.mark.parametrize("method", sorted(NumericalSearchEstimator.available_methods()))
def test_estimation_methods_complete_simulation(method: str) -> None:
  """All supported estimator methods should complete an end-to-end run."""
  item_bank = ItemBank.generate_item_bank(80, seed=42)
  result = run_simulation(
    item_bank,
    MaxInfoSelector(),
    LengthStopper(max_items=6),
    estimator=NumericalSearchEstimator(method=method),
  )

  assert all(len(session.administered_item_ids) == 6 for session in result.sessions)


@pytest.mark.integration
@pytest.mark.parametrize(
  "factory",
  [
    pytest.param(lambda: LinearSelector(list(range(8))), id="linear"),
    pytest.param(lambda: AStratSelector(8), id="astrat"),
    pytest.param(lambda: AStratBBlockSelector(8), id="astrat_bblock"),
    pytest.param(lambda: MaxInfoStratSelector(8), id="max_info_strat"),
    pytest.param(lambda: MaxInfoBBlockSelector(8), id="max_info_bblock"),
    pytest.param(lambda: The54321Selector(8), id="54321"),
    pytest.param(lambda: RandomesqueSelector(3), id="randomesque"),
  ],
)
def test_finite_selectors_complete_fixed_length_runs(factory: Callable[[], BaseSelector]) -> None:
  """Finite selectors should drive the shared simulation path without legacy adapters."""
  item_bank = ItemBank.generate_item_bank(80, seed=42)
  result = run_simulation(item_bank, factory(), LengthStopper(max_items=8))

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
  """Infinite selectors should work end to end with variable-length stopping rules."""
  item_bank = ItemBank.generate_item_bank(100, seed=42)
  result = run_simulation(item_bank, selector, MinErrorStopper(0.6, min_items=3, max_items=10))

  assert all(3 <= len(session.administered_item_ids) <= 10 for session in result.sessions)


@pytest.mark.integration
@pytest.mark.parametrize("method", ["item_info", "cluster_info", "weighted_info"])
def test_cluster_selector_completes_runs(method: str) -> None:
  """ClusterSelector should operate through SimulationRunner for all supported methods."""
  item_bank = ItemBank.generate_item_bank(60, seed=42)
  clusters = list(KMeans(n_clusters=4, n_init="auto", random_state=42).fit_predict(item_bank.items))
  result = run_simulation(
    item_bank,
    ClusterSelector(clusters=clusters, method=method, r_max=0.3),
    LengthStopper(max_items=6),
  )

  assert all(len(session.administered_item_ids) == 6 for session in result.sessions)


@pytest.mark.integration
@pytest.mark.parametrize(
  "stopper,expected_min,expected_max",
  [
    pytest.param(LengthStopper(max_items=5), 5, 5, id="fixed_length"),
    pytest.param(MinErrorStopper(0.6, min_items=3, max_items=9), 3, 9, id="min_error"),
    pytest.param(
      ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.8, min_items=3, max_items=9),
      3,
      9,
      id="confidence_interval",
    ),
  ],
)
def test_stoppers_complete_runs(stopper: BaseStopper, expected_min: int, expected_max: int) -> None:
  """Supported stoppers should terminate sessions within their configured limits."""
  item_bank = ItemBank.generate_item_bank(100, seed=42)
  result = run_simulation(item_bank, MaxInfoSelector(), stopper)

  assert all(expected_min <= len(session.administered_item_ids) <= expected_max for session in result.sessions)


@pytest.mark.integration
def test_manual_and_batched_flows_share_equivalent_fixed_length_behavior() -> None:
  """Manual and automated execution should agree on core stopping semantics."""
  item_bank = ItemBank.generate_item_bank(40, seed=42)
  initializer = FixedPointInitializer(0.0)
  selector = MaxInfoSelector()
  estimator = NumericalSearchEstimator()
  stopper = LengthStopper(max_items=4)

  manual_runner = SimulationRunner(item_bank, initializer, selector, estimator, stopper, seed=42)
  batched_runner = SimulationRunner(
    item_bank,
    initializer,
    MaxInfoSelector(),
    NumericalSearchEstimator(),
    stopper,
    seed=42,
  )

  manual_result = manual_runner.run([0.0])
  batched_result = batched_runner.run(1)

  assert len(manual_result.sessions[0].administered_item_ids) == 4
  assert len(batched_result.sessions[0].administered_item_ids) == 4
