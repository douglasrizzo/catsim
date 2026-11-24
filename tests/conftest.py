"""Shared test fixtures for the refactored CAT architecture."""

from __future__ import annotations

from collections.abc import Callable

import matplotlib
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from catsim.engine import CatEngine
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  BaseSelector,
  ClusterSelector,
  IntervalInfoSelector,
  KLSelector,
  LinearSelector,
  MaxInfoBBlockSelector,
  MaxInfoSelector,
  MaxInfoStratSelector,
  MEISelector,
  MLWISelector,
  ProgressiveSelector,
  ProportionalSelector,
  RandomesqueSelector,
  RandomSelector,
  The54321Selector,
  UrrySelector,
)
from catsim.simulation import SimulationRunner
from catsim.stopping import MaxLengthStopper

# The test length the position-dependent selectors of SELECTOR_FACTORIES are built for.
TEST_SIZE = 10

# Every concrete selector catsim.selection exports, each factory building its selector for a bank of n_items items.
# A stratified selector needs an item in every stratum, so its test is no longer than the bank.
SELECTOR_FACTORIES: dict[str, Callable[[int], BaseSelector]] = {
  "MaxInfoSelector": lambda _: MaxInfoSelector(),
  "UrrySelector": lambda _: UrrySelector(),
  "KLSelector": lambda _: KLSelector(),
  "IntervalInfoSelector": lambda _: IntervalInfoSelector(),
  "RandomSelector": lambda _: RandomSelector(),
  "RandomesqueSelector": lambda _: RandomesqueSelector(bin_size=5),
  "The54321Selector": lambda _: The54321Selector(),
  "LinearSelector": lambda n_items: LinearSelector(list(range(n_items))),
  "ProgressiveSelector": lambda _: ProgressiveSelector(test_size=TEST_SIZE),
  "ProportionalSelector": lambda _: ProportionalSelector(test_size=TEST_SIZE),
  "ClusterSelector": lambda n_items: ClusterSelector(clusters=[i % 3 for i in range(n_items)]),
  "AStratSelector": lambda n_items: AStratSelector(test_size=min(TEST_SIZE, n_items)),
  "AStratBBlockSelector": lambda n_items: AStratBBlockSelector(test_size=min(TEST_SIZE, n_items)),
  "MaxInfoStratSelector": lambda n_items: MaxInfoStratSelector(test_size=min(TEST_SIZE, n_items)),
  "MaxInfoBBlockSelector": lambda n_items: MaxInfoBBlockSelector(test_size=min(TEST_SIZE, n_items)),
  "MEISelector": lambda _: MEISelector(),
  "MLWISelector": lambda _: MLWISelector(),
}

# pytester runs pytest on generated test modules (test_stopping checks what pytest collects from catsim)
pytest_plugins = ["pytester"]


@pytest.fixture
def item_bank() -> ItemBank:
  """Provide a deterministic item bank for tests."""
  return ItemBank.generate(50, rng=42)


@pytest.fixture
def fixed_initializer() -> FixedPointInitializer:
  """Provide a deterministic initializer."""
  return FixedPointInitializer(0.0)


@pytest.fixture
def random_selector() -> RandomSelector:
  """Provide a random selector."""
  return RandomSelector()


@pytest.fixture
def max_info_selector() -> MaxInfoSelector:
  """Provide a max-information selector."""
  return MaxInfoSelector()


@pytest.fixture
def estimator() -> NumericalSearchEstimator:
  """Provide the default numerical estimator."""
  return NumericalSearchEstimator()


@pytest.fixture
def fixed_length_stopper() -> MaxLengthStopper:
  """Provide a fixed-length stopper."""
  return MaxLengthStopper(10)


@pytest.fixture
def simulation_runner(item_bank: ItemBank) -> SimulationRunner:
  """Provide a simulation runner that gives every examinee 10 random items."""
  return SimulationRunner(
    CatEngine(item_bank, FixedPointInitializer(0.0), RandomSelector(), NumericalSearchEstimator(), MaxLengthStopper(10))
  )


@pytest.fixture
def simulation_result(simulation_runner: SimulationRunner):
  """Provide a completed simulation result for five examinees."""
  return simulation_runner.run([-1.0, -0.5, 0.0, 0.5, 1.0], rng=42)


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
  """Ensure tests do not leak open matplotlib figures."""
  yield
  plt.close("all")
