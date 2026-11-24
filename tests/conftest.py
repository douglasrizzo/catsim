"""Fixtures shared by the test suite, and the registry of every selector the selection tests run."""

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

# Every concrete selector catsim.selection exports, each factory building its selector for a bank of n_items items.
# The selection properties run sessions until the selector has no item left, so the selectors that can run out before
# the bank does are built to: LinearSelector lists every other item, from the last down, and a stratified selector plans
# a test twice the bank's length in two strata, so its first stage outlasts its first stratum.
SELECTOR_FACTORIES: dict[str, Callable[[int], BaseSelector]] = {
  "MaxInfoSelector": lambda _: MaxInfoSelector(),
  "UrrySelector": lambda _: UrrySelector(),
  "KLSelector": lambda _: KLSelector(),
  "IntervalInfoSelector": lambda _: IntervalInfoSelector(),
  "RandomSelector": lambda _: RandomSelector(),
  "RandomesqueSelector": lambda _: RandomesqueSelector(bin_size=5),
  "The54321Selector": lambda _: The54321Selector(),
  "LinearSelector": lambda n_items: LinearSelector(list(range(n_items - 1, -1, -2))),
  "ProgressiveSelector": lambda _: ProgressiveSelector(test_size=10),
  "ProportionalSelector": lambda _: ProportionalSelector(test_size=10),
  "ClusterSelector": lambda n_items: ClusterSelector(clusters=[i % 3 for i in range(n_items)]),
  "AStratSelector": lambda n_items: AStratSelector(test_size=2 * n_items, n_strata=2),
  "AStratBBlockSelector": lambda n_items: AStratBBlockSelector(test_size=2 * n_items, n_strata=2),
  "MaxInfoStratSelector": lambda n_items: MaxInfoStratSelector(test_size=2 * n_items, n_strata=2),
  "MaxInfoBBlockSelector": lambda n_items: MaxInfoBBlockSelector(test_size=2 * n_items, n_strata=2),
  "MEISelector": lambda _: MEISelector(),
  "MLWISelector": lambda _: MLWISelector(),
}

# The selectors of SELECTOR_FACTORIES that run out once half the bank, rounded up, is administered: the linear list
# holds that many items, and a first stratum of two holds that many whether or not the selector blocks.
HALF_BANK_SELECTORS = frozenset({
  "LinearSelector",
  "AStratSelector",
  "AStratBBlockSelector",
  "MaxInfoStratSelector",
  "MaxInfoBBlockSelector",
})

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
