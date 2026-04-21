"""Shared test fixtures for the refactored CAT architecture."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer, RandomInitializer
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector, RandomSelector
from catsim.simulation import SimulationRunner
from catsim.stopping import MinErrorStopper
from catsim.stopping import TestLengthStopper as LengthStopper


@pytest.fixture
def rng() -> np.random.Generator:
  """Provide a deterministic RNG for tests."""
  return np.random.default_rng(42)


@pytest.fixture
def item_bank() -> ItemBank:
  """Provide a deterministic item bank for tests."""
  return ItemBank.generate_item_bank(50, seed=42)


@pytest.fixture
def fixed_initializer() -> FixedPointInitializer:
  """Provide a deterministic initializer."""
  return FixedPointInitializer(0.0)


@pytest.fixture
def random_initializer() -> RandomInitializer:
  """Provide a random initializer with deterministic range."""
  return RandomInitializer()


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
def fixed_length_stopper() -> LengthStopper:
  """Provide a fixed-length stopper."""
  return LengthStopper(max_items=10)


@pytest.fixture
def min_error_stopper() -> MinErrorStopper:
  """Provide a min-error stopper."""
  return MinErrorStopper(0.5, max_items=10)


@pytest.fixture
def simulation_runner(item_bank: ItemBank) -> SimulationRunner:
  """Provide a deterministic simulation runner."""
  return SimulationRunner(
    item_bank,
    FixedPointInitializer(0.0),
    RandomSelector(),
    NumericalSearchEstimator(),
    LengthStopper(max_items=10),
    seed=42,
  )


@pytest.fixture
def simulation_result(simulation_runner: SimulationRunner):
  """Provide a completed simulation result."""
  return simulation_runner.run(5)


@pytest.fixture(autouse=True)
def close_matplotlib_figures():
  """Ensure tests do not leak open matplotlib figures."""
  yield
  plt.close("all")
