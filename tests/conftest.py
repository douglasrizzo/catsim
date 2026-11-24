"""Shared test fixtures for the refactored CAT architecture."""

from __future__ import annotations

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from catsim.engine import CatEngine
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector, RandomSelector
from catsim.simulation import SimulationRunner
from catsim.stopping import MaxLengthStopper

# pytester runs pytest on generated test modules (test_stopping checks what pytest collects from catsim)
pytest_plugins = ["pytester"]


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
