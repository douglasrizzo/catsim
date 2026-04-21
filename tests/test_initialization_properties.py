"""Property-based tests for CAT initialization."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from catsim.initialization import InitializationDistribution, RandomInitializer
from catsim.item_bank import ItemBank
from tests.hypothesis_strategies import uniform_initializer_bounds

INIT_SETTINGS = settings(max_examples=35, deadline=None)


@given(bounds=uniform_initializer_bounds(), seed=st.integers(min_value=0, max_value=2**31 - 1))
@INIT_SETTINGS
def test_random_uniform_initializer_stays_in_bounds(bounds: tuple[float, float], seed: int) -> None:
  lo, hi = bounds
  init = RandomInitializer(dist_type=InitializationDistribution.UNIFORM, dist_params=(lo, hi))
  bank = ItemBank.generate_item_bank(5, seed=0)
  rng = np.random.default_rng(seed)
  theta = init.initialize(item_bank=bank, rng=rng)
  low, high = min(lo, hi), max(lo, hi)
  assert low <= theta <= high
