"""Property-based tests for :class:`catsim.CatEngine`."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from hypothesis import assume, given, settings
from hypothesis import strategies as st

from catsim import CatEngine, Estimate, ItemBank, Session
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer
from catsim.selection import MaxInfoSelector
from catsim.stopping import MaxLengthStopper
from tests.hypothesis_strategies import item_params_matrix

if TYPE_CHECKING:
  import numpy as np

ENGINE_SETTINGS = settings(max_examples=20)


@given(
  params=item_params_matrix(min_items=6, max_items=16),
  administered_prefix=st.integers(min_value=0, max_value=10),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@ENGINE_SETTINGS
def test_select_returns_unseen_item_within_bank(
  params: np.ndarray,
  administered_prefix: int,
  seed: int,
) -> None:
  """The engine hands the selector the session's administered items, so the item it selects is a new one.

  An engine that built the selection context without the session's items, or from another session's, would get an
  administered item back.
  """
  n = params.shape[0]
  assume(administered_prefix < n - 1)
  bank = ItemBank(params)
  engine = CatEngine(
    bank,
    FixedPointInitializer(0.0),
    MaxInfoSelector(),
    NumericalSearchEstimator(),
    MaxLengthStopper(500),
  )
  administered = tuple(range(administered_prefix))
  session = Session(
    administered_items=administered,
    responses=(True,) * administered_prefix,
    estimates=(Estimate(0.0, math.inf),) * (administered_prefix + 1),
  )
  item = engine.select(session, rng=seed).item
  assert type(item) is int
  assert 0 <= item < n
  assert item not in administered
