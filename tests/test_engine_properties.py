"""Property-based tests for :class:`catsim.engine.CatEngine`."""

from __future__ import annotations

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from catsim.engine import CatEngine, RunContext
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector
from catsim.state import CatSessionState, SessionStatus
from catsim.stopping import TestLengthStopper as LengthStopper
from tests.hypothesis_strategies import item_params_matrix

ENGINE_SETTINGS = settings(max_examples=20, deadline=None)


@given(
  params=item_params_matrix(min_items=6, max_items=16),
  administered_prefix=st.integers(min_value=0, max_value=10),
  seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@ENGINE_SETTINGS
def test_select_next_returns_unseen_item_within_bank(
  params: np.ndarray,
  administered_prefix: int,
  seed: int,
) -> None:
  n = params.shape[0]
  assume(administered_prefix < n - 1)
  bank = ItemBank(params, validate=True)
  engine = CatEngine(
    FixedPointInitializer(0.0),
    MaxInfoSelector(),
    NumericalSearchEstimator(method="bounded"),
    LengthStopper(max_items=500),
  )
  administered = list(range(administered_prefix))
  state = CatSessionState(
    session_id=0,
    current_theta=0.0,
    status=SessionStatus.IN_PROGRESS,
    administered_item_ids=administered,
  )
  ctx = RunContext(rng=np.random.default_rng(seed))
  item_id = engine.select_next(state, bank, ctx)
  assert item_id is not None
  assert isinstance(item_id, (int, np.integer))
  assert 0 <= int(item_id) < n
  assert int(item_id) not in administered
