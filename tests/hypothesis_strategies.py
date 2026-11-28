"""Hypothesis strategies shared by property-based tests."""

from __future__ import annotations

import numpy as np
from hypothesis import strategies as st
from hypothesis.strategies import DrawFn, composite


@composite
def pl4_item_row(draw: DrawFn) -> tuple[float, float, float, float]:
  """One valid 4PL item row (discrimination, difficulty, guessing, upper asymptote)."""
  a = draw(st.floats(min_value=0.25, max_value=2.5, allow_nan=False, allow_infinity=False))
  b = draw(st.floats(min_value=-3.5, max_value=3.5, allow_nan=False, allow_infinity=False))
  c = draw(st.floats(min_value=0.0, max_value=0.25, allow_nan=False, allow_infinity=False))
  d = draw(st.floats(min_value=0.55, max_value=1.0, allow_nan=False, allow_infinity=False))
  if d <= c + 0.05:
    d = min(1.0, c + 0.2)
  return (a, b, c, float(d))


@composite
def item_params_matrix(draw: DrawFn, *, min_items: int = 1, max_items: int = 18) -> np.ndarray:
  """An ``(n, 4)`` validated item-parameter matrix."""
  n = draw(st.integers(min_value=min_items, max_value=max_items))
  rows = [draw(pl4_item_row()) for _ in range(n)]
  return np.asarray(rows, dtype=np.float64)
