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


@composite
def scale_mapping(draw: DrawFn) -> tuple[float, float, float, float]:
  """``theta_min, theta_max, scale_min, scale_max`` with strict ordering."""
  theta_min = draw(st.floats(min_value=-5.0, max_value=2.0, allow_nan=False, allow_infinity=False))
  theta_max = draw(st.floats(min_value=theta_min + 0.5, max_value=6.0, allow_nan=False, allow_infinity=False))
  scale_min = draw(st.floats(min_value=-200.0, max_value=400.0, allow_nan=False, allow_infinity=False))
  scale_max = draw(st.floats(min_value=scale_min + 1.0, max_value=900.0, allow_nan=False, allow_infinity=False))
  return (theta_min, theta_max, scale_min, scale_max)


@composite
def uniform_initializer_bounds(draw: DrawFn) -> tuple[float, float]:
  """Ordered ``(lo, hi)`` for :class:`catsim.initialization.RandomInitializer` uniform draws."""
  lo = draw(st.floats(min_value=-6.0, max_value=4.0, allow_nan=False, allow_infinity=False))
  hi = draw(st.floats(min_value=lo + 0.25, max_value=6.0, allow_nan=False, allow_infinity=False))
  return (lo, hi)
