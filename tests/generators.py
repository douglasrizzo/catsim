"""Stand-ins for numpy's Generator, for tests that script the random draws a strategy makes."""

from __future__ import annotations

import numpy as np
from typing_extensions import override

from catsim.typing import FloatArray, IntArray


class DeterministicRNG(np.random.Generator):
  """Deterministic stand-in for numpy's Generator in selector tests.

  ``uniform`` returns the given values, which must match the requested size, and ``choice`` records its candidates and
  probabilities in ``choice_calls`` and returns the most probable candidate, or the first without probabilities.
  """

  def __init__(self, uniform_values: list[float] | FloatArray) -> None:
    super().__init__(np.random.PCG64(0))
    self._uniform_values = np.asarray(uniform_values, dtype=float)
    self.choice_calls: list[tuple[IntArray, FloatArray | None]] = []

  # The overrides are unannotated: Generator's methods are overloads that these stubs do not reproduce.
  @override
  def uniform(self, low=0.0, high=1.0, size=None):
    del low, high
    if size is None:
      return float(self._uniform_values[0])
    expected_size = int(np.prod(size)) if isinstance(size, tuple) else int(size)
    if expected_size != self._uniform_values.size:
      msg = f"expected {expected_size} uniform values, got {self._uniform_values.size}"
      raise ValueError(msg)
    return self._uniform_values.copy()

  @override
  def choice(self, a, size=None, replace=True, p=None, axis=0, shuffle=True):
    del size, replace, axis, shuffle
    candidate_indices = np.asarray(a, dtype=int)
    probabilities = None if p is None else np.asarray(p, dtype=float)
    self.choice_calls.append((
      candidate_indices.copy(),
      probabilities.copy() if probabilities is not None else None,
    ))
    if probabilities is None:
      return int(candidate_indices[0])
    return int(candidate_indices[np.argmax(probabilities)])
