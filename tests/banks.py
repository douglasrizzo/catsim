"""Build the small item banks whose expected selections a test works out by hand."""

from __future__ import annotations

import numpy as np
from irtlib import ItemBank


def two_pl_bank(discriminations: list[float], difficulties: list[float]) -> ItemBank:
  """Build a 2PL bank, whose item information at theta is a^2 P (1 - P)."""
  n_items = len(discriminations)
  return ItemBank(np.column_stack([discriminations, difficulties, np.zeros(n_items), np.ones(n_items)]))
