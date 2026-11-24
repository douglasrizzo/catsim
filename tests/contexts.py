"""Build the StepContext a test passes to a strategy it calls directly."""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from catsim import ItemBank, StepContext
from catsim.typing import FloatArray


def make_context(
  item_bank: ItemBank,
  administered_items: Sequence[int] = (),
  responses: Sequence[bool] | None = None,
  *,
  theta: float = 0.0,
  se: float = math.inf,
  rng: np.random.Generator | None = None,
  exposure_rates: FloatArray | None = None,
  excluded_items: Sequence[int] = (),
) -> StepContext:
  """Return a context for a session that administered ``administered_items``.

  ``responses`` defaults to all correct, for strategies that do not read them, and ``rng`` to a generator seeded with
  0, for strategies that draw from it only to break ties.
  """
  return StepContext(
    item_bank=item_bank,
    rng=np.random.default_rng(0) if rng is None else rng,
    administered_items=tuple(administered_items),
    responses=(True,) * len(administered_items) if responses is None else tuple(responses),
    excluded_items=tuple(excluded_items),
    theta=theta,
    se=se,
    exposure_rates=exposure_rates,
  )
