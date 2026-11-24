"""Tests for catsim.context.StepContext."""

import dataclasses

import numpy as np
import pytest

from catsim import ItemBank, StepContext

BANK = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0]]))


def test_lists_are_stored_as_tuples() -> None:
  """A list kept as given would let the caller, or a strategy, change the history a frozen context describes."""
  items, responses, excluded = [0, 2], [True, False], [1]

  context = StepContext(
    item_bank=BANK,
    rng=np.random.default_rng(0),
    administered_items=items,
    responses=responses,
    excluded_items=excluded,
  )
  items.append(1)
  responses.append(True)
  excluded.clear()

  assert (context.administered_items, context.responses, context.excluded_items) == ((0, 2), (True, False), (1,))


@pytest.mark.parametrize(
  ("administered_items", "responses", "match"),
  [
    pytest.param([0, 1, 2], [True], "same length", id="too_short"),
    pytest.param([0, 1, 2], [True, True, True, False], "same length", id="too_long"),
    pytest.param([], [True], "same length", id="responses_without_items"),
    pytest.param([0, 1, 2], [1.0, 0.5, 0.0], "only Boolean", id="not_boolean"),
  ],
)
def test_rejects_responses_that_do_not_fit_the_items(
  administered_items: list[int], responses: list[bool], match: str
) -> None:
  """A context holds one Boolean response per administered item, so no strategy has to check.

  Before the check lived here, WLE spread a one-element vector over three items and MLE took its all-correct shortcut
  on it, both returning an estimate for answers that were never given.
  """
  with pytest.raises(ValueError, match=match):
    StepContext(
      item_bank=BANK, rng=np.random.default_rng(0), administered_items=administered_items, responses=responses
    )


def test_contexts_with_exposure_rates_compare_by_identity() -> None:
  """A generated __eq__ compares the exposure-rate arrays elementwise and raises; a generated __hash__ fails on them."""
  rates = np.array([0.1, 0.2, 0.3])
  context = StepContext(item_bank=BANK, rng=np.random.default_rng(0), exposure_rates=rates)
  twin = dataclasses.replace(context, exposure_rates=rates.copy())

  assert context != twin
  assert len({context, twin}) == 2
