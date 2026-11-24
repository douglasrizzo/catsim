"""Tests for catsim.context.StepContext."""

import dataclasses

import numpy as np
import pytest

from catsim import Estimate, ItemBank, StepContext

BANK = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0]]))


def test_lists_are_stored_as_tuples() -> None:
  """A list kept as given would let the caller, or a strategy, change the history a frozen context describes."""
  items, responses, excluded = [0, 2], [True, False], [1]

  context = StepContext(
    item_bank=BANK,
    rng=np.random.default_rng(0),
    administered_items=items,  # ty: ignore[invalid-argument-type]
    responses=responses,  # ty: ignore[invalid-argument-type]
    excluded_items=excluded,  # ty: ignore[invalid-argument-type]
  )
  items.append(1)
  responses.append(True)
  excluded.clear()

  assert (context.administered_items, context.responses, context.excluded_items) == ((0, 2), (True, False), (1,))


@pytest.mark.parametrize(
  ("administered_items", "responses", "match"),
  [
    pytest.param((0, 1, 2), (True,), "same length", id="too_short"),
    pytest.param((0, 1, 2), (True, True, True, False), "same length", id="too_long"),
    pytest.param((), (True,), "same length", id="responses_without_items"),
    pytest.param((0, 1, 2), (1.0, 0.5, 0.0), "only Boolean", id="not_boolean"),
  ],
)
def test_rejects_responses_that_do_not_fit_the_items(
  administered_items: tuple[int, ...], responses: tuple[bool, ...], match: str
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


@pytest.mark.parametrize("field", ["administered_items", "excluded_items"])
@pytest.mark.parametrize("index", [-1, 3])
def test_rejects_item_indices_outside_the_bank(field: str, index: int) -> None:
  """NumPy reads -1 as the last item, so excluding item -1 silently dropped item 2; 3 raised a bare IndexError."""
  administered, excluded = ((index,), ()) if field == "administered_items" else ((), (index,))
  with pytest.raises(ValueError, match=f"{field} must hold item indices between 0 and 2"):
    StepContext(
      item_bank=BANK,
      rng=np.random.default_rng(0),
      administered_items=administered,
      responses=(True,) * len(administered),
      excluded_items=excluded,
    )


@pytest.mark.parametrize(
  ("theta", "se", "evaluations", "match"),
  [
    pytest.param(float("nan"), 0.3, None, "theta must be finite", id="nan_theta"),
    pytest.param(float("inf"), 0.3, None, "theta must be finite", id="infinite_theta"),
    pytest.param(0.0, float("nan"), None, "se must be non-negative", id="nan_se"),
    pytest.param(0.0, -0.1, None, "se must be non-negative", id="negative_se"),
    pytest.param(0.0, 0.3, -1, "evaluations must be non-negative", id="negative_count"),
  ],
)
def test_estimate_rejects_values_no_stopper_can_judge(
  theta: float, se: float, evaluations: int | None, match: str
) -> None:
  """A NaN error fails every comparison: a confidence-interval stopper then stops at once, and an SE stopper never."""
  with pytest.raises(ValueError, match=match):
    Estimate(theta, se, evaluations)
