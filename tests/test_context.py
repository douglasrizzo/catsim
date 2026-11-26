"""Tests for catsim.context.StepContext and catsim.Estimate, and the input checks they make (ADR-0009)."""

import dataclasses
import math
import re

import numpy as np
import pytest

from catsim import Estimate, ItemBank, StepContext

BANK = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0]]))


def _context(**fields: object) -> StepContext:
  """Return a context on BANK with a seeded generator and the given fields."""
  return StepContext(item_bank=BANK, rng=np.random.default_rng(0), **fields)  # ty: ignore[invalid-argument-type]


def test_lists_are_stored_as_tuples() -> None:
  """A list kept as given would let the caller, or a strategy, change the history a frozen context describes."""
  items, responses, excluded = [0, 2], [True, False], [1]

  context = _context(administered_items=items, responses=responses, excluded_items=excluded)
  items.append(1)
  responses.append(True)
  excluded.clear()

  assert (context.administered_items, context.responses, context.excluded_items) == ((0, 2), (True, False), (1,))


@pytest.mark.parametrize(
  ("administered_items", "responses"),
  [
    pytest.param((0, 1, 2), (True,), id="too_short"),
    pytest.param((0, 1, 2), (True, True, True, False), id="too_long"),
    pytest.param((), (True,), id="responses_without_items"),
  ],
)
def test_rejects_responses_that_do_not_fit_the_items(
  administered_items: tuple[int, ...], responses: tuple[bool, ...]
) -> None:
  """A context holds one response per administered item, so no strategy has to check.

  Before the check lived here, WLE spread a one-element vector over three items and MLE took its all-correct shortcut
  on it, both returning an estimate for answers that were never given.
  """
  with pytest.raises(ValueError, match="there must be one response per administered item"):
    _context(administered_items=administered_items, responses=responses)


def test_integer_responses_0_and_1_are_stored_as_bools() -> None:
  """An application that stores answers as 0 and 1 passes them as they are (ADR-0009), and strategies read bools.

  Fails if 1 and 0, or NumPy's integers and Booleans, are rejected, or are stored as given, so that a strategy that
  compares a response with ``is True`` misreads them.
  """
  context = _context(administered_items=(0, 1, 2), responses=(1, np.int64(0), np.True_))

  assert context.responses == (True, False, True)
  assert [type(response) for response in context.responses] == [bool, bool, bool]


@pytest.mark.parametrize(
  ("response", "error", "message"),
  [
    pytest.param(1.0, TypeError, "responses[1] is 1.0; responses must be bools or the integers 0 and 1", id="float"),
    pytest.param("1", TypeError, "responses[1] is '1'; responses must be bools or the integers 0 and 1", id="str"),
    pytest.param(None, TypeError, "responses[1] is None; responses must be bools or the integers 0 and 1", id="None"),
    pytest.param(2, ValueError, "responses[1] is 2; an integer response must be 0 or 1", id="two"),
  ],
)
def test_rejects_a_response_that_is_not_a_boolean_or_0_or_1(
  response: object, error: type[Exception], message: str
) -> None:
  """bool(1.0), bool("1") and bool(2) are all True, so converting what can be converted would misscore silently."""
  with pytest.raises(error, match=re.escape(message)):
    _context(administered_items=(0, 1), responses=(True, response))


def test_contexts_with_exposure_rates_compare_by_identity() -> None:
  """A generated __eq__ compares the exposure-rate arrays elementwise and raises; a generated __hash__ fails on them."""
  rates = np.array([0.1, 0.2, 0.3])
  context = _context(exposure_rates=rates)
  twin = dataclasses.replace(context, exposure_rates=rates.copy())

  assert context != twin
  assert len({context, twin}) == 2


@pytest.mark.parametrize("field", ["administered_items", "excluded_items"])
@pytest.mark.parametrize("index", [-1, 3])
def test_rejects_item_indices_outside_the_bank(field: str, index: int) -> None:
  """NumPy reads -1 as the last item, so excluding item -1 silently dropped item 2; 3 raised a bare IndexError."""
  administered, excluded = ((index,), ()) if field == "administered_items" else ((), (index,))

  with pytest.raises(ValueError, match=re.escape(f"{field}[0] is {index}; item indices must be between 0 and 2")):
    _context(administered_items=administered, responses=(True,) * len(administered), excluded_items=excluded)


def test_rejects_a_repeated_item() -> None:
  """A repeated item was accepted, and its response counted twice in the estimate (D29: the first silent case).

  The message names the item by its index and its name (ADR-0004); without the name it gave the index only.
  """
  with pytest.raises(
    ValueError,
    match=re.escape(
      "administered_items[2] is 1 ('item_1'); item indices must not repeat, and administered_items[0] is 1 too"
    ),
  ):
    _context(administered_items=(1, 0, 1), responses=(True, False, True))


@pytest.mark.parametrize("index", [3.7, np.float64(2.0)], ids=["fraction", "whole_float"])
def test_rejects_an_item_index_that_is_not_an_integer(index: float) -> None:
  """An index of 3.7 was accepted on a bank of five items, and NumPy read it as item 3 (D29: the second silent case)."""
  with pytest.raises(TypeError, match=re.escape(f"administered_items[1] is {index!r}; item indices must be integers")):
    StepContext(
      item_bank=ItemBank.generate(5, rng=0),
      rng=np.random.default_rng(0),
      administered_items=(0, index),  # ty: ignore[invalid-argument-type]
      responses=(True, False),
    )


def test_rejects_a_bool_as_an_item_index() -> None:
  """True is the integer 1 to Python, so a flag stored in place of an item would read as item 1."""
  with pytest.raises(TypeError, match=re.escape("excluded_items[0] is True; item indices must be integers, not bools")):
    _context(excluded_items=(True,))


def test_rejects_an_item_both_administered_and_excluded() -> None:
  """An item is given or withheld, never both; a selector told both would skip it, and the record is corrupt.

  The message names the item by its index and its name (ADR-0004); without the name it gave the index only.
  """
  with pytest.raises(
    ValueError,
    match=re.escape(
      "excluded_items[1] is 0 ('item_0'); an item cannot be both administered and excluded, and administered_items[1] "
      "is 0 too"
    ),
  ):
    _context(administered_items=(2, 0), responses=(True, True), excluded_items=(1, 0))


@pytest.mark.parametrize("theta", [math.nan, math.inf], ids=["nan", "inf"])
def test_rejects_an_estimate_that_is_not_finite(theta: float) -> None:
  """A NaN estimate was accepted, and changed the item a selector chose (D29: the third silent case)."""
  with pytest.raises(ValueError, match=re.escape(f"theta is {theta!r}; ability estimates must be finite")):
    _context(theta=theta)


@pytest.mark.parametrize("se", [-1.0, 0.0, math.nan], ids=["negative", "zero", "nan"])
def test_rejects_a_standard_error_that_is_not_positive(se: float) -> None:
  """A standard error of -1 was accepted, and a standard-error stopper ended the test at once (D29: the fourth case)."""
  with pytest.raises(ValueError, match=re.escape(f"se is {se!r}; standard errors must be positive")):
    _context(administered_items=(0,), responses=(True,), se=se)


@pytest.mark.parametrize(
  ("fields", "message"),
  [
    pytest.param({"theta": "0.5"}, "theta is '0.5'; ability estimates must be real numbers", id="theta"),
    pytest.param({"se": None}, "se is None; standard errors must be real numbers", id="se"),
  ],
)
def test_rejects_an_estimate_or_standard_error_that_is_not_a_number(fields: dict[str, object], message: str) -> None:
  """A theta stored as a string failed inside math.isfinite with "must be real number, not str", naming no input."""
  with pytest.raises(TypeError, match=re.escape(message)):
    _context(**fields)


def test_a_contexts_exposure_rates_cannot_be_changed() -> None:
  """A strategy that wrote into its context's rates, such as zeroing the capped ones, changed the caller's array and
  the rates of every later selection of the session.
  """
  rates = np.array([0.1, 0.2, 0.3])
  context = _context(exposure_rates=rates)

  assert context.exposure_rates is not None
  with pytest.raises(ValueError, match="read-only"):
    context.exposure_rates[0] = 0.0
  rates[1] = 0.9
  assert context.exposure_rates.tolist() == [0.1, 0.2, 0.3]


def test_rejects_exposure_counts_passed_as_rates() -> None:
  """Counts passed as rates were all at or above any cap, so a capped selector held back no item (the fifth case)."""
  with pytest.raises(ValueError, match=re.escape("exposure_rates[0] is 4.0; exposure rates must be between 0 and 1")):
    _context(exposure_rates=np.array([4, 0, 2]))


def test_rejects_exposure_rates_of_the_wrong_length() -> None:
  """Rates for another bank were accepted, and failed later with an IndexError deep in NumPy (the sixth case)."""
  with pytest.raises(
    ValueError,
    match=re.escape("exposure_rates has shape (2,); there must be one exposure rate per item of the bank (3)"),
  ):
    _context(exposure_rates=np.array([0.1, 0.2]))


@pytest.mark.parametrize(
  "rates", [np.array([True, False, True]), np.array(["0.1", "0.2", "0.3"])], ids=["booleans", "strings"]
)
def test_rejects_exposure_rates_that_are_not_numbers(rates: np.ndarray) -> None:
  """NumPy reads a Boolean mask as rates 0 and 1, and converts "0.1" to 0.1 when told to, so both would pass."""
  with pytest.raises(TypeError, match="exposure rates must be real numbers"):
    _context(exposure_rates=rates)


def test_replace_checks_the_changed_context() -> None:
  """A wrapper's dataclasses.replace builds a context the strategies trust, so it must be checked as built."""
  context = _context(administered_items=(0,), responses=(True,))

  with pytest.raises(ValueError, match="item indices must not repeat"):
    dataclasses.replace(context, excluded_items=(1, 1))


@pytest.mark.parametrize(
  ("theta", "se", "evaluations", "message"),
  [
    pytest.param(math.nan, 0.3, None, "theta is nan; ability estimates must be finite", id="nan_theta"),
    pytest.param(math.inf, 0.3, None, "theta is inf; ability estimates must be finite", id="infinite_theta"),
    pytest.param(0.0, math.nan, None, "se is nan; standard errors must be positive", id="nan_se"),
    pytest.param(0.0, -0.1, None, "se is -0.1; standard errors must be positive", id="negative_se"),
    pytest.param(0.0, 0.0, None, "se is 0.0; standard errors must be positive", id="zero_se"),
    pytest.param(0.0, 0.3, -1, "evaluations is -1; an evaluation count must not be negative", id="negative_count"),
  ],
)
def test_estimate_rejects_values_no_stopper_can_judge(
  theta: float, se: float, evaluations: int | None, message: str
) -> None:
  """A NaN error fails every comparison: a confidence-interval stopper then stops at once, and an SE stopper never.

  A zero error claims a perfectly known ability, which no finite information gives.
  """
  with pytest.raises(ValueError, match=re.escape(message)):
    Estimate(theta, se, evaluations)
