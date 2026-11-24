"""Tests for the contract every selector shares and for the helpers of catsim.selection.base."""

import importlib
import inspect
from collections.abc import Callable

import numpy as np
import pytest

from catsim import StepContext
from catsim.estimation import NumericalSearchEstimator
from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  BaseSelector,
  ClusterSelector,
  KLSelector,
  MaxInfoBBlockSelector,
  MaxInfoSelector,
  MaxInfoStratSelector,
  MEISelector,
  MLWISelector,
  ProgressiveSelector,
  ProportionalSelector,
  RandomesqueSelector,
  below_cap,
  remaining,
)
from tests.conftest import SELECTOR_FACTORIES
from tests.contexts import make_context


class TestRemaining:
  """Tests for remaining, the items a selector may still administer."""

  BANK = ItemBank.generate(6, rng=0)

  def test_returns_the_items_not_administered_in_ascending_order(self) -> None:
    """An administered item left in would be administered twice; a dropped one could never be chosen.

    With no items administered, a tuple index of () would select the whole mask and leave no candidate.
    """
    assert remaining(make_context(self.BANK, [4, 1])).tolist() == [0, 2, 3, 5]
    assert remaining(make_context(self.BANK)).tolist() == [0, 1, 2, 3, 4, 5]

  def test_leaves_out_the_excluded_items(self) -> None:
    """An excluded item left in could be handed back to the exposure-control wrapper that had just declined it."""
    assert remaining(make_context(self.BANK, [4], excluded_items=[0, 5])).tolist() == [1, 2, 3]

  @pytest.mark.parametrize(("administered", "excluded"), [([2, 0, 1, 5, 4, 3], []), ([2, 0, 1], [5, 4, 3])])
  def test_raises_when_every_item_is_administered_or_excluded(
    self, administered: list[int], excluded: list[int]
  ) -> None:
    """An empty candidate array would reach the scorer instead of ending the session with no item to give."""
    with pytest.raises(NoItemsAvailableError, match="no more items"):
      remaining(make_context(self.BANK, administered, excluded_items=excluded))


class TestBelowCap:
  """Tests for below_cap, the exposure filter of the capped selectors."""

  RATES = np.array([0.1, 0.9, 0.5, 0.2, 0.0, 0.6])

  def test_keeps_the_candidates_below_the_cap(self) -> None:
    """Item 2 sits exactly at the cap and is passed over; item 4 is below it but not a candidate."""
    assert below_cap(np.array([0, 2, 3, 5]), self.RATES, 0.5).tolist() == [0, 3]

  @pytest.mark.parametrize("rates", [RATES, None], ids=["every_candidate_at_or_over_the_cap", "no_rates"])
  def test_falls_back_to_every_candidate(self, rates: np.ndarray | None) -> None:
    """With no candidate below the cap, or no rates to compare, an empty result would leave no item to administer."""
    assert below_cap(np.array([1, 2, 5]), rates, 0.5).tolist() == [1, 2, 5]


CAPPED_SELECTORS: dict[str, Callable[[float], BaseSelector]] = {
  "MaxInfoSelector": lambda r_max: MaxInfoSelector(r_max=r_max),
  "KLSelector": lambda r_max: KLSelector(r_max=r_max),
  "MEISelector": lambda r_max: MEISelector(r_max=r_max),
  "MLWISelector": lambda r_max: MLWISelector(r_max=r_max),
  "ProgressiveSelector": lambda r_max: ProgressiveSelector(test_size=10, r_max=r_max),
  "ProportionalSelector": lambda r_max: ProportionalSelector(test_size=10, r_max=r_max),
  "ClusterSelector": lambda r_max: ClusterSelector(clusters=[0, 1], r_max=r_max),
}


@pytest.mark.parametrize("r_max", [0.0, -0.1, 1.5])
@pytest.mark.parametrize("make_selector", CAPPED_SELECTORS.values(), ids=CAPPED_SELECTORS.keys())
def test_capped_selectors_reject_r_max_outside_zero_to_one(
  make_selector: Callable[[float], BaseSelector], r_max: float
) -> None:
  """A cap of 0 used to be accepted: every item is at or above it, so the selector fell back to all items.

  It then capped nothing, exactly like r_max = 1.
  """
  with pytest.raises(ValueError, match=r"r_max must be in \(0, 1\]"):
    make_selector(r_max)


def test_capped_selectors_cover_every_selector_with_r_max() -> None:
  """A selector that takes r_max but is left out of CAPPED_SELECTORS could accept r_max = 0 unnoticed."""
  takes_r_max = {
    name
    for name, cls in vars(importlib.import_module("catsim.selection")).items()
    if isinstance(cls, type) and issubclass(cls, BaseSelector) and "r_max" in inspect.signature(cls).parameters
  }

  assert takes_r_max == set(CAPPED_SELECTORS)


@pytest.mark.parametrize("test_size", [0, -5])
@pytest.mark.parametrize(
  "selector_class",
  [
    AStratSelector,
    AStratBBlockSelector,
    MaxInfoStratSelector,
    MaxInfoBBlockSelector,
    ProgressiveSelector,
    ProportionalSelector,
  ],
)
def test_selectors_reject_test_size_below_one(selector_class: Callable[..., BaseSelector], test_size: int) -> None:
  """ProgressiveSelector(test_size=-5) used to construct and select by maximum information from the first item.

  ProportionalSelector(test_size=0) constructed too, and the stratified selectors rejected it only through a message
  about n_strata.
  """
  with pytest.raises(ValueError, match="test_size must be at least 1"):
    selector_class(test_size=test_size)


_N_ITEMS = 30


@pytest.mark.parametrize("make_selector", SELECTOR_FACTORIES.values(), ids=SELECTOR_FACTORIES.keys())
def test_select_takes_a_minimal_context(make_selector: Callable[[int], BaseSelector]) -> None:
  """Given only an item bank and a generator, every selector returns the bank's one item as a builtin int.

  A selector that still took the old arguments, or that needs a field the minimal context leaves at its default (no
  responses, no exposure rates), fails here, and numpy integers must not leak into session state.
  """
  item_bank = ItemBank.generate(1, rng=3)

  item_id = make_selector(1).select(StepContext(item_bank=item_bank, rng=np.random.default_rng(0)))

  assert type(item_id) is int
  assert item_id == 0


@pytest.mark.parametrize("make_selector", SELECTOR_FACTORIES.values(), ids=SELECTOR_FACTORIES.keys())
def test_select_never_returns_an_excluded_item(make_selector: Callable[[int], BaseSelector]) -> None:
  """A selector that drew its candidates without leaving out excluded_items would pick its first pick again.

  Each round excludes the earlier picks and selects with the same seed, so a selector that ignored them would repeat
  the first one.
  """
  item_bank = ItemBank.generate(_N_ITEMS, rng=3)
  selector = make_selector(_N_ITEMS)
  excluded: list[int] = []

  for _ in range(3):
    context = make_context(item_bank, theta=0.3, excluded_items=excluded, rng=np.random.default_rng(0))
    excluded.append(selector.select(context))

  assert len(set(excluded)) == 3


SELECTOR_REPRS = {
  "MaxInfoSelector": "MaxInfoSelector(r_max=0.5)",
  "UrrySelector": "UrrySelector()",
  "KLSelector": "KLSelector(window=2.0, r_max=0.5)",
  "IntervalInfoSelector": "IntervalInfoSelector(interval=0.5, confidence=0.9)",
  "RandomSelector": "RandomSelector()",
  "RandomesqueSelector": "RandomesqueSelector(bin_size=4)",
  "The54321Selector": "The54321Selector(initial_bin_size=3)",
  "LinearSelector": "LinearSelector(items=[3, 1, 2])",
  "ProgressiveSelector": "ProgressiveSelector(test_size=12, acceleration=2.0, r_max=0.5)",
  "ProportionalSelector": "ProportionalSelector(test_size=12, acceleration=-1.0, r_max=0.5)",
  "ClusterSelector": (
    "ClusterSelector(clusters=[0, 1, 0], method=ClusterMethod.CLUSTER_INFO, r_max=0.5, "
    "r_control=CapFallback.AGGRESSIVE)"
  ),
  "AStratSelector": "AStratSelector(test_size=12, n_strata=4)",
  "AStratBBlockSelector": "AStratBBlockSelector(test_size=12, n_strata=4)",
  "MaxInfoStratSelector": "MaxInfoStratSelector(test_size=12, n_strata=4)",
  "MaxInfoBBlockSelector": "MaxInfoBBlockSelector(test_size=12, n_strata=4)",
  "MEISelector": "MEISelector(estimator=NumericalSearchEstimator(tol=0.001, bounds=(-4.0, 4.0)), r_max=0.5)",
  "MEISelector-default-estimator": (
    "MEISelector(estimator=NumericalSearchEstimator(tol=1e-06, bounds=(-6.0, 6.0)), r_max=1.0)"
  ),
  "MLWISelector": "MLWISelector(n_nodes=21, bounds=(-4.0, 4.0), r_max=0.5)",
}


@pytest.mark.parametrize("source", SELECTOR_REPRS.values(), ids=SELECTOR_REPRS.keys())
def test_repr_is_the_constructor_call(source: str) -> None:
  """A selector's repr names every constructor argument with its value, so evaluating it rebuilds the selector.

  Selectors used to print display names, some without their arguments (MaxInfo, Randomesque, the stratified
  selectors) and others under names their constructors do not take (s=, k=, Q=).
  """
  namespace = {
    **vars(importlib.import_module("catsim.selection")),
    "NumericalSearchEstimator": NumericalSearchEstimator,
  }

  assert repr(eval(source, namespace)) == source


@pytest.mark.parametrize(
  ("numpy_source", "source"),
  [
    ("RandomesqueSelector(bin_size=np.int64(4))", "RandomesqueSelector(bin_size=4)"),
    ("The54321Selector(initial_bin_size=np.int64(3))", "The54321Selector(initial_bin_size=3)"),
    ("AStratSelector(test_size=np.int64(12), n_strata=np.int64(4))", "AStratSelector(test_size=12, n_strata=4)"),
    (
      "ProgressiveSelector(test_size=np.int64(12), acceleration=2.0, r_max=0.5)",
      "ProgressiveSelector(test_size=12, acceleration=2.0, r_max=0.5)",
    ),
  ],
)
def test_numpy_integer_arguments_are_stored_as_python_ints(numpy_source: str, source: str) -> None:
  """Integer arguments taken from a NumPy array are stored as Python ints, so the repr still evaluates.

  Fails if np.int64(4) is kept as given: the repr then reads bin_size=np.int64(4), naming a module that the
  catsim.selection namespace does not have.
  """
  namespace = {**vars(importlib.import_module("catsim.selection")), "np": np}

  assert repr(eval(numpy_source, namespace)) == source


@pytest.mark.parametrize(
  "build",
  [
    lambda: RandomesqueSelector(bin_size=2.5),  # ty: ignore[invalid-argument-type]
    lambda: AStratSelector(test_size=12.5),  # ty: ignore[invalid-argument-type]
    lambda: AStratSelector(test_size=12, n_strata=4.5),  # ty: ignore[invalid-argument-type]
  ],
  ids=["bin_size", "test_size", "n_strata"],
)
def test_integer_arguments_reject_fractions(build: Callable[[], BaseSelector]) -> None:
  """A fractional size is rejected at construction instead of being stored and used to slice or count items."""
  with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
    build()


def test_selector_tables_cover_every_concrete_selector() -> None:
  """Every concrete selector exported by catsim.selection runs through the context and repr tests above.

  A selector left out of SELECTOR_FACTORIES could reject a minimal context or select an excluded item, and one left
  out of SELECTOR_REPRS could hide an argument from its repr, without any test noticing.
  """
  exported = {
    name
    for name, cls in vars(importlib.import_module("catsim.selection")).items()
    if isinstance(cls, type) and issubclass(cls, BaseSelector) and not inspect.isabstract(cls)
  }

  assert exported == set(SELECTOR_FACTORIES)
  assert exported == {source.partition("(")[0] for source in SELECTOR_REPRS.values()}
