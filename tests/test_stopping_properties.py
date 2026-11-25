"""Property-based tests for stopping rules."""

from __future__ import annotations

from dataclasses import dataclass

from hypothesis import assume, example, given, settings
from hypothesis import strategies as st

from catsim.compat import override
from catsim.item_bank import ItemBank
from catsim.stopping import BaseStopper, ConfidenceIntervalStopper, MaxLengthStopper, StopRule
from tests.contexts import make_context

STOP_SETTINGS = settings(max_examples=40)


@given(
  max_items=st.integers(min_value=1, max_value=20),
  bank_size=st.integers(min_value=8, max_value=40),
  administered_count=st.integers(min_value=0, max_value=35),
)
@example(max_items=20, bank_size=30, administered_count=19)
@example(max_items=20, bank_size=30, administered_count=20)
# the whole bank is administered before the cap: the engine, not the stopper, ends that session
@example(max_items=20, bank_size=10, administered_count=10)
@STOP_SETTINGS
def test_length_stopper_stops_once_max_items_are_administered(
  max_items: int,
  bank_size: int,
  administered_count: int,
) -> None:
  """A length stopper that stops one item early or late, or because the bank ran out, fails."""
  assume(administered_count <= bank_size)
  bank = ItemBank.generate(bank_size, rng=123)
  stopper = MaxLengthStopper(max_items)
  administered = list(range(administered_count))

  stopped = stopper.stop(make_context(bank, administered, theta=0.0))

  assert stopped is (administered_count >= max_items)


@dataclass(frozen=True)
class FixedStopper(BaseStopper):
  """Stopper whose criterion holds, or not, whatever the session; `label` tells equal stubs apart."""

  holds: bool
  label: int

  @override
  def stop(self, context) -> bool:
    return self.holds


RULE_BANK = ItemBank.generate(12, rng=0)


@given(
  holds=st.lists(st.booleans(), min_size=1, max_size=5),
  min_items=st.integers(min_value=0, max_value=10),
  administered_count=st.integers(min_value=0, max_value=12),
)
@STOP_SETTINGS
def test_stop_rule_stops_once_min_items_are_administered_and_any_stopper_holds(
  holds: list[bool],
  min_items: int,
  administered_count: int,
) -> None:
  """A rule that stops before min_items, ignores a stopper that holds, or stops when none holds fails."""
  rule = StopRule(*(FixedStopper(held, label) for label, held in enumerate(holds)), min_items=min_items)

  stopped = rule.stop(make_context(RULE_BANK, list(range(administered_count)), theta=0.0))

  assert stopped is (administered_count >= min_items and any(holds))


@given(
  holds=st.lists(st.booleans(), min_size=1, max_size=5),
  min_items=st.integers(min_value=0, max_value=10),
  administered_count=st.integers(min_value=0, max_value=12),
)
@STOP_SETTINGS
def test_stop_rule_satisfied_returns_exactly_the_stoppers_that_hold(
  holds: list[bool],
  min_items: int,
  administered_count: int,
) -> None:
  """satisfied() that drops a stopper that holds, adds one that does not, reorders them or applies min_items fails."""
  stoppers = [FixedStopper(held, label) for label, held in enumerate(holds)]
  rule = StopRule(*stoppers, min_items=min_items)

  satisfied = rule.satisfied(make_context(RULE_BANK, list(range(administered_count)), theta=0.0))

  assert satisfied == tuple(stopper for stopper in stoppers if stopper.holds)


CI_SETTINGS = settings(max_examples=25)


@given(
  params=st.integers(min_value=4, max_value=12).flatmap(
    lambda n: st.integers(min_value=0, max_value=999_999).map(lambda seed: ItemBank.generate(n, rng=seed))
  ),
  theta=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
  confidence=st.floats(0.5, 0.99, allow_nan=False, allow_infinity=False),
  bounds=st.lists(
    st.floats(-4.0, 4.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=5,
  ).map(sorted),
)
@CI_SETTINGS
def test_confidence_interval_stopper_never_stops_with_zero_items(
  params: ItemBank,
  theta: float,
  confidence: float,
  bounds: list[float],
) -> None:
  """A stopper that classifies an examinee before any item, with an infinite interval, fails."""
  bank = params
  assume(len(set(bounds)) == len(bounds))
  stopper = ConfidenceIntervalStopper(interval_bounds=bounds, confidence=confidence)
  assert stopper.stop(make_context(bank, [], theta=theta)) is False
