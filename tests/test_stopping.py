"""Tests for catsim.stopping module."""

from __future__ import annotations

import numpy as np
import pytest

from catsim import stopping
from catsim.item_bank import ItemBank
from catsim.stopping import ConfidenceIntervalStopper, MaxLengthStopper, StandardErrorStopper, StopRule


def identical_1pl_bank(n_items: int, difficulty: float) -> ItemBank:
  """Return `n_items` copies of a 1PL item; at theta = `difficulty`, n of them give a standard error of 2/sqrt(n)."""
  return ItemBank(np.tile([1.0, difficulty, 0.0, 1.0], (n_items, 1)))


class TestMaxLengthStopper:
  """Tests for MaxLengthStopper."""

  @pytest.mark.parametrize("value", [0, -5])
  def test_rejects_max_items_below_one(self, value: int) -> None:
    """A maximum below one item, which would end every test before it starts, is accepted instead of raising."""
    with pytest.raises(ValueError, match="max_items must be at least 1"):
      MaxLengthStopper(value)


class TestStandardErrorStopper:
  """Tests for StandardErrorStopper."""

  @pytest.mark.parametrize("value", [0.0, -0.3, float("nan")])
  def test_rejects_a_target_that_is_not_positive(self, value: float) -> None:
    """A target of zero or less, or NaN, which no standard error ever reaches, is accepted instead of raising."""
    with pytest.raises(ValueError, match="max_se must be positive"):
      StandardErrorStopper(value)

  def test_first_stops_when_the_standard_error_reaches_the_target(self) -> None:
    """A stopper that compares strictly (first stopping at 17), or reads the wrong items or theta, fails.

    At theta = b, n identical 1PL items give a standard error of 2/sqrt(n), which is 0.5 exactly at n = 16.
    """
    theta = 0.3
    item_bank = identical_1pl_bank(20, difficulty=theta)
    stopper = StandardErrorStopper(0.5)

    stops = [stopper.stop(item_bank=item_bank, administered_items=list(range(n)), theta=theta) for n in range(21)]

    assert stops == [n >= 16 for n in range(21)]


class TestConfidenceIntervalStopper:
  """Tests for ConfidenceIntervalStopper."""

  @pytest.mark.parametrize("confidence", [1.5, 0.0, 1.0])
  def test_rejects_a_confidence_outside_zero_and_one(self, confidence: float) -> None:
    """A confidence level of 0, 1 or more, which gives no usable interval, is accepted instead of raising."""
    with pytest.raises(ValueError, match="between 0 and 1"):
      ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=confidence)

  def test_rejects_unsorted_bounds(self) -> None:
    """Unsorted bounds, which do not describe consecutive bands, are accepted instead of raising."""
    with pytest.raises(ValueError, match="sorted"):
      ConfidenceIntervalStopper([2.0, 0.0, -2.0])

  def test_rejects_empty_bounds(self) -> None:
    """An empty list of bounds, which leaves a single band and so stops every test at once, is accepted."""
    with pytest.raises(ValueError, match="at least one"):
      ConfidenceIntervalStopper([])

  @pytest.mark.parametrize("theta", [-4.0, -2.0, 0.0, 2.0, 4.0])
  def test_first_stops_when_the_interval_fits_in_a_band(self, theta: float) -> None:
    """A stopper that misreads a band, outer or inner, or stops while a bound still splits the interval, fails.

    At theta = b, n identical 1PL items give a standard error of 2/sqrt(n), so the 95% interval is
    theta +/- 1.96 * 2/sqrt(n). The bounds [-3, -1, 1, 3] make five bands, and each estimate lies in a
    different one, 1 from its nearest bound, so the interval first fits in its band when
    3.92/sqrt(n) <= 1, at n = 16 (at n = 15 the half-width is 1.012).
    """
    item_bank = identical_1pl_bank(20, difficulty=theta)
    stopper = ConfidenceIntervalStopper([-3.0, -1.0, 1.0, 3.0])

    stops = [stopper.stop(item_bank=item_bank, administered_items=list(range(n)), theta=theta) for n in range(21)]

    assert stops == [n >= 16 for n in range(21)]

  @pytest.mark.parametrize("theta", [-3.0, -1.0, 1.0, 3.0])
  def test_never_stops_while_the_estimate_sits_on_a_bound(self, theta: float) -> None:
    """A stopper that skips a bound, merging the two bands around it, classifies an estimate that sits on it.

    The interval around an estimate always contains the estimate, so with the estimate on a bound it never
    fits in one band, however many items are administered.
    """
    item_bank = identical_1pl_bank(20, difficulty=theta)
    stopper = ConfidenceIntervalStopper([-3.0, -1.0, 1.0, 3.0])

    stops = [stopper.stop(item_bank=item_bank, administered_items=list(range(n)), theta=theta) for n in range(21)]

    assert not any(stops)


class TestStopRule:
  """Tests for StopRule."""

  def test_rejects_a_rule_without_stoppers(self) -> None:
    """A rule with no stoppers, which can never stop a test, is accepted instead of raising."""
    with pytest.raises(ValueError, match="at least one stopper"):
      StopRule()

  def test_rejects_negative_min_items(self) -> None:
    """A negative minimum length is accepted instead of raising."""
    with pytest.raises(ValueError, match="min_items must not be negative"):
      StopRule(MaxLengthStopper(10), min_items=-1)

  def test_rejects_a_child_that_is_not_a_stopper(self) -> None:
    """A number passed where a stopper belongs, as in StopRule(StandardErrorStopper(0.3), 40), is accepted.

    The rule would then fail on its first stop check, in the middle of a run, instead of when it is built.
    """
    with pytest.raises(TypeError, match="got int"):
      StopRule(StandardErrorStopper(0.3), 40)


def test_importing_every_public_stopper_collects_no_test_class(pytester: pytest.Pytester) -> None:
  """A public stopper named like a test class, as TestLengthStopper was, makes pytest warn wherever it is imported."""
  pytester.makepyfile(test_user_module=f"from catsim.stopping import {', '.join(stopping.__all__)}\n")

  result = pytester.runpytest("--collect-only", "-q")

  result.stdout.no_fnmatch_line("*PytestCollectionWarning*")
  assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
