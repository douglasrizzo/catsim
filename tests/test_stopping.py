"""Tests for catsim.stopping module."""

from __future__ import annotations

import numpy as np
import pytest

from catsim.engine import CatEngine, RunContext, SimulatedResponseProvider
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector
from catsim.simulation import SimulationRunner
from catsim.state import SessionStatus
from catsim.stopping import (
  BaseStopper,
  ConfidenceIntervalStopper,
  MinErrorStopper,
)
from catsim.stopping import (
  TestLengthStopper as LengthStopper,
)


class TestTestLengthStopperInit:
  """Tests for TestLengthStopper initialization."""

  def test_init_no_constraints(self) -> None:
    stopper = LengthStopper()
    assert stopper.min_items is None
    assert stopper.max_items is None

  def test_init_with_min_items(self) -> None:
    stopper = LengthStopper(min_items=5)
    assert stopper.min_items == 5
    assert stopper.max_items is None

  def test_init_with_max_items(self) -> None:
    stopper = LengthStopper(max_items=20)
    assert stopper.min_items is None
    assert stopper.max_items == 20

  def test_init_with_both_constraints(self) -> None:
    stopper = LengthStopper(min_items=5, max_items=20)
    assert stopper.min_items == 5
    assert stopper.max_items == 20

  @pytest.mark.parametrize("value", [0, -5])
  def test_init_invalid_min_items(self, value: int) -> None:
    with pytest.raises(ValueError, match="min_items must be positive"):
      LengthStopper(min_items=value)

  @pytest.mark.parametrize("value", [0, -5])
  def test_init_invalid_max_items(self, value: int) -> None:
    with pytest.raises(ValueError, match="max_items must be positive"):
      LengthStopper(max_items=value)

  def test_init_min_greater_than_max(self) -> None:
    with pytest.raises(ValueError, match=r"min_items.*cannot be greater than max_items"):
      LengthStopper(min_items=20, max_items=10)

  def test_init_min_equals_max(self) -> None:
    stopper = LengthStopper(min_items=10, max_items=10)
    assert stopper.min_items == 10
    assert stopper.max_items == 10


class TestTestLengthStopperStop:
  """Tests for TestLengthStopper.stop()."""

  def test_stop_max_items_reached(self) -> None:
    stopper = LengthStopper(max_items=5)
    item_bank = ItemBank.generate_item_bank(100)

    assert stopper.stop(item_bank=item_bank, administered_items=[0, 1, 2, 3, 4], theta=0.0) is True

  def test_stop_max_items_not_reached(self) -> None:
    stopper = LengthStopper(max_items=10)
    item_bank = ItemBank.generate_item_bank(100)

    assert stopper.stop(item_bank=item_bank, administered_items=[0, 1, 2], theta=0.0) is False

  def test_stop_min_items_not_reached(self) -> None:
    stopper = LengthStopper(min_items=5, max_items=20)
    item_bank = ItemBank.generate_item_bank(100)

    assert stopper.stop(item_bank=item_bank, administered_items=[0, 1, 2], theta=0.0) is False

  def test_stop_item_bank_exhausted(self) -> None:
    stopper = LengthStopper(max_items=100)
    item_bank = ItemBank.generate_item_bank(5)

    assert stopper.stop(item_bank=item_bank, administered_items=[0, 1, 2, 3, 4], theta=0.0) is True

  def test_stop_requires_item_bank(self) -> None:
    stopper = LengthStopper(max_items=10)

    with pytest.raises(ValueError, match="item_bank is required"):
      stopper.stop(item_bank=None, administered_items=[], theta=0.0)

  def test_stop_requires_administered_items(self) -> None:
    stopper = LengthStopper(max_items=10)
    item_bank = ItemBank.generate_item_bank(10)

    with pytest.raises(ValueError, match="administered_items is required"):
      stopper.stop(item_bank=item_bank, administered_items=None, theta=0.0)


class TestMinErrorStopperInit:
  """Tests for MinErrorStopper initialization."""

  def test_init_with_min_error(self) -> None:
    stopper = MinErrorStopper(0.3)
    assert stopper.min_error == pytest.approx(0.3)
    assert stopper.min_items is None
    assert stopper.max_items is None

  def test_init_with_all_params(self) -> None:
    stopper = MinErrorStopper(0.3, min_items=5, max_items=30)
    assert stopper.min_error == pytest.approx(0.3)
    assert stopper.min_items == 5
    assert stopper.max_items == 30

  @pytest.mark.parametrize("value", [0.0, -0.3])
  def test_init_invalid_min_error(self, value: float) -> None:
    with pytest.raises(ValueError, match="min_error must be positive"):
      MinErrorStopper(value)

  def test_str_representation(self) -> None:
    stopper = MinErrorStopper(0.3, min_items=5, max_items=30)
    string = str(stopper)
    assert "min_error=0.3" in string
    assert "min_items=5" in string
    assert "max_items=30" in string


class TestMinErrorStopperStop:
  """Tests for MinErrorStopper.stop()."""

  def test_stop_when_error_below_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("catsim.stopping.stopping.irt.see", lambda theta, items: 0.2)
    stopper = MinErrorStopper(0.3, max_items=50)
    item_bank = ItemBank.generate_item_bank(100, seed=42)

    result = stopper.stop(item_bank=item_bank, administered_items=list(range(20)), theta=0.0)

    assert result is True

  def test_stop_when_error_above_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("catsim.stopping.stopping.irt.see", lambda theta, items: 0.5)
    stopper = MinErrorStopper(0.3, max_items=50)
    item_bank = ItemBank.generate_item_bank(100, seed=42)

    result = stopper.stop(item_bank=item_bank, administered_items=[0, 1], theta=0.0)

    assert result is False

  def test_stop_respects_max_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("catsim.stopping.stopping.irt.see", lambda theta, items: 0.5)
    stopper = MinErrorStopper(0.01, max_items=5)
    item_bank = ItemBank.generate_item_bank(100, seed=42)

    result = stopper.stop(item_bank=item_bank, administered_items=[0, 1, 2, 3, 4], theta=0.0)

    assert result is True

  def test_stop_requires_theta(self) -> None:
    stopper = MinErrorStopper(0.3)
    item_bank = ItemBank.generate_item_bank(10, seed=42)

    with pytest.raises(ValueError, match="theta is required for MinErrorStopper"):
      stopper.stop(item_bank=item_bank, administered_items=[0], theta=None)


class TestConfidenceIntervalStopperInit:
  """Tests for ConfidenceIntervalStopper initialization."""

  def test_init_with_bounds(self) -> None:
    stopper = ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.95)
    assert stopper.min_items is None
    assert stopper.max_items is None

  def test_init_with_all_params(self) -> None:
    stopper = ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.90, min_items=5, max_items=30)
    assert stopper.min_items == 5
    assert stopper.max_items == 30

  @pytest.mark.parametrize("confidence", [1.5, 0.0, 1.0])
  def test_init_invalid_confidence(self, confidence: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
      ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=confidence)

  def test_init_unsorted_bounds_raises(self) -> None:
    with pytest.raises(ValueError, match="sorted"):
      ConfidenceIntervalStopper([2.0, 0.0, -2.0], confidence=0.95)

  def test_init_empty_bounds_raises(self) -> None:
    with pytest.raises(ValueError, match="at least one"):
      ConfidenceIntervalStopper([], confidence=0.95)

  def test_str_representation(self) -> None:
    stopper = ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.95, max_items=30)
    assert "ConfidenceIntervalStopper" in str(stopper)


class TestConfidenceIntervalStopperStop:
  """Tests for ConfidenceIntervalStopper.stop()."""

  def test_stop_respects_max_items(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("catsim.stopping.stopping.irt.confidence_interval", lambda theta, items, c: (-10.0, 10.0))
    stopper = ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.95, max_items=5)
    item_bank = ItemBank.generate_item_bank(100, seed=42)

    assert stopper.stop(item_bank=item_bank, administered_items=[0, 1, 2, 3, 4], theta=0.0) is True

  def test_stop_when_ci_falls_inside_middle_interval(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("catsim.stopping.stopping.irt.confidence_interval", lambda theta, items, c: (0.1, 0.9))
    stopper = ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.95, max_items=100)
    item_bank = ItemBank.generate_item_bank(100, seed=42)

    assert stopper.stop(item_bank=item_bank, administered_items=list(range(10)), theta=0.5) is True

  def test_stop_when_ci_crosses_interval_boundary(self, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("catsim.stopping.stopping.irt.confidence_interval", lambda theta, items, c: (-0.5, 0.5))
    stopper = ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.95, max_items=100)
    item_bank = ItemBank.generate_item_bank(100, seed=42)

    assert stopper.stop(item_bank=item_bank, administered_items=list(range(10)), theta=0.0) is False

  def test_stop_requires_theta(self) -> None:
    stopper = ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.95)
    item_bank = ItemBank.generate_item_bank(10, seed=42)

    with pytest.raises(ValueError, match="theta is required for ConfidenceIntervalStopper"):
      stopper.stop(item_bank=item_bank, administered_items=[0], theta=None)


class TestBaseStopperAbstract:
  """Tests for BaseStopper abstract base class."""

  def test_cannot_instantiate_base_stopper(self) -> None:
    with pytest.raises(TypeError):
      BaseStopper()  # type: ignore[abstract]


def test_test_length_stopper_with_engine_run() -> None:
  """A fixed-length stopper should stop a manual engine session at the configured length."""
  item_bank = ItemBank.generate_item_bank(30, seed=42)
  engine = CatEngine(
    FixedPointInitializer(0.0),
    MaxInfoSelector(),
    NumericalSearchEstimator(),
    LengthStopper(max_items=4),
  )
  context = RunContext(rng=np.random.default_rng(42))
  session = engine.start_session(session_id=0, item_bank=item_bank, context=context, true_theta=0.0)

  final_session = engine.run_session(session, item_bank, SimulatedResponseProvider(), context)

  assert final_session.status == SessionStatus.STOPPED
  assert final_session.stop_reason == "stop_rule"
  assert len(final_session.administered_item_ids) == 4


def test_min_error_stopper_with_simulation_runner(monkeypatch: pytest.MonkeyPatch) -> None:
  """SimulationRunner should respect MinErrorStopper through the shared engine path."""
  monkeypatch.setattr("catsim.stopping.stopping.irt.see", lambda theta, items: 0.1)
  item_bank = ItemBank.generate_item_bank(30, seed=42)
  runner = SimulationRunner(
    item_bank,
    FixedPointInitializer(0.0),
    MaxInfoSelector(),
    NumericalSearchEstimator(),
    MinErrorStopper(0.3, min_items=2, max_items=10),
    seed=42,
  )

  result = runner.run(5)

  assert all(session.status == SessionStatus.STOPPED for session in result.sessions)
  assert all(len(session.administered_item_ids) == 2 for session in result.sessions)
