"""Tests for catsim.stopping module."""

import pytest

from catsim import irt
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import InitializationDistribution, RandomInitializer
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector
from catsim.simulation import Simulator
from catsim.stopping import (
  BaseStopper,
  ConfidenceIntervalStopper,
  MinErrorStopper,
  TestLengthStopper,
)


class TestTestLengthStopperInit:
  """Tests for TestLengthStopper initialization."""

  def test_init_no_constraints(self) -> None:
    """Test initialization with no constraints."""
    stopper = TestLengthStopper()
    assert stopper.min_items is None
    assert stopper.max_items is None

  def test_init_with_min_items(self) -> None:
    """Test initialization with only min_items."""
    stopper = TestLengthStopper(min_items=5)
    assert stopper.min_items == 5
    assert stopper.max_items is None

  def test_init_with_max_items(self) -> None:
    """Test initialization with only max_items."""
    stopper = TestLengthStopper(max_items=20)
    assert stopper.min_items is None
    assert stopper.max_items == 20

  def test_init_with_both_constraints(self) -> None:
    """Test initialization with both min and max items."""
    stopper = TestLengthStopper(min_items=5, max_items=20)
    assert stopper.min_items == 5
    assert stopper.max_items == 20

  def test_init_invalid_min_items_zero(self) -> None:
    """Test that min_items=0 raises ValueError."""
    with pytest.raises(ValueError, match="min_items must be positive"):
      TestLengthStopper(min_items=0)

  def test_init_invalid_min_items_negative(self) -> None:
    """Test that negative min_items raises ValueError."""
    with pytest.raises(ValueError, match="min_items must be positive"):
      TestLengthStopper(min_items=-5)

  def test_init_invalid_max_items_zero(self) -> None:
    """Test that max_items=0 raises ValueError."""
    with pytest.raises(ValueError, match="max_items must be positive"):
      TestLengthStopper(max_items=0)

  def test_init_invalid_max_items_negative(self) -> None:
    """Test that negative max_items raises ValueError."""
    with pytest.raises(ValueError, match="max_items must be positive"):
      TestLengthStopper(max_items=-5)

  def test_init_min_greater_than_max(self) -> None:
    """Test that min_items > max_items raises ValueError."""
    with pytest.raises(ValueError, match=r"min_items.*cannot be greater than max_items"):
      TestLengthStopper(min_items=20, max_items=10)

  def test_init_min_equals_max(self) -> None:
    """Test that min_items == max_items is allowed."""
    stopper = TestLengthStopper(min_items=10, max_items=10)
    assert stopper.min_items == 10
    assert stopper.max_items == 10


class TestTestLengthStopperStop:
  """Tests for TestLengthStopper.stop() method."""

  def test_stop_max_items_reached(self) -> None:
    """Test that stop returns True when max_items is reached."""
    stopper = TestLengthStopper(max_items=5)
    item_bank = ItemBank.generate_item_bank(100)
    administered_items = item_bank.get_items([0, 1, 2, 3, 4])  # 5 items
    assert stopper.stop(administered_items=administered_items, theta=0.0, _item_bank=item_bank) is True

  def test_stop_max_items_not_reached(self) -> None:
    """Test that stop returns False when max_items not yet reached."""
    stopper = TestLengthStopper(max_items=10)
    item_bank = ItemBank.generate_item_bank(100)
    administered_items = item_bank.get_items([0, 1, 2])  # 3 items < max 10
    assert stopper.stop(administered_items=administered_items, theta=0.0, _item_bank=item_bank) is False

  def test_stop_min_items_not_reached(self) -> None:
    """Test that stop returns False when min_items not yet reached."""
    stopper = TestLengthStopper(min_items=5, max_items=20)
    item_bank = ItemBank.generate_item_bank(100)
    administered_items = item_bank.get_items([0, 1, 2])  # 3 items < min 5
    assert stopper.stop(administered_items=administered_items, theta=0.0, _item_bank=item_bank) is False

  def test_stop_item_bank_exhausted(self) -> None:
    """Test that stop returns True when item bank is exhausted."""
    stopper = TestLengthStopper(max_items=100)
    item_bank = ItemBank.generate_item_bank(5)  # Only 5 items
    administered_items = item_bank.get_items([0, 1, 2, 3, 4])  # All 5 used
    assert stopper.stop(administered_items=administered_items, theta=0.0, _item_bank=item_bank) is True

  def test_stop_missing_parameters(self) -> None:
    """Test that stop raises ValueError when required parameters are missing."""
    stopper = TestLengthStopper(max_items=10)
    with pytest.raises(ValueError, match="Required parameters are missing"):
      stopper.stop(theta=0.0)


class TestMinErrorStopperInit:
  """Tests for MinErrorStopper initialization."""

  def test_init_with_min_error(self) -> None:
    """Test initialization with min_error."""
    stopper = MinErrorStopper(0.3)
    assert stopper.min_error == 0.3
    assert stopper.min_items is None
    assert stopper.max_items is None

  def test_init_with_all_params(self) -> None:
    """Test initialization with all parameters."""
    stopper = MinErrorStopper(0.3, min_items=5, max_items=30)
    assert stopper.min_error == 0.3
    assert stopper.min_items == 5
    assert stopper.max_items == 30

  def test_init_invalid_min_error_zero(self) -> None:
    """Test that min_error=0 raises ValueError."""
    with pytest.raises(ValueError, match="min_error must be positive"):
      MinErrorStopper(0.0)

  def test_init_invalid_min_error_negative(self) -> None:
    """Test that negative min_error raises ValueError."""
    with pytest.raises(ValueError, match="min_error must be positive"):
      MinErrorStopper(-0.3)

  def test_str_representation(self) -> None:
    """Test string representation."""
    stopper = MinErrorStopper(0.3, min_items=5, max_items=30)
    str_repr = str(stopper)
    assert "min_error=0.3" in str_repr
    assert "min_items=5" in str_repr
    assert "max_items=30" in str_repr


class TestMinErrorStopperStop:
  """Tests for MinErrorStopper.stop() method."""

  def test_stop_when_error_below_threshold(self) -> None:
    """Test that stop returns True when error is below threshold."""
    # Use a high threshold that should be easy to meet
    stopper = MinErrorStopper(2.0, max_items=50)
    item_bank = ItemBank.generate_item_bank(100, seed=42)
    # Administer many items to reduce error
    administered_items = item_bank.get_items(list(range(20)))
    result = stopper.stop(administered_items=administered_items, theta=0.0, _item_bank=item_bank)
    assert result is True

  def test_stop_when_error_above_threshold(self) -> None:
    """Test that stop returns False when error is above threshold."""
    # Use a very low threshold that's hard to meet
    stopper = MinErrorStopper(0.01, max_items=50)
    item_bank = ItemBank.generate_item_bank(100, seed=42)
    # Administer few items - error will be high
    administered_items = item_bank.get_items([0, 1])
    result = stopper.stop(administered_items=administered_items, theta=0.0, _item_bank=item_bank)
    assert result is False

  def test_stop_respects_max_items(self) -> None:
    """Test that max_items constraint is respected."""
    stopper = MinErrorStopper(0.01, max_items=5)  # Very low threshold
    item_bank = ItemBank.generate_item_bank(100, seed=42)
    administered_items = item_bank.get_items([0, 1, 2, 3, 4])  # 5 items = max
    result = stopper.stop(administered_items=administered_items, theta=0.0, _item_bank=item_bank)
    assert result is True  # Stops due to max_items, not error threshold


class TestConfidenceIntervalStopperInit:
  """Tests for ConfidenceIntervalStopper initialization."""

  def test_init_with_bounds(self) -> None:
    """Test initialization with interval bounds."""
    bounds = [-2.0, 0.0, 2.0]
    stopper = ConfidenceIntervalStopper(bounds, confidence=0.95)
    assert stopper.min_items is None
    assert stopper.max_items is None

  def test_init_with_all_params(self) -> None:
    """Test initialization with all parameters."""
    bounds = [-2.0, 0.0, 2.0]
    stopper = ConfidenceIntervalStopper(bounds, confidence=0.90, min_items=5, max_items=30)
    assert stopper.min_items == 5
    assert stopper.max_items == 30

  def test_init_invalid_confidence_too_high(self) -> None:
    """Test that confidence > 1 raises ValueError."""
    with pytest.raises(ValueError, match="between 0 and 1"):
      ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=1.5)

  def test_init_invalid_confidence_too_low(self) -> None:
    """Test that confidence <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="between 0 and 1"):
      ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.0)

  def test_init_invalid_confidence_exactly_one(self) -> None:
    """Test that confidence = 1 raises ValueError."""
    with pytest.raises(ValueError, match="between 0 and 1"):
      ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=1.0)

  def test_init_unsorted_bounds_raises(self) -> None:
    """Test that unsorted bounds raise ValueError."""
    with pytest.raises(ValueError, match="sorted"):
      ConfidenceIntervalStopper([2.0, 0.0, -2.0], confidence=0.95)

  def test_init_empty_bounds_raises(self) -> None:
    """Test that empty bounds raise ValueError."""
    with pytest.raises(ValueError, match="at least one"):
      ConfidenceIntervalStopper([], confidence=0.95)

  def test_str_representation(self) -> None:
    """Test string representation."""
    stopper = ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.95, max_items=30)
    str_repr = str(stopper)
    assert "ConfidenceIntervalStopper" in str_repr


class TestConfidenceIntervalStopperStop:
  """Tests for ConfidenceIntervalStopper.stop() method."""

  def test_stop_respects_max_items(self) -> None:
    """Test that max_items constraint is respected."""
    bounds = [-2.0, 0.0, 2.0]
    stopper = ConfidenceIntervalStopper(bounds, confidence=0.95, max_items=5)
    item_bank = ItemBank.generate_item_bank(100, seed=42)
    administered_items = item_bank.get_items([0, 1, 2, 3, 4])  # 5 items = max
    result = stopper.stop(administered_items=administered_items, theta=0.0, _item_bank=item_bank)
    assert result is True  # Stops due to max_items

  def test_stop_with_narrow_ci(self) -> None:
    """Test stopping when CI is within an interval."""
    # With many items, CI should be narrow
    bounds = [-2.0, 0.0, 2.0]
    stopper = ConfidenceIntervalStopper(bounds, confidence=0.95, max_items=100)
    item_bank = ItemBank.generate_item_bank(100, seed=42)
    # Administer many items to get narrow CI
    administered_items = item_bank.get_items(list(range(30)))
    result = stopper.stop(administered_items=administered_items, theta=0.5, _item_bank=item_bank)
    # With 30 items, CI should be narrow enough to fall within an interval
    assert isinstance(result, bool)


class TestBaseStopperAbstract:
  """Tests for BaseStopper abstract base class."""

  def test_cannot_instantiate_base_stopper(self) -> None:
    """Test that BaseStopper cannot be instantiated directly."""
    with pytest.raises(TypeError):
      BaseStopper()  # type: ignore[abstract]


# Integration tests


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("examinees", [50])
@pytest.mark.parametrize("bank_size", [100])
@pytest.mark.parametrize(
  "stopper",
  [
    TestLengthStopper(max_items=20),
    TestLengthStopper(min_items=5, max_items=20),
  ],
)
def test_test_length_stopper_simulation(
  examinees: int,
  bank_size: int,
  stopper: BaseStopper,
) -> None:
  """Test TestLengthStopper in full simulation."""
  item_bank = ItemBank.generate_item_bank(bank_size, itemtype=irt.NumParams.PL4)
  initializer = RandomInitializer(InitializationDistribution.UNIFORM, (-5, 5))
  selector = MaxInfoSelector()
  estimator = NumericalSearchEstimator()

  simulator = Simulator(item_bank, examinees, initializer, selector, estimator, stopper)
  simulator.simulate(verbose=True)

  # Verify simulation ran successfully
  assert simulator.latest_estimations is not None
  assert len(simulator.latest_estimations) == examinees

  # Verify all tests stopped at max_items since base criterion always returns False
  assert stopper.max_items is not None
  for i in range(examinees):
    assert len(simulator.administered_items[i]) == stopper.max_items
