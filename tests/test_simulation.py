"""Tests for catsim.simulation module."""

import numpy as np
import pytest

from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer, RandomInitializer
from catsim.item_bank import ItemBank
from catsim.selection import MaxInfoSelector, RandomSelector
from catsim.simulation import Simulator
from catsim.stopping import MinErrorStopper, TestLengthStopper


class TestSimulatorInit:
  """Tests for Simulator initialization."""

  def test_init_with_item_bank(self) -> None:
    """Test initialization with ItemBank object."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=10)
    assert simulator.item_bank is item_bank
    assert len(simulator.examinees) == 10

  def test_init_with_numpy_array(self) -> None:
    """Test initialization with numpy array (auto-converted to ItemBank)."""
    items = np.array([
      [1.0, 0.0, 0.0, 1.0],
      [1.2, 0.5, 0.0, 1.0],
    ])
    simulator = Simulator(items, examinees=5)
    assert isinstance(simulator.item_bank, ItemBank)
    assert len(simulator.item_bank) == 2

  def test_init_with_invalid_item_bank_raises(self) -> None:
    """Test that invalid item_bank type raises TypeError."""
    with pytest.raises(TypeError, match=r"must be an ItemBank or numpy\.ndarray"):
      Simulator([[1.0, 0.0, 0.0, 1.0]], examinees=5)  # type: ignore[arg-type]

  def test_init_with_examinees_int(self) -> None:
    """Test initialization with integer number of examinees."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=20)
    assert len(simulator.examinees) == 20

  def test_init_with_examinees_array(self) -> None:
    """Test initialization with explicit examinee abilities."""
    item_bank = ItemBank.generate_item_bank(50)
    abilities = [-1.0, 0.0, 1.0, 2.0]
    simulator = Simulator(item_bank, examinees=abilities)
    assert len(simulator.examinees) == 4
    assert np.allclose(simulator.examinees, abilities)

  def test_init_with_components(self) -> None:
    """Test initialization with all components."""
    item_bank = ItemBank.generate_item_bank(50)
    initializer = FixedPointInitializer(0.0)
    selector = MaxInfoSelector()
    estimator = NumericalSearchEstimator()
    stopper = TestLengthStopper(max_items=10)

    simulator = Simulator(
      item_bank,
      examinees=5,
      initializer=initializer,
      selector=selector,
      estimator=estimator,
      stopper=stopper,
    )

    assert simulator.initializer is initializer
    assert simulator.selector is selector
    assert simulator.estimator is estimator
    assert simulator.stopper is stopper

  def test_init_seed_reproducibility(self) -> None:
    """Test that same seed produces same examinee abilities."""
    item_bank = ItemBank.generate_item_bank(50)
    sim1 = Simulator(item_bank, examinees=10, seed=42)
    sim2 = Simulator(item_bank, examinees=10, seed=42)
    assert np.allclose(sim1.examinees, sim2.examinees)

  def test_init_different_seeds_produce_different_examinees(self) -> None:
    """Test that different seeds produce different examinee abilities."""
    item_bank = ItemBank.generate_item_bank(50)
    sim1 = Simulator(item_bank, examinees=10, seed=42)
    sim2 = Simulator(item_bank, examinees=10, seed=43)
    assert not np.allclose(sim1.examinees, sim2.examinees)


class TestSimulatorProperties:
  """Tests for Simulator properties."""

  def test_item_bank_property(self) -> None:
    """Test item_bank property."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=10)
    assert simulator.item_bank is item_bank

  def test_items_property(self) -> None:
    """Test items property returns item matrix."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=10)
    assert np.array_equal(simulator.items, item_bank.items)

  def test_examinees_property(self) -> None:
    """Test examinees property."""
    abilities = [0.0, 1.0, 2.0]
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=abilities)
    assert len(simulator.examinees) == 3

  def test_administered_items_initially_empty(self) -> None:
    """Test that administered_items is empty initially."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=5)
    assert len(simulator.administered_items) == 5
    assert all(len(items) == 0 for items in simulator.administered_items)

  def test_estimations_initially_empty(self) -> None:
    """Test that estimations is empty initially."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=5)
    assert len(simulator.estimations) == 5
    assert all(len(est) == 0 for est in simulator.estimations)

  def test_response_vectors_initially_empty(self) -> None:
    """Test that response_vectors is empty initially."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=5)
    assert len(simulator.response_vectors) == 5
    assert all(len(resp) == 0 for resp in simulator.response_vectors)

  def test_duration_initially_zero(self) -> None:
    """Test that duration is zero before simulation."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=5)
    assert simulator.duration == 0.0

  def test_bias_mse_rmse_initially_zero(self) -> None:
    """Test that evaluation metrics are zero initially."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=5)
    assert simulator.bias == 0.0
    assert simulator.mse == 0.0
    assert simulator.rmse == 0.0

  def test_rng_property(self) -> None:
    """Test that rng property returns numpy Generator."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=5)
    assert isinstance(simulator.rng, np.random.Generator)


class TestSimulatorSimulate:
  """Tests for Simulator.simulate() method."""

  def test_simulate_basic(self) -> None:
    """Test basic simulation execution."""
    item_bank = ItemBank.generate_item_bank(100)
    initializer = RandomInitializer()
    selector = RandomSelector()
    estimator = NumericalSearchEstimator()
    stopper = TestLengthStopper(max_items=10)

    simulator = Simulator(
      item_bank,
      examinees=5,
      initializer=initializer,
      selector=selector,
      estimator=estimator,
      stopper=stopper,
    )
    simulator.simulate()

    # Check that items were administered
    assert all(len(items) == 10 for items in simulator.administered_items)

  def test_simulate_with_components_passed(self) -> None:
    """Test simulation with components passed to simulate()."""
    item_bank = ItemBank.generate_item_bank(100)
    simulator = Simulator(item_bank, examinees=5)

    simulator.simulate(
      initializer=RandomInitializer(),
      selector=RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=TestLengthStopper(max_items=10),
    )

    assert all(len(items) == 10 for items in simulator.administered_items)

  def test_simulate_populates_estimations(self) -> None:
    """Test that simulation populates estimations."""
    item_bank = ItemBank.generate_item_bank(100)
    simulator = Simulator(
      item_bank,
      examinees=5,
      initializer=RandomInitializer(),
      selector=RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=TestLengthStopper(max_items=10),
    )
    simulator.simulate()

    # Each examinee should have estimations (initial + after each item)
    assert all(len(est) > 0 for est in simulator.estimations)

  def test_simulate_populates_response_vectors(self) -> None:
    """Test that simulation populates response vectors."""
    item_bank = ItemBank.generate_item_bank(100)
    simulator = Simulator(
      item_bank,
      examinees=5,
      initializer=RandomInitializer(),
      selector=RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=TestLengthStopper(max_items=10),
    )
    simulator.simulate()

    # Each examinee should have responses
    assert all(len(resp) == 10 for resp in simulator.response_vectors)
    # Responses should be boolean-like (either Python bool or numpy bool)
    assert all(isinstance(r, (bool, np.bool_)) for resp in simulator.response_vectors for r in resp)

  def test_simulate_updates_duration(self) -> None:
    """Test that simulation updates duration."""
    item_bank = ItemBank.generate_item_bank(100)
    simulator = Simulator(
      item_bank,
      examinees=5,
      initializer=RandomInitializer(),
      selector=RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=TestLengthStopper(max_items=10),
    )
    simulator.simulate()

    assert simulator.duration > 0.0

  def test_simulate_computes_metrics(self) -> None:
    """Test that simulation computes evaluation metrics."""
    item_bank = ItemBank.generate_item_bank(100)
    simulator = Simulator(
      item_bank,
      examinees=10,
      initializer=RandomInitializer(),
      selector=MaxInfoSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=MinErrorStopper(0.5, max_items=20),
    )
    simulator.simulate()

    # Metrics should be computed after simulation
    # These are floats, not zero (unless by very unlikely coincidence)
    assert isinstance(simulator.bias, float)
    assert isinstance(simulator.mse, float)
    assert isinstance(simulator.rmse, float)

  def test_simulate_latest_estimations(self) -> None:
    """Test latest_estimations property after simulation."""
    item_bank = ItemBank.generate_item_bank(100)
    simulator = Simulator(
      item_bank,
      examinees=5,
      initializer=RandomInitializer(),
      selector=RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=TestLengthStopper(max_items=10),
    )
    simulator.simulate()

    latest = simulator.latest_estimations
    assert len(latest) == 5
    # Each latest estimation should be the last in the list
    for i, est in enumerate(latest):
      assert est == simulator.estimations[i][-1]

  def test_simulate_with_verbose(self) -> None:
    """Test simulation with verbose output."""
    item_bank = ItemBank.generate_item_bank(100)
    simulator = Simulator(
      item_bank,
      examinees=3,
      initializer=RandomInitializer(),
      selector=RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=TestLengthStopper(max_items=5),
    )
    # Should not raise
    simulator.simulate(verbose=True)


class TestSimulatorExamineeSetter:
  """Tests for examinee setter functionality."""

  def test_examinees_setter_with_int(self) -> None:
    """Test setting examinees with integer."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=5)
    simulator.examinees = 10  # type: ignore[assignment]
    assert len(simulator.examinees) == 10

  def test_examinees_setter_with_list(self) -> None:
    """Test setting examinees with list."""
    item_bank = ItemBank.generate_item_bank(50)
    simulator = Simulator(item_bank, examinees=5)
    new_abilities = [0.0, 1.0, 2.0]
    simulator.examinees = new_abilities  # type: ignore[assignment]
    assert len(simulator.examinees) == 3
    assert np.allclose(simulator.examinees, new_abilities)


class TestSimulatorIntegration:
  """Integration tests for Simulator."""

  @pytest.mark.slow
  def test_full_simulation_with_fixed_abilities(self) -> None:
    """Test full simulation with known abilities."""
    item_bank = ItemBank.generate_item_bank(200)
    true_abilities = [-2.0, -1.0, 0.0, 1.0, 2.0]

    simulator = Simulator(
      item_bank,
      examinees=true_abilities,
      initializer=FixedPointInitializer(0.0),
      selector=MaxInfoSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=MinErrorStopper(0.3, max_items=30),
    )
    simulator.simulate()

    # Final estimates should be reasonably close to true abilities
    latest = simulator.latest_estimations
    for i, true_theta in enumerate(true_abilities):
      # Allow some error (within 1 standard deviation)
      assert abs(latest[i] - true_theta) < 1.5, f"Examinee {i}: expected ~{true_theta}, got {latest[i]}"

  @pytest.mark.slow
  def test_simulation_with_different_stoppers(self) -> None:
    """Test that different stoppers produce different test lengths."""
    item_bank = ItemBank.generate_item_bank(200)

    # Short test
    sim_short = Simulator(
      item_bank,
      examinees=5,
      initializer=RandomInitializer(),
      selector=RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=TestLengthStopper(max_items=5),
      seed=42,
    )
    sim_short.simulate()

    # Long test
    sim_long = Simulator(
      item_bank,
      examinees=5,
      initializer=RandomInitializer(),
      selector=RandomSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=TestLengthStopper(max_items=20),
      seed=42,
    )
    sim_long.simulate()

    # Verify different test lengths
    short_lengths = [len(items) for items in sim_short.administered_items]
    long_lengths = [len(items) for items in sim_long.administered_items]

    assert all(s == 5 for s in short_lengths)
    assert all(length == 20 for length in long_lengths)
