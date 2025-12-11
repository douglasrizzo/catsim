"""Tests for catsim.plot module."""

import matplotlib.pyplot as plt
import pytest
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from catsim import plot
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import RandomInitializer
from catsim.item_bank import ItemBank
from catsim.plot import PlotType
from catsim.selection import MaxInfoSelector
from catsim.simulation import Simulator
from catsim.stopping import MinErrorStopper


class TestPlotType:
  """Tests for PlotType enum."""

  def test_plot_type_icc(self) -> None:
    """Test ICC plot type exists."""
    assert PlotType.ICC is not None

  def test_plot_type_iic(self) -> None:
    """Test IIC plot type exists."""
    assert PlotType.IIC is not None

  def test_plot_type_both(self) -> None:
    """Test BOTH plot type exists."""
    assert PlotType.BOTH is not None


class TestItemCurve:
  """Tests for item_curve function."""

  def test_item_curve_icc(self) -> None:
    """Test ICC plot."""
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, ptype=PlotType.ICC)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_curve_iic(self) -> None:
    """Test IIC plot."""
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, ptype=PlotType.IIC)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_curve_both(self) -> None:
    """Test BOTH plot type."""
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, ptype=PlotType.BOTH)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_curve_with_title(self) -> None:
    """Test with custom title."""
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, title="Test Item")
    assert ax.get_title() == "Test Item"
    plt.close("all")

  def test_item_curve_with_existing_axes(self) -> None:
    """Test plotting on existing axes."""
    _, ax_orig = plt.subplots()
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, ax=ax_orig)
    assert ax is ax_orig
    plt.close("all")

  def test_item_curve_with_max_info(self) -> None:
    """Test with max_info marker."""
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, ptype=PlotType.IIC, max_info=True)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_curve_with_figsize(self) -> None:
    """Test with custom figure size."""
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, figsize=(10, 6))
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_curve_various_params(self) -> None:
    """Test with various item parameters."""
    params = [
      (0.5, -1.0, 0.0, 1.0),
      (1.5, 0.0, 0.2, 1.0),
      (2.0, 1.0, 0.1, 0.95),
    ]
    for a, b, c, d in params:
      ax = plot.item_curve(a=a, b=b, c=c, d=d)
      assert isinstance(ax, Axes)
    plt.close("all")


class TestGen3dDatasetScatter:
  """Tests for gen3d_dataset_scatter function."""

  def test_gen3d_scatter_basic(self) -> None:
    """Test basic 3D scatter plot."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    ax = plot.gen3d_dataset_scatter(item_bank)
    assert isinstance(ax, Axes3D)
    plt.close("all")

  def test_gen3d_scatter_with_title(self) -> None:
    """Test with custom title."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    ax = plot.gen3d_dataset_scatter(item_bank, title="Test 3D Scatter")
    assert ax.get_title() == "Test 3D Scatter"
    plt.close("all")


class TestItemExposure:
  """Tests for item_exposure function."""

  @pytest.fixture
  def simulator(self) -> Simulator:
    """Create a simulator with completed simulation."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    sim = Simulator(
      item_bank,
      examinees=10,
      initializer=RandomInitializer(),
      selector=MaxInfoSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=MinErrorStopper(0.5, max_items=10),
    )
    sim.simulate()
    return sim

  def test_item_exposure_basic(self, simulator: Simulator) -> None:
    """Test basic item exposure plot."""
    ax = plot.item_exposure(simulator=simulator)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_exposure_histogram(self, simulator: Simulator) -> None:
    """Test histogram mode."""
    ax = plot.item_exposure(simulator=simulator, hist=True)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_exposure_by_parameter(self, simulator: Simulator) -> None:
    """Test exposure by different parameters."""
    for par in ["a", "b", "c", "d"]:
      ax = plot.item_exposure(simulator=simulator, par=par)
      assert isinstance(ax, Axes)
    plt.close("all")


class TestTestProgress:
  """Tests for test_progress function."""

  @pytest.fixture
  def simulator(self) -> Simulator:
    """Create a simulator with completed simulation."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    sim = Simulator(
      item_bank,
      examinees=5,
      initializer=RandomInitializer(),
      selector=MaxInfoSelector(),
      estimator=NumericalSearchEstimator(),
      stopper=MinErrorStopper(0.5, max_items=10),
    )
    sim.simulate()
    return sim

  def test_test_progress_basic(self, simulator: Simulator) -> None:
    """Test basic test progress plot."""
    ax = plot.test_progress(simulator=simulator, index=0)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_test_progress_with_title(self, simulator: Simulator) -> None:
    """Test with custom title."""
    ax = plot.test_progress(simulator=simulator, index=0, title="Test Progress")
    assert ax.get_title() == "Test Progress"
    plt.close("all")

  def test_test_progress_with_info(self, simulator: Simulator) -> None:
    """Test with information curve."""
    ax = plot.test_progress(simulator=simulator, index=0, info=True)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_test_progress_with_see(self, simulator: Simulator) -> None:
    """Test with standard error curve."""
    ax = plot.test_progress(simulator=simulator, index=0, see=True)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_test_progress_with_reliability(self, simulator: Simulator) -> None:
    """Test with reliability curve."""
    ax = plot.test_progress(simulator=simulator, index=0, reliability=True)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_test_progress_all_options(self, simulator: Simulator) -> None:
    """Test with all options enabled."""
    ax = plot.test_progress(
      simulator=simulator,
      index=0,
      title="Full Progress",
      info=True,
      see=True,
      reliability=True,
    )
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_test_progress_different_indices(self, simulator: Simulator) -> None:
    """Test with different examinee indices."""
    for i in range(min(3, len(simulator.examinees))):
      ax = plot.test_progress(simulator=simulator, index=i)
      assert isinstance(ax, Axes)
    plt.close("all")


@pytest.mark.slow
@pytest.mark.integration
def test_plots_integration() -> None:
  """Integration test for plot functionalities with full simulation."""
  initializer = RandomInitializer()
  selector = MaxInfoSelector()
  estimator = NumericalSearchEstimator()
  stopper = MinErrorStopper(0.5, max_items=20)
  s = Simulator(ItemBank.generate_item_bank(100), 10)
  s.simulate(initializer, selector, estimator, stopper, verbose=True)

  # Verify simulation produced results before plotting
  assert s.items is not None, "Simulation did not produce items"
  assert len(s.items) > 0, "No items in simulation"

  for item in s.items[0:10]:
    plot.item_curve(item[0], item[1], item[2], item[3], title="Test plot", ptype=PlotType.ICC, max_info=False)
    plot.item_curve(item[0], item[1], item[2], item[3], title="Test plot", ptype=PlotType.IIC, max_info=True)
    plot.item_curve(item[0], item[1], item[2], item[3], title="Test plot", ptype=PlotType.BOTH, max_info=True)
    plt.close("all")

  plot.gen3d_dataset_scatter(s.item_bank)
  plot.test_progress(
    title="Test progress",
    simulator=s,
    index=0,
    info=True,
    see=True,
    reliability=True,
  )
  plot.item_exposure(simulator=s)
  plot.item_exposure(simulator=s, par="a")
  plot.item_exposure(simulator=s, par="b")
  plot.item_exposure(simulator=s, par="c")
  plot.item_exposure(simulator=s, par="d")
  plot.item_exposure(simulator=s, hist=True)

  # close all plots after testing
  plt.close("all")
