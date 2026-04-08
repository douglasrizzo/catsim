"""Tests for catsim.plot module under the new architecture."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from catsim import plot
from catsim.plot import PlotType


class TestPlotType:
  """Tests for PlotType enum."""

  def test_plot_type_members_exist(self) -> None:
    """All documented plot types should exist."""
    assert PlotType.ICC is not None
    assert PlotType.IIC is not None
    assert PlotType.BOTH is not None


class TestItemCurve:
  """Tests for item_curve()."""

  @pytest.mark.parametrize("ptype", [PlotType.ICC, PlotType.IIC, PlotType.BOTH])
  def test_item_curve_returns_axes(self, ptype: PlotType) -> None:
    """Every item curve mode should return axes."""
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, ptype=ptype)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_curve_respects_title(self) -> None:
    """Custom titles should be applied."""
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, title="Test Item")
    assert ax.get_title() == "Test Item"
    plt.close("all")


class TestGen3dDatasetScatter:
  """Tests for gen3d_dataset_scatter()."""

  def test_gen3d_scatter_returns_3d_axes(self, item_bank) -> None:
    """3D dataset plots should return 3D axes."""
    ax = plot.gen3d_dataset_scatter(item_bank)
    assert isinstance(ax, Axes3D)
    plt.close("all")


class TestItemExposure:
  """Tests for item_exposure()."""

  def test_item_exposure_from_simulation_result(self, simulation_result) -> None:
    """SimulationResult should be accepted directly."""
    ax = plot.item_exposure(simulation=simulation_result)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_exposure_histogram(self, simulation_result) -> None:
    """Histogram mode should return axes."""
    ax = plot.item_exposure(simulation=simulation_result, hist=True)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_exposure_manual_inputs(self, item_bank) -> None:
    """Manual item_bank plus exposure_rates should be supported."""
    rates = np.linspace(0.0, 1.0, item_bank.n_items)
    ax = plot.item_exposure(item_bank=item_bank, exposure_rates=rates, par="b")
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_item_exposure_requires_input(self) -> None:
    """At least one plottable object must be provided."""
    with pytest.raises(ValueError, match="must be passed"):
      plot.item_exposure()


class TestTestProgress:
  """Tests for test_progress()."""

  def test_test_progress_from_simulation_result(self, simulation_result) -> None:
    """SimulationResult plus index should be accepted."""
    ax = plot.test_progress(simulation=simulation_result, index=0)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_test_progress_with_quality_curves(self, simulation_result) -> None:
    """Information, variance, SEE, and reliability options should render."""
    ax = plot.test_progress(
      simulation=simulation_result,
      index=0,
      info=True,
      var=True,
      see=True,
      reliability=True,
    )
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_test_progress_manual_inputs(self, item_bank) -> None:
    """Manual thetas and administered items should still be supported."""
    administered_items = item_bank.get_items([0, 1, 2])
    thetas = [0.0, 0.1, 0.2, 0.3]
    ax = plot.test_progress(thetas=thetas, administered_items=administered_items, true_theta=0.0)
    assert isinstance(ax, Axes)
    plt.close("all")

  def test_test_progress_requires_index_with_simulation(self, simulation_result) -> None:
    """SimulationResult calls require an index."""
    with pytest.raises(ValueError, match="index must be provided"):
      plot.test_progress(simulation=simulation_result)

  def test_test_progress_rejects_mismatched_lengths(self, item_bank) -> None:
    """Theta history and administered items must align."""
    administered_items = item_bank.get_items([0, 1, 2])
    with pytest.raises(ValueError, match="not the same"):
      plot.test_progress(thetas=[0.0, 0.1], administered_items=administered_items, true_theta=0.0)


@pytest.mark.slow
@pytest.mark.integration
def test_plots_integration(simulation_result) -> None:
  """Integration test for plotting from a completed simulation result."""
  for item in simulation_result.item_bank.items[:5]:
    plot.item_curve(item[0], item[1], item[2], item[3], title="Test plot", ptype=PlotType.ICC, max_info=False)
    plt.close("all")
    plot.item_curve(item[0], item[1], item[2], item[3], title="Test plot", ptype=PlotType.IIC, max_info=True)
    plt.close("all")
    plot.item_curve(item[0], item[1], item[2], item[3], title="Test plot", ptype=PlotType.BOTH, max_info=True)
    plt.close("all")

  plot.gen3d_dataset_scatter(simulation_result.item_bank)
  plt.close("all")
  plot.test_progress(simulation=simulation_result, index=0, info=True, see=True, reliability=True)
  plt.close("all")
  plot.item_exposure(simulation=simulation_result)
  plt.close("all")
  plot.item_exposure(simulation=simulation_result, par="a")
  plt.close("all")
  plot.item_exposure(simulation=simulation_result, par="b")
  plt.close("all")
  plot.item_exposure(simulation=simulation_result, par="c")
  plt.close("all")
  plot.item_exposure(simulation=simulation_result, par="d")
  plt.close("all")
  plot.item_exposure(simulation=simulation_result, hist=True)

  plt.close("all")
