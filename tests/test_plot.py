"""Tests for catsim.plot module under the new architecture."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from catsim import plot
from catsim.plot import PlotType


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

  # Each parameter column orders the four items differently.
  ITEMS = np.array([
    [1.2, 0.5, 0.10, 0.95],
    [0.8, -1.0, 0.25, 0.90],
    [2.0, 2.0, 0.00, 1.00],
    [1.5, 0.0, 0.15, 0.85],
  ])
  RATES = np.array([0.1, 0.2, 0.3, 0.4])

  @pytest.mark.parametrize(
    ("par", "xlabel", "expected_x", "expected_y"),
    [
      ("a", "Item discrimination", [0.8, 1.2, 1.5, 2.0], [0.2, 0.1, 0.4, 0.3]),
      ("b", "Item difficulty", [-1.0, 0.0, 0.5, 2.0], [0.2, 0.4, 0.1, 0.3]),
      ("c", "Item Guessing", [0.0, 0.10, 0.15, 0.25], [0.3, 0.1, 0.4, 0.2]),
      ("d", "Item upper asymptote", [0.85, 0.90, 0.95, 1.00], [0.4, 0.2, 0.1, 0.3]),
    ],
  )
  def test_item_exposure_plots_rates_against_the_parameter(
    self, par: str, xlabel: str, expected_x: list[float], expected_y: list[float]
  ) -> None:
    """The x data is the sorted parameter named on the axis, each paired with its item's exposure.

    Fails if the x axis carries ranks (0, 1, 2, ...) under a parameter label, if an exposure is
    paired with another item's parameter, or if a parameter reads the wrong item bank column.
    """
    ax = plot.item_exposure(item_bank=self.ITEMS, exposure_rates=self.RATES, par=par)
    (line,) = ax.get_lines()
    assert ax.get_xlabel() == xlabel
    np.testing.assert_allclose(line.get_xdata(), expected_x)
    np.testing.assert_allclose(line.get_ydata(), expected_y)

  def test_item_exposure_rates_alone_plot_in_item_order(self) -> None:
    """Exposure rates without an item bank plot against the item index when no parameter is asked for.

    Fails if the call demands an item bank it does not need, or reorders the items.
    """
    ax = plot.item_exposure(exposure_rates=self.RATES)
    (line,) = ax.get_lines()
    assert ax.get_xlabel() == "Items"
    np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2, 3])
    np.testing.assert_allclose(line.get_ydata(), self.RATES)

  def test_item_exposure_from_simulation_result(self, simulation_result) -> None:
    """A SimulationResult supplies the exposure of each item: its share of sessions that gave it.

    Fails if the plot draws anything other than the result's per-item exposure, in item order.
    """
    n_items = simulation_result.item_bank.n_items
    administered = np.concatenate([session.administered_item_ids for session in simulation_result.sessions])
    expected = np.bincount(administered, minlength=n_items) / len(simulation_result.sessions)

    ax = plot.item_exposure(simulation=simulation_result)
    (line,) = ax.get_lines()
    np.testing.assert_array_equal(line.get_xdata(), np.arange(n_items))
    np.testing.assert_allclose(line.get_ydata(), expected)

  def test_item_exposure_histogram(self) -> None:
    """The histogram counts the items in each exposure bin, with at least three bins.

    Rates [0, 0, 0.5, 1] in three equal bins over [0, 1] give counts [2, 1, 1]. Fails if the histogram
    is drawn from other data or with another number of bins.
    """
    ax = plot.item_exposure(exposure_rates=np.array([0.0, 0.0, 0.5, 1.0]), hist=True)
    assert [patch.get_height() for patch in ax.patches] == [2, 1, 1]

  @pytest.mark.parametrize(
    ("kwargs", "message"),
    [
      ({}, "Exposure data must be passed"),
      ({"item_bank": ITEMS}, "Exposure data must be passed"),
      ({"exposure_rates": RATES, "par": "b"}, "item_bank must be passed"),
      ({"item_bank": ITEMS, "exposure_rates": RATES[:3]}, "3 values but the item bank has 4"),
      ({"item_bank": ITEMS, "exposure_rates": RATES, "par": "e"}, "Unsupported parameter 'e'"),
    ],
    ids=["nothing", "item-bank-alone", "par-without-item-bank", "length-mismatch", "unknown-par"],
  )
  def test_item_exposure_rejects_incomplete_inputs(self, kwargs: dict, message: str) -> None:
    """Inputs that cannot give a true plot raise a ValueError before any figure is created.

    Fails if an item bank alone plots zero exposure for every item, if a parameter is plotted
    without the item bank that holds it, if rates and items of different lengths are paired, if an
    unknown parameter is accepted, or if a figure is left open behind the error.
    """
    with pytest.raises(ValueError, match=message):
      plot.item_exposure(**kwargs)
    assert plt.get_fignums() == []

  @pytest.mark.parametrize("extra", ["item_bank", "exposure_rates"])
  def test_item_exposure_rejects_data_beside_a_simulation(self, simulation_result, extra: str) -> None:
    """A SimulationResult already holds the item bank and exposure, so passing either again is an error.

    Fails if the extra argument is silently ignored in favour of the simulation's own data.
    """
    kwargs = {"item_bank": self.ITEMS, "exposure_rates": self.RATES}
    with pytest.raises(ValueError, match="not both"):
      plot.item_exposure(simulation=simulation_result, **{extra: kwargs[extra]})


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
