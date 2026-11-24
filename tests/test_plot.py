"""Tests for catsim.plot module under the new architecture."""

from __future__ import annotations

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D

from catsim import plot
from catsim.plot import PlotType


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
  """Run ``code`` in a fresh interpreter, whose module cache holds nothing the test session imported."""
  return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)


class TestOptionalMatplotlib:
  """Tests for matplotlib as the optional plot extra."""

  def test_importing_catsim_loads_no_plotting_module(self) -> None:
    """Importing catsim and its public modules other than catsim.plot loads neither catsim.plot nor matplotlib.

    Fails if any of them imports matplotlib, even indirectly, so that ``import catsim`` breaks for users who did not
    install the plot extra.
    """
    result = _run_python(
      "import sys\n"
      "import catsim, catsim.estimation, catsim.initialization, catsim.irt, catsim.metrics\n"
      "import catsim.selection, catsim.stopping, catsim.typing\n"
      "print(sorted(name for name in sys.modules if name.partition('.')[0] in {'matplotlib', 'mpl_toolkits'}"
      " or name == 'catsim.plot'))\n"
    )

    assert (result.returncode, result.stdout) == (0, "[]\n"), result.stderr

  def test_plot_without_matplotlib_names_the_extra(self) -> None:
    """Without matplotlib, importing catsim.plot raises an ImportError that says to install catsim[plot].

    Fails if the user gets matplotlib's bare ModuleNotFoundError, or an error that drops the original one.
    """
    result = _run_python("import sys\nsys.modules['matplotlib'] = None\nimport catsim.plot\n")

    assert result.returncode == 1
    assert "The above exception was the direct cause of the following exception" in result.stderr
    assert result.stderr.splitlines()[-1] == (
      "ImportError: catsim.plot needs matplotlib; install it with: pip install 'catsim[plot]'"
    )


class TestItemCurve:
  """Tests for item_curve()."""

  # A three-parameter item whose discrimination has more digits than the parameter box should show.
  ITEM = (1.1669737841771746, -0.4321, 0.2, 1.0)

  @pytest.mark.parametrize("ptype", list(PlotType))
  def test_item_curve_rounds_the_parameters(self, ptype: PlotType) -> None:
    """The parameter box shows each parameter to three significant digits.

    Fails if a parameter is printed at full float precision (a = 1.1669737841771746), which made the
    box wider than the axes.
    """
    ax = plot.item_curve(*self.ITEM, ptype=ptype)
    texts = [text.get_text() for axis in ax.figure.axes for text in axis.texts]
    assert texts == ["$a = 1.17$\n$b = -0.432$\n$c = 0.2$\n$d = 1$"]

  @pytest.mark.parametrize("ptype", list(PlotType))
  def test_item_curve_parameter_box_stays_inside_the_axes_and_off_the_curves(self, ptype: PlotType) -> None:
    """The parameter box lies within the axes and no plotted point falls under it.

    Fails if the box sticks out past the edge of the axes, or sits where a curve passes, as the
    information curve's tail does in the lower right corner.
    """
    ax = plot.item_curve(*self.ITEM, ptype=ptype)
    figure = ax.figure
    figure.canvas.draw()
    (annotation,) = [text for axis in figure.axes for text in axis.texts]
    box = annotation.get_bbox_patch().get_window_extent()
    axes_box = ax.get_window_extent()

    assert axes_box.x0 <= box.x0 < box.x1 <= axes_box.x1
    assert axes_box.y0 <= box.y0 < box.y1 <= axes_box.y1
    for axis in figure.axes:
      for line in axis.get_lines():
        assert box.count_contains(line.get_transform().transform(line.get_xydata())) == 0

  def test_item_curve_respects_title(self) -> None:
    """A title passed in is set on the axes.

    Fails if the title argument is ignored.
    """
    ax = plot.item_curve(a=1.0, b=0.0, c=0.0, d=1.0, title="Test Item")
    assert ax.get_title() == "Test Item"


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
    administered = np.concatenate([session.administered_items for session in simulation_result.sessions])
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

  # Two-parameter items (c = 0, d = 1): each item's information is a^2 P (1 - P), with
  # P = 1 / (1 + exp(-a (theta - b))).
  ITEMS = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 0.5, 0.0, 1.0], [0.8, -0.5, 0.0, 1.0]])
  THETAS = (0.0, 0.4, 0.7, 0.9)

  @classmethod
  def _info_after(cls, k: int) -> float:
    """Test information of the first k items at the estimate made after them, in closed form."""
    a, b = cls.ITEMS[:k, 0], cls.ITEMS[:k, 1]
    p = 1 / (1 + np.exp(-a * (cls.THETAS[k] - b)))
    return float(np.sum(a**2 * p * (1 - p)))

  @pytest.mark.parametrize(
    ("flag", "label", "from_info"),
    [
      ("info", r"$I(\theta)$", lambda info: info),
      ("var", "$Var$", lambda info: 1 / info),
      ("see", "$SEE$", lambda info: np.sqrt(1 / info)),
      ("reliability", "Reliability", lambda info: 1 - 1 / info),
    ],
  )
  def test_test_progress_pairs_each_estimate_with_the_items_before_it(self, flag: str, label: str, from_info) -> None:
    """The quality curve at x items uses the estimate after x responses and those x items, from x = 1.

    Fails if point x also counts item x + 1 (which made the plotted information drop at the last
    item), or if the curve starts at x = 0, where no item has been given.
    """
    ax = plot.test_progress(thetas=list(self.THETAS), administered_items=self.ITEMS, **{flag: True})
    (line,) = [line for line in ax.get_lines() if line.get_label() == label]
    info = np.array([self._info_after(k) for k in (1, 2, 3)])
    np.testing.assert_array_equal(line.get_xdata(), [1, 2, 3])
    np.testing.assert_allclose(line.get_ydata(), from_info(info))

  def test_test_progress_from_simulation_result(self, simulation_result) -> None:
    """The chosen session's estimates, item difficulties and true ability are plotted.

    Fails if another session is plotted, if the initial estimate is not drawn at x = 0, or if the difficulty of item k
    is not drawn at x = k.
    """
    session = simulation_result.sessions[2]
    ax = plot.test_progress(simulation=simulation_result, index=2)
    lines = {line.get_label(): line for line in ax.get_lines()}
    n_items = len(session.administered_items)

    np.testing.assert_array_equal(lines[r"$\hat{\theta}$"].get_xdata(), np.arange(n_items + 1))
    np.testing.assert_allclose(lines[r"$\hat{\theta}$"].get_ydata(), session.theta_history)
    np.testing.assert_array_equal(lines["Item difficulty"].get_xdata(), np.arange(1, n_items + 1))
    np.testing.assert_allclose(
      lines["Item difficulty"].get_ydata(), simulation_result.item_bank.items[session.administered_items, 1]
    )
    (true_theta_line,) = [collection for collection in ax.collections if collection.get_label() == r"$\theta$"]
    (true_theta_segment,) = true_theta_line.get_segments()
    np.testing.assert_allclose(true_theta_segment, [[0, session.true_theta], [n_items, session.true_theta]])

  def test_test_progress_requires_index_with_simulation(self, simulation_result) -> None:
    """A SimulationResult holds many sessions, so it needs an index.

    Fails if the call guesses a session instead of raising.
    """
    with pytest.raises(ValueError, match="index must be provided"):
      plot.test_progress(simulation=simulation_result)

  def test_test_progress_rejects_mismatched_lengths(self, item_bank) -> None:
    """The theta history holds one more value than there are items (the initial estimate).

    Fails if histories and items that do not line up are plotted anyway.
    """
    administered_items = item_bank.get_items([0, 1, 2])
    with pytest.raises(ValueError, match="not the same"):
      plot.test_progress(thetas=[0.0, 0.1], administered_items=administered_items, true_theta=0.0)
