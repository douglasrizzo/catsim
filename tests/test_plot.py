"""Tests for catsim.plot module under the new architecture."""

from __future__ import annotations

import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import pytest
from mpl_toolkits.mplot3d import Axes3D

from catsim import ItemBank, plot
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
  ITEM = np.array([1.1669737841771746, -0.4321, 0.2, 1.0])
  A, B, C = ITEM[:3]

  @classmethod
  def _icc(cls, theta: np.ndarray) -> np.ndarray:
    """The item's characteristic curve in closed form, c + (1 - c) / (1 + exp(-a (theta - b)))."""
    return cls.C + (1 - cls.C) / (1 + np.exp(-cls.A * (theta - cls.B)))

  @classmethod
  def _information(cls, theta: np.ndarray) -> np.ndarray:
    """The three-parameter item's information in closed form, a^2 (P - c)^2 (1 - P) / ((1 - c)^2 P)."""
    p = cls._icc(theta)
    return cls.A**2 * (p - cls.C) ** 2 * (1 - p) / ((1 - cls.C) ** 2 * p)

  def test_item_curve_draws_the_items_characteristic_curve(self) -> None:
    """The curve spans four units around the difficulty and passes through (c + d) / 2 = 0.6 at it.

    Fails if the parameters are read in another order than [a, b, c, d], if the curve is centred elsewhere, or if the
    information is drawn instead.
    """
    ax = plot.item_curve(self.ITEM, kind=PlotType.ICC)
    (line,) = ax.get_lines()
    x, y = line.get_xdata(), line.get_ydata()

    np.testing.assert_allclose([x[0], x[-1]], [self.B - 4, self.B + 4])
    np.testing.assert_allclose(np.interp(self.B, x, y), 0.6)
    np.testing.assert_allclose(y, self._icc(x))
    assert ax.get_ylabel() == r"$P(\theta)$"

  def test_item_curve_marks_the_peak_of_the_information_in_the_curves_colour(self) -> None:
    """The information is a^2 / 6 at the difficulty, and the dot sits at Birnbaum's peak, in the curve's colour.

    For d = 1 the peak is at b + log((1 + sqrt(1 + 8c)) / 2) / a. Fails if the dot is off the peak or off the curve, or
    if it takes the next colour of the axes' cycle instead of the curve's.
    """
    ax = plot.item_curve(self.ITEM, kind=PlotType.IIC)
    curve, dot = ax.get_lines()
    x, y = curve.get_xdata(), curve.get_ydata()
    peak = self.B + np.log((1 + np.sqrt(1 + 8 * self.C)) / 2) / self.A

    np.testing.assert_allclose(np.interp(self.B, x, y), self.A**2 / 6)
    np.testing.assert_allclose(y, self._information(x))
    np.testing.assert_allclose(dot.get_xydata(), [[peak, self._information(peak)]])
    assert dot.get_color() == curve.get_color()

  @pytest.mark.parametrize(
    ("kind", "max_info", "expected"),
    [
      ("icc", True, [[r"$P(\theta)$"]]),
      ("iic", True, [[r"$I(\theta)$", "Maximum information"]]),
      ("iic", False, [[r"$I(\theta)$"]]),
      ("both", True, [[r"$P(\theta)$"], [r"$I(\theta)$", "Maximum information"]]),
      ("both", False, [[r"$P(\theta)$"], [r"$I(\theta)$"]]),
    ],
  )
  def test_item_curve_draws_the_curves_of_its_kind(self, kind: str, max_info: bool, expected: list[list[str]]) -> None:
    """Each kind draws its own curves, the dot only with max_info, and BOTH puts the information on a second y axis.

    Fails if a kind draws another curve, if max_info=False still marks the peak, or if BOTH draws both curves against
    one y axis, whose scale cannot suit a probability and an information at once.
    """
    ax = plot.item_curve(self.ITEM, kind=kind, max_info=max_info)
    assert [[line.get_label() for line in axes.get_lines()] for axes in ax.figure.axes] == expected

  @pytest.mark.parametrize("kind", list(PlotType))
  def test_item_curve_rounds_the_parameters(self, kind: PlotType) -> None:
    """The parameter box shows each parameter to three significant digits.

    Fails if a parameter is printed at full float precision (a = 1.1669737841771746), which made the
    box wider than the axes.
    """
    ax = plot.item_curve(self.ITEM, kind=kind)
    texts = [text.get_text() for axis in ax.figure.axes for text in axis.texts]
    assert texts == ["$a = 1.17$\n$b = -0.432$\n$c = 0.2$\n$d = 1$"]

  @pytest.mark.parametrize("kind", list(PlotType))
  def test_item_curve_parameter_box_stays_inside_the_axes_and_off_the_curves(self, kind: PlotType) -> None:
    """The parameter box lies within the axes and no plotted point falls under it.

    Fails if the box sticks out past the edge of the axes, or sits where a curve passes, as the
    information curve's tail does in the lower right corner.
    """
    ax = plot.item_curve(self.ITEM, kind=kind)
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

  @pytest.mark.parametrize(
    ("item", "kind", "message"),
    [
      (ITEM[:3], PlotType.ICC, r"one row of four parameters \[a, b, c, d\], got shape \(3,\)"),
      (np.array([ITEM, ITEM]), PlotType.ICC, r"got shape \(2, 4\)"),
      (ITEM, "pdf", "'pdf' is not a valid PlotType"),
    ],
    ids=["three-parameters", "two-items", "unknown-kind"],
  )
  def test_item_curve_rejects_bad_input_before_drawing(self, item: np.ndarray, kind: str, message: str) -> None:
    """An item that is not one row of four parameters, or an unknown kind, raises before any figure exists.

    Fails if a matrix of items draws one curve per item under a box that cannot show their parameters, or if a bad call
    leaves an empty figure open behind the error.
    """
    with pytest.raises(ValueError, match=message):
      plot.item_curve(item, kind=kind)
    assert plt.get_fignums() == []


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
  ITEM_BANK = ItemBank(
    np.array([
      [1.2, 0.5, 0.10, 0.95],
      [0.8, -1.0, 0.25, 0.90],
      [2.0, 2.0, 0.00, 1.00],
      [1.5, 0.0, 0.15, 0.85],
    ])
  )
  RATES = np.array([0.1, 0.2, 0.3, 0.4])

  @pytest.mark.parametrize(
    ("par", "xlabel", "expected_x", "expected_y"),
    [
      ("a", "Item discrimination", [0.8, 1.2, 1.5, 2.0], [0.2, 0.1, 0.4, 0.3]),
      ("b", "Item difficulty", [-1.0, 0.0, 0.5, 2.0], [0.2, 0.4, 0.1, 0.3]),
      ("c", "Item guessing parameter", [0.0, 0.10, 0.15, 0.25], [0.3, 0.1, 0.4, 0.2]),
      ("d", "Item upper asymptote", [0.85, 0.90, 0.95, 1.00], [0.4, 0.2, 0.1, 0.3]),
    ],
  )
  def test_item_exposure_plots_rates_against_the_parameter(
    self, par: str, xlabel: str, expected_x: list[float], expected_y: list[float]
  ) -> None:
    """The x data is the sorted parameter named on the axis, each paired with its item's exposure rate.

    Fails if the x axis carries ranks (0, 1, 2, ...) under a parameter label, if a rate is paired with
    another item's parameter, or if a parameter reads the wrong item bank column.
    """
    ax = plot.item_exposure(self.RATES, item_bank=self.ITEM_BANK, par=par)
    (line,) = ax.get_lines()
    assert (ax.get_xlabel(), ax.get_ylabel()) == (xlabel, "Exposure rate")
    np.testing.assert_allclose(line.get_xdata(), expected_x)
    np.testing.assert_allclose(line.get_ydata(), expected_y)

  def test_item_exposure_rates_alone_plot_in_item_order(self) -> None:
    """Exposure rates without an item bank plot against the item index when no parameter is asked for.

    Fails if the call demands an item bank it does not need, or reorders the items.
    """
    ax = plot.item_exposure(self.RATES)
    (line,) = ax.get_lines()
    assert ax.get_xlabel() == "Items"
    np.testing.assert_array_equal(line.get_xdata(), [0, 1, 2, 3])
    np.testing.assert_allclose(line.get_ydata(), self.RATES)

  def test_item_exposure_histogram(self) -> None:
    """The histogram counts the items in each exposure bin, with at least three bins.

    Rates [0, 0, 0.5, 1] in three equal bins over [0, 1] give counts [2, 1, 1]. Fails if the histogram
    is drawn from other data or with another number of bins.
    """
    ax = plot.item_exposure(np.array([0.0, 0.0, 0.5, 1.0]), hist=True)
    assert [patch.get_height() for patch in ax.patches] == [2, 1, 1]

  @pytest.mark.parametrize(
    ("rates", "kwargs", "message"),
    [
      (RATES, {"par": "b"}, "item_bank must be passed"),
      (RATES, {"item_bank": ITEM_BANK, "par": "b", "hist": True}, "pass par='b' or hist=True, not both"),
      (RATES[:3], {"item_bank": ITEM_BANK}, "3 values but the item bank has 4"),
      (RATES, {"item_bank": ITEM_BANK, "par": "e"}, "Unsupported parameter 'e'"),
      (np.array([RATES, RATES]), {}, r"one rate per item, got an array of shape \(2, 4\)"),
    ],
    ids=["par-without-item-bank", "par-with-hist", "length-mismatch", "unknown-par", "two-dimensional-rates"],
  )
  def test_item_exposure_rejects_inputs_that_cannot_give_a_true_plot(
    self, rates: np.ndarray, kwargs: dict, message: str
  ) -> None:
    """Inputs that cannot give a true plot raise a ValueError before any figure is created.

    Fails if a parameter is plotted without the item bank that holds it, if a histogram silently ignores the parameter
    it was given, if rates and items of different lengths are paired, if an unknown parameter or a matrix of rates is
    accepted, or if a figure is left open behind the error.
    """
    with pytest.raises(ValueError, match=message):
      plot.item_exposure(rates, **kwargs)
    assert plt.get_fignums() == []


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


@pytest.mark.parametrize(
  "draw",
  [
    lambda ax, title: plot.item_curve(TestItemCurve.ITEM, ax=ax, title=title),
    lambda ax, title: plot.item_exposure(TestItemExposure.RATES, ax=ax, title=title),
  ],
  ids=["item_curve", "item_exposure"],
)
def test_plot_draws_on_the_given_axes_and_returns_them(draw) -> None:
  """A plot function draws on the axes it is given, sets the title there and returns those axes.

  Fails if a function opens a figure of its own despite ``ax``, ignores ``title``, or returns other axes than the
  caller's.
  """
  figure, ax = plt.subplots()

  assert draw(ax, "A title") is ax
  assert ax.get_title() == "A title"
  assert ax.get_lines() or ax.patches
  assert plt.get_fignums() == [figure.number]
