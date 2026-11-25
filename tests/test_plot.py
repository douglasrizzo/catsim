"""Tests for catsim.plot module under the new architecture."""

from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.patches import Rectangle

from catsim import Estimate, ItemBank, Session, plot
from catsim.plot import PlotType
from catsim.typing import FloatArray

if TYPE_CHECKING:
  from matplotlib.lines import Line2D


class _ExposureOptions(TypedDict, total=False):
  """The keyword options of plot.item_exposure that the rejection tests pass."""

  item_bank: ItemBank
  par: str
  hist: bool


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
  def _icc(cls, theta: FloatArray | float) -> FloatArray | float:
    """The item's characteristic curve in closed form, c + (1 - c) / (1 + exp(-a (theta - b)))."""
    return cls.C + (1 - cls.C) / (1 + np.exp(-cls.A * (theta - cls.B)))

  @classmethod
  def _information(cls, theta: FloatArray | float) -> FloatArray | float:
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
    x, y = np.asarray(line.get_xdata()), np.asarray(line.get_ydata())

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
    x, y = np.asarray(curve.get_xdata()), np.asarray(curve.get_ydata())
    peak = self.B + np.log((1 + np.sqrt(1 + 8 * self.C)) / 2) / self.A

    np.testing.assert_allclose(np.interp(self.B, x, y), self.A**2 / 6)
    np.testing.assert_allclose(y, self._information(x))
    np.testing.assert_allclose(np.asarray(dot.get_xydata()), [[peak, self._information(peak)]])
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
    box_patch = annotation.get_bbox_patch()
    assert box_patch is not None
    box = box_patch.get_window_extent()
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
    np.testing.assert_allclose(np.asarray(line.get_xdata()), expected_x)
    np.testing.assert_allclose(np.asarray(line.get_ydata()), expected_y)

  def test_item_exposure_rates_alone_plot_in_item_order(self) -> None:
    """Exposure rates without an item bank plot against the item index when no parameter is asked for.

    Fails if the call demands an item bank it does not need, or reorders the items.
    """
    ax = plot.item_exposure(self.RATES)
    (line,) = ax.get_lines()
    assert ax.get_xlabel() == "Items"
    np.testing.assert_array_equal(np.asarray(line.get_xdata()), [0, 1, 2, 3])
    np.testing.assert_allclose(np.asarray(line.get_ydata()), self.RATES)

  def test_item_exposure_histogram(self) -> None:
    """The histogram counts the items in each exposure bin, with at least three bins.

    Rates [0, 0, 0.5, 1] in three equal bins over [0, 1] give counts [2, 1, 1]. Fails if the histogram
    is drawn from other data, with another number of bins, or with patches other than bars.
    """
    ax = plot.item_exposure(np.array([0.0, 0.0, 0.5, 1.0]), hist=True)
    heights = [patch.get_height() if isinstance(patch, Rectangle) else None for patch in ax.patches]
    assert heights == [2, 1, 1]

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
    self, rates: FloatArray, kwargs: _ExposureOptions, message: str
  ) -> None:
    """Inputs that cannot give a true plot raise a ValueError before any figure is created.

    Fails if a parameter is plotted without the item bank that holds it, if a histogram silently ignores the parameter
    it was given, if rates and items of different lengths are paired, if an unknown parameter or a matrix of rates is
    accepted, or if a figure is left open behind the error.
    """
    with pytest.raises(ValueError, match=message):
      plot.item_exposure(rates, **kwargs)
    assert plt.get_fignums() == []


class TestSessionProgress:
  """Tests for session_progress()."""

  # Three items whose difficulties differ from every estimate and standard error below.
  ITEM_BANK = ItemBank(np.array([[1.0, -0.5, 0.0, 1.0], [1.0, 0.3, 0.0, 1.0], [1.0, 1.2, 0.0, 1.0]]))
  # The initial estimate has an infinite standard error, as CatEngine.start records it.
  SESSION = Session(
    estimates=(Estimate(0.0, np.inf, 0), Estimate(0.8, 1.1, 3), Estimate(0.5, 0.9, 4), Estimate(0.6, 0.7, 5)),
    administered_items=(2, 0, 1),
    responses=(True, False, True),
  )

  @staticmethod
  def _lines(ax) -> dict[str, Line2D]:
    """Return the lines of the plot and of its twin axes, by label."""
    return {line.get_label(): line for axes in ax.figure.axes for line in axes.get_lines()}

  def test_session_progress_plots_the_recorded_estimates_at_each_test_length(self) -> None:
    """After k items the plot shows the k-th estimate and standard error the session recorded, and the k-th item.

    Fails if an estimate is paired with another number of items, if the standard error is recomputed rather than read
    from the session (what the stopper judged), if the initial estimate's infinite error is drawn, if difficulties are
    taken in bank order rather than administration order, or if the true ability given is not drawn.
    """
    ax = plot.session_progress(self.SESSION, self.ITEM_BANK, true_theta=0.4)
    lines = self._lines(ax)

    estimate = lines[r"Ability estimate $\hat{\theta}$"]
    np.testing.assert_array_equal(np.asarray(estimate.get_xdata()), [0, 1, 2, 3])
    np.testing.assert_allclose(np.asarray(estimate.get_ydata()), [0.0, 0.8, 0.5, 0.6])
    standard_error = lines["Standard error"]
    np.testing.assert_array_equal(np.asarray(standard_error.get_xdata()), [1, 2, 3])
    np.testing.assert_allclose(np.asarray(standard_error.get_ydata()), [1.1, 0.9, 0.7])
    assert standard_error.axes is not ax
    difficulty = lines["Item difficulty"]
    np.testing.assert_array_equal(np.asarray(difficulty.get_xdata()), [1, 2, 3])
    np.testing.assert_allclose(np.asarray(difficulty.get_ydata()), [1.2, -0.5, 0.3])
    np.testing.assert_allclose(np.asarray(lines[r"True ability $\theta$"].get_ydata()), [0.4, 0.4])

  def test_session_progress_band_spans_one_standard_error_around_each_estimate(self) -> None:
    """The band runs from estimate minus to estimate plus its standard error, from the first item on.

    Fails if the band uses another width, or reaches x = 0, where the initial estimate's standard error is infinite.
    """
    ax = plot.session_progress(self.SESSION, self.ITEM_BANK)
    (band,) = ax.collections
    vertices = {(float(x), round(float(y), 9)) for path in band.get_paths() for x, y in np.asarray(path.vertices)}

    assert vertices == {(1.0, -0.3), (1.0, 1.9), (2.0, -0.4), (2.0, 1.4), (3.0, -0.1), (3.0, 1.3)}

  def test_session_progress_without_true_ability_draws_no_true_ability_line(self) -> None:
    """A recorded or live examinee has no known true ability, so the plot draws no line for one.

    Fails if a plot without ``true_theta`` raises, or draws a line at some made-up ability.
    """
    live = Session(estimates=self.SESSION.estimates[:2], administered_items=(2,), responses=(True,))
    ax = plot.session_progress(live, self.ITEM_BANK)

    assert set(self._lines(ax)) == {r"Ability estimate $\hat{\theta}$", "Item difficulty", "Standard error"}

  def test_session_progress_ticks_whole_numbers_of_items(self) -> None:
    """The x axis counts items, so its ticks are whole numbers.

    Fails if a short test gets fractional ticks such as 0.5 or 2.5 items.
    """
    ax = plot.session_progress(self.SESSION, self.ITEM_BANK)
    ax.figure.canvas.draw()

    ticks = ax.get_xticks()
    assert len(ticks) > 1
    np.testing.assert_array_equal(ticks, np.round(ticks))

  def test_session_progress_rejects_items_outside_the_bank_before_drawing(self) -> None:
    """A session whose items are not in the item bank raises before any figure exists.

    A session cannot hold a negative index, but it is built without the bank, so it can hold one past it. Fails if a
    bad call leaves an empty figure open behind the error, or reads past the bank's parameters.
    """
    with pytest.raises(IndexError, match="not items of the bank"):
      plot.session_progress(self.SESSION, ItemBank(self.ITEM_BANK.items[:2]))
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
  "draw",
  [
    lambda ax, title: plot.item_curve(TestItemCurve.ITEM, ax=ax, title=title),
    lambda ax, title: plot.item_exposure(TestItemExposure.RATES, ax=ax, title=title),
    lambda ax, title: plot.session_progress(
      TestSessionProgress.SESSION, TestSessionProgress.ITEM_BANK, ax=ax, title=title
    ),
  ],
  ids=["item_curve", "item_exposure", "session_progress"],
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


def test_importing_every_plot_name_collects_no_test(pytester: pytest.Pytester) -> None:
  """A public catsim.plot name that pytest collects runs as a test in every test module that imports it.

  Fails if any catsim.plot name looks like a test to pytest, as test_progress did.
  """
  names = [
    name
    for name in vars(plot)
    if not name.startswith("_") and getattr(getattr(plot, name), "__module__", None) == "catsim.plot"
  ]
  pytester.makepyfile(test_user_module=f"from catsim.plot import {', '.join(names)}\n")

  result = pytester.runpytest("--collect-only", "-q")

  assert result.ret == pytest.ExitCode.NO_TESTS_COLLECTED
