"""Tests for catsim.plot module under the new architecture."""

from __future__ import annotations

import itertools
import math
import subprocess
import sys
from collections.abc import Callable
from typing import TYPE_CHECKING, TypedDict

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import same_color, to_rgb
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import AxLine
from matplotlib.patches import Polygon, Rectangle

from catsim import Estimate, ItemBank, Session, SimulationResult, StopReason, plot
from catsim.plot import PlotType
from catsim.typing import FloatArray

if TYPE_CHECKING:
  import numpy.typing as npt
  from matplotlib.axes import Axes
  from matplotlib.lines import Line2D


class _ExposureOptions(TypedDict, total=False):
  """The keyword options of plot.item_exposure that the rejection tests pass."""

  par: str
  hist: bool


def _x_tick_labels(ax: Axes) -> list[tuple[float, str, bool]]:
  """Return each x tick of a drawn plot as its position, its label and whether the label is read as mathtext."""
  ax.figure.canvas.draw()
  return [(float(label.get_position()[0]), label.get_text(), label.get_parse_math()) for label in ax.get_xticklabels()]


def _check_ticks_count(ax: Axes, names: tuple[str, ...]) -> None:
  """Check that the drawn x axis ticks whole numbers labelled as numbers, none with an item's name."""
  ticks = _x_tick_labels(ax)
  positions = [position for position, _, _ in ticks]
  assert len(positions) > 1
  np.testing.assert_array_equal(positions, np.round(positions))
  assert not {label for _, label, _ in ticks} & set(names)


def _shows_grid(ax: Axes) -> bool:
  """Return whether the drawn axes show any grid line."""
  ax.figure.canvas.draw()
  return any(line.get_visible() for line in [*ax.xaxis.get_gridlines(), *ax.yaxis.get_gridlines()])


def _ended(items: tuple[int, ...], thetas: tuple[float, ...], ses: tuple[float, ...] | None = None) -> Session:
  """Return an ended session that gave ``items``, all answered correctly, with estimate ``thetas[k]`` after k + 1 items.

  The standard errors are ``ses``, or 0.5 each; the initial estimate is 0 with an infinite standard error.
  """
  errors = (0.5,) * len(thetas) if ses is None else ses
  estimates = (Estimate(0.0, math.inf, 0), *(Estimate(theta, se, 1) for theta, se in zip(thetas, errors, strict=True)))
  return Session(
    administered_items=items, responses=(True,) * len(items), estimates=estimates, stop_reason=StopReason.STOPPER
  )


def _run(
  sessions: list[Session], true_thetas: list[float] | FloatArray | None = None, n_items: int = 12
) -> SimulationResult:
  """Return a result of ``sessions`` on a generated bank of ``n_items`` items, its proposals its administrations."""
  counts = np.bincount([item for session in sessions for item in session.administered_items], minlength=n_items)
  return SimulationResult(
    item_bank=ItemBank.generate(n_items, rng=0),
    sessions=tuple(sessions),
    proposal_counts=counts,
    duration=0.0,
    true_thetas=None if true_thetas is None else np.asarray(true_thetas, dtype=float),
  )


def _logistic(x: float) -> float:
  """The logistic function, 1 / (1 + exp(-x)): a two-parameter item's characteristic curve at a(theta - b) = x."""
  return 1 / (1 + math.exp(-x))


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

  # A three-parameter item whose discrimination has more digits than the parameter box should show, as item 1 of a
  # bank whose item 0 differs in every parameter. Its name has a lone dollar sign, which mathtext would pair up.
  ITEM = np.array([1.1669737841771746, -0.4321, 0.2, 1.0])
  ITEM_BANK = ItemBank(np.array([[2.0, 1.5, 0.0, 0.9], ITEM]), names=["other", "fee $5"])
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
    """The curve spans four units around the item's difficulty and passes through (c + d) / 2 = 0.6 at it.

    Fails if the parameters are read in another order than [a, b, c, d] or from another item of the bank, if the curve
    is centred elsewhere, or if the information is drawn instead.
    """
    ax = plot.item_curve(1, self.ITEM_BANK, kind=PlotType.ICC)
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
    ax = plot.item_curve(1, self.ITEM_BANK, kind=PlotType.IIC)
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
    ax = plot.item_curve(1, self.ITEM_BANK, kind=kind, max_info=max_info)
    assert [[line.get_label() for line in axes.get_lines()] for axes in ax.figure.axes] == expected

  def test_item_curve_draws_both_curves_in_the_first_two_colours_of_the_style(self) -> None:
    """With both curves, the characteristic curve and its axis take the style's first colour, information the second.

    Fails if the information takes another colour of the cycle, such as the fourth, which is grey in matplotlib's ggplot
    and Petroff cycles, if an axis's label or tick labels differ from its curve's colour, or if a colour is kept as a
    reference such as "C1", which matplotlib looks up when the figure is drawn: a figure drawn after its
    ``plt.style.context`` block, as a notebook draws it at the end of the cell, would then take the default style's
    colours.
    """
    with plt.style.context({"axes.prop_cycle": plt.cycler(color=["#112233", "#445566", "#778899", "#aabbcc"])}):
      ax = plot.item_curve(1, self.ITEM_BANK, kind=PlotType.BOTH)
    (second,) = [axes for axes in ax.figure.axes if axes is not ax]
    (characteristic,) = ax.get_lines()
    information, dot = second.get_lines()

    for artist, colour in [
      (characteristic, "#112233"),
      (ax.yaxis.label, "#112233"),
      (information, "#445566"),
      (dot, "#445566"),
      (second.yaxis.label, "#445566"),
      *[(label, "#112233") for label in ax.get_yticklabels()],
      *[(label, "#445566") for label in second.get_yticklabels()],
    ]:
      assert same_color(artist.get_color(), colour), artist

  @pytest.mark.parametrize("kind", list(PlotType))
  def test_item_curve_box_names_the_item_and_rounds_its_parameters(self, kind: PlotType) -> None:
    """The box gives the item's name, its dollar sign escaped, then each parameter to three significant digits.

    Fails if the box leaves the item unnamed or names another item, if a parameter is printed at full float precision
    (a = 1.1669737841771746), which made the box wider than the axes, or if the name's lone dollar sign is left to
    mathtext, which then finds an odd number of them and prints the parameters' as literal dollar signs.
    """
    ax = plot.item_curve(1, self.ITEM_BANK, kind=kind)
    texts = [text.get_text() for axis in ax.figure.axes for text in axis.texts]
    assert texts == ["fee \\$5\n$a = 1.17$\n$b = -0.432$\n$c = 0.2$\n$d = 1$"]

  @pytest.mark.parametrize("kind", list(PlotType))
  def test_item_curve_parameter_box_stays_inside_the_axes_and_off_the_curves(self, kind: PlotType) -> None:
    """The parameter box lies within the axes and no plotted point falls under it.

    Fails if the box sticks out past the edge of the axes, or sits where a curve passes, as the
    information curve's tail does in the lower right corner.
    """
    ax = plot.item_curve(1, self.ITEM_BANK, kind=kind)
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
    ("item", "kind", "error", "message"),
    [
      (-1, PlotType.ICC, ValueError, r"item is -1; item indices must be between 0 and 1"),
      (2, PlotType.ICC, ValueError, r"item is 2; item indices must be between 0 and 1"),
      (1.0, PlotType.ICC, TypeError, r"item is 1.0; item indices must be integers"),
      (ITEM, PlotType.ICC, TypeError, r"item is array\(.*\); item indices must be integers"),
      (1, "pdf", ValueError, "'pdf' is not a valid PlotType"),
    ],
    ids=["negative-item", "item-past-the-bank", "float-item", "parameter-row", "unknown-kind"],
  )
  def test_item_curve_rejects_bad_input_before_drawing(
    self, item: object, kind: str, error: type[Exception], message: str
  ) -> None:
    """An index that is not an item of the bank, a non-integer, or an unknown kind raises before any figure exists.

    Fails if a negative index draws an item counted from the end of the bank, if a float such as 1.0 or a row of
    parameters, the argument item_curve once took, is read as an index, or if a bad call leaves an empty figure open
    behind the error.
    """
    with pytest.raises(error, match=message):
      plot.item_curve(item, self.ITEM_BANK, kind=kind)  # ty: ignore[invalid-argument-type]
    assert plt.get_fignums() == []


class TestItemExposure:
  """Tests for item_exposure()."""

  # Each parameter column orders the four items differently, and the names are neither generated nor sorted.
  ITEM_BANK = ItemBank(
    np.array([
      [1.2, 0.5, 0.10, 0.95],
      [0.8, -1.0, 0.25, 0.90],
      [2.0, 2.0, 0.00, 1.00],
      [1.5, 0.0, 0.15, 0.85],
    ]),
    names=["vocab_9", "$x$ + 1", "algebra", "reading"],
  )
  RATES = np.array([0.1, 0.2, 0.3, 0.4])

  @pytest.mark.parametrize(
    ("par", "xlabel", "expected_x"),
    [
      ("a", "Item discrimination", [1.2, 0.8, 2.0, 1.5]),
      ("b", "Item difficulty", [0.5, -1.0, 2.0, 0.0]),
      ("c", "Item guessing parameter", [0.10, 0.25, 0.00, 0.15]),
      ("d", "Item upper asymptote", [0.95, 0.90, 1.00, 0.85]),
    ],
  )
  def test_item_exposure_plots_each_item_as_a_point_at_its_parameter(
    self, par: str, xlabel: str, expected_x: list[float]
  ) -> None:
    """Each item is a point at its value of the parameter named on the axis, paired with its own rate, and unjoined.

    Fails if the x axis carries ranks under a parameter label, if a rate is paired with another item's parameter, if a
    parameter reads the wrong item bank column, or if a line joins the points, which draws a zigzag through items that
    are not related to their neighbours.
    """
    ax = plot.item_exposure(self.RATES, self.ITEM_BANK, par=par)
    (points,) = ax.get_lines()

    assert (ax.get_xlabel(), ax.get_ylabel()) == (xlabel, "Exposure rate")
    np.testing.assert_allclose(np.asarray(points.get_xdata()), expected_x)
    np.testing.assert_allclose(np.asarray(points.get_ydata()), self.RATES)
    assert points.get_linestyle() == "None"

  def test_item_exposure_sorts_the_items_from_the_most_exposed(self) -> None:
    """Without a parameter, the rates fall from the most to the least exposed item, and each tick names its item.

    Rates 0.1, 0.2, 0.3 and 0.4 in bank order come out as 0.4, 0.3, 0.2 and 0.1, at positions 1 to 4. Fails if the
    items keep their bank order, joined by a line that relates items that are not related, if a tick names another
    item than the one drawn there, or if a name of two dollar signs is read as mathtext.
    """
    ax = plot.item_exposure(self.RATES, self.ITEM_BANK)
    (line,) = ax.get_lines()

    assert ax.get_xlabel() == "Items, from the most to the least exposed"
    np.testing.assert_array_equal(np.asarray(line.get_xdata()), [1, 2, 3, 4])
    np.testing.assert_allclose(np.asarray(line.get_ydata()), [0.4, 0.3, 0.2, 0.1])
    assert _x_tick_labels(ax) == [
      (1.0, "reading", False),
      (2.0, "algebra", False),
      (3.0, "$x$ + 1", False),
      (4.0, "vocab_9", False),
    ]

  @pytest.mark.parametrize("n_items", [30, 31])
  def test_item_exposure_names_items_while_they_fit_on_the_axis(self, n_items: int) -> None:
    """Thirty items get one named tick each, equal rates in bank order; a bank of 31 ticks whole numbers instead.

    Fails if a large bank crowds one name per item onto the axis, where they overlap, or if a bank of 30 loses its
    names.
    """
    item_bank = ItemBank.generate(n_items, rng=0)
    ax = plot.item_exposure(np.zeros(n_items), item_bank)

    if n_items == 30:
      assert [label for _, label, _ in _x_tick_labels(ax)] == list(item_bank.names)
    else:
      _check_ticks_count(ax, item_bank.names)

  def test_item_exposure_histogram(self) -> None:
    """The histogram counts the items in each exposure bin, with at least three bins.

    Rates [0, 0, 0.5, 1] in three equal bins over [0, 1] give counts [2, 1, 1]. Fails if the histogram
    is drawn from other data, with another number of bins, or with patches other than bars.
    """
    ax = plot.item_exposure(np.array([0.0, 0.0, 0.5, 1.0]), self.ITEM_BANK, hist=True)
    heights = [patch.get_height() if isinstance(patch, Rectangle) else None for patch in ax.patches]
    assert heights == [2, 1, 1]

  @pytest.mark.parametrize(
    ("rates", "kwargs", "message"),
    [
      (RATES, {"par": "b", "hist": True}, "pass par='b' or hist=True, not both"),
      (RATES[:3], {}, r"has shape \(3,\); there must be one exposure rate per item of the bank \(4\)"),
      (RATES, {"par": "e"}, "Unsupported parameter 'e'"),
      (np.array([RATES, RATES]), {}, r"has shape \(2, 4\); there must be one exposure rate per item"),
    ],
    ids=["par-with-hist", "length-mismatch", "unknown-par", "two-dimensional-rates"],
  )
  def test_item_exposure_rejects_inputs_that_cannot_give_a_true_plot(
    self, rates: FloatArray, kwargs: _ExposureOptions, message: str
  ) -> None:
    """Inputs that cannot give a true plot raise a ValueError before any figure is created.

    Fails if a histogram silently ignores the parameter it was given, if rates and items of different lengths are
    paired, if an unknown parameter or a matrix of rates is accepted, or if a figure is left open behind the error.
    """
    with pytest.raises(ValueError, match=message):
      plot.item_exposure(rates, self.ITEM_BANK, **kwargs)
    assert plt.get_fignums() == []


class TestSessionProgress:
  """Tests for session_progress()."""

  # Three items whose difficulties differ from every estimate and standard error below.
  ITEM_BANK = ItemBank(
    np.array([[1.0, -0.5, 0.0, 1.0], [1.0, 0.3, 0.0, 1.0], [1.0, 1.2, 0.0, 1.0]]), names=["easy", "medium", "hard"]
  )
  # The initial estimate has an infinite standard error, as CatEngine.start records it.
  SESSION = Session(
    estimates=(Estimate(0.0, np.inf, 0), Estimate(0.8, 1.1, 3), Estimate(0.5, 0.9, 4), Estimate(0.6, 0.7, 5)),
    administered_items=(2, 0, 1),
    responses=(True, False, True),
  )

  @staticmethod
  def _lines(ax) -> dict[str, Line2D]:
    """Return the lines of the plot, by label."""
    return {line.get_label(): line for line in ax.get_lines()}

  def test_session_progress_plots_the_recorded_estimates_at_each_test_length(self) -> None:
    """After k items the plot shows the k-th estimate the session recorded, and the k-th item's difficulty.

    Fails if an estimate is paired with another number of items, if difficulties are taken in bank order rather than
    administration order, or if the true ability given is not drawn.
    """
    ax = plot.session_progress(self.SESSION, self.ITEM_BANK, true_theta=0.4)
    lines = self._lines(ax)

    estimate = lines[r"Ability estimate $\hat{\theta}$"]
    np.testing.assert_array_equal(np.asarray(estimate.get_xdata()), [0, 1, 2, 3])
    np.testing.assert_allclose(np.asarray(estimate.get_ydata()), [0.0, 0.8, 0.5, 0.6])
    difficulty = lines["Item difficulty"]
    np.testing.assert_array_equal(np.asarray(difficulty.get_xdata()), [1, 2, 3])
    np.testing.assert_allclose(np.asarray(difficulty.get_ydata()), [1.2, -0.5, 0.3])
    np.testing.assert_allclose(np.asarray(lines[r"True ability $\theta$"].get_ydata()), [0.4, 0.4])

  def test_session_progress_band_spans_one_standard_error_around_each_estimate(self) -> None:
    """The band runs from estimate minus to estimate plus the standard error the session recorded, from the first item.

    Fails if the band uses another width, if the standard error is recomputed rather than read from the session (what
    the stopper judged), or if the band reaches x = 0, where the initial estimate's standard error is infinite.
    """
    ax = plot.session_progress(self.SESSION, self.ITEM_BANK)
    (band,) = ax.collections
    vertices = {(float(x), round(float(y), 9)) for path in band.get_paths() for x, y in np.asarray(path.vertices)}

    assert vertices == {(1.0, -0.3), (1.0, 1.9), (2.0, -0.4), (2.0, 1.4), (3.0, -0.1), (3.0, 1.3)}

  def test_session_progress_keeps_the_styles_colours_once_the_style_ends(self) -> None:
    """The estimate and its band take the style's first colour, and the difficulties its second.

    Fails if a colour is kept as a reference such as "C1", which matplotlib looks up when the figure is drawn: a figure
    drawn after its ``plt.style.context`` block would then mix the default style's colours with the style's.
    """
    with plt.style.context({"axes.prop_cycle": plt.cycler(color=["#112233", "#445566", "#778899"])}):
      ax = plot.session_progress(self.SESSION, self.ITEM_BANK)
    lines = self._lines(ax)
    (band,) = ax.collections

    for artist, colour in [
      (lines[r"Ability estimate $\hat{\theta}$"], "#112233"),
      (lines["Item difficulty"], "#445566"),
    ]:
      assert same_color(artist.get_color(), colour), artist
    np.testing.assert_allclose(np.asarray(band.get_facecolor())[0, :3], to_rgb("#112233"))

  def test_session_progress_without_true_ability_draws_no_true_ability_line(self) -> None:
    """A recorded or live examinee has no known true ability, so the plot draws no line for one.

    Fails if a plot without ``true_theta`` raises, or draws a line at some made-up ability.
    """
    live = Session(estimates=self.SESSION.estimates[:2], administered_items=(2,), responses=(True,))
    ax = plot.session_progress(live, self.ITEM_BANK)

    assert set(self._lines(ax)) == {r"Ability estimate $\hat{\theta}$", "Item difficulty"}

  def test_session_progress_draws_on_one_scale(self) -> None:
    """The plot has one y axis, of ability, and the band of one standard error carries the precision.

    Fails if the standard error comes back on a second scale: it repeats the band, and the legend, placed among the
    first axes' lines only, can cover its line.
    """
    ax = plot.session_progress(self.SESSION, self.ITEM_BANK)

    assert ax.figure.axes == [ax]

  def test_session_progress_names_the_item_given_at_each_step(self) -> None:
    """Each step's tick gives the name of the item administered at it, and the initial estimate gets no tick.

    Fails if the names come in bank order rather than administration order, if a tick gives an index, if the names
    are shifted by one step, so that the initial estimate is named after the first item, or if they are read as
    mathtext.
    """
    ax = plot.session_progress(self.SESSION, self.ITEM_BANK)

    assert _x_tick_labels(ax) == [(1.0, "hard", False), (2.0, "easy", False), (3.0, "medium", False)]

  def test_session_progress_counts_the_items_of_a_long_session(self) -> None:
    """A session of 31 items ticks whole numbers of items administered instead of one name per step.

    Fails if 31 names crowd onto the axis, where they overlap, or if a view zoomed in to a few of its steps gets
    fractional ticks such as 2.5 items.
    """
    item_bank = ItemBank.generate(31, rng=0)
    session = Session(
      administered_items=tuple(range(31)),
      responses=tuple(item % 2 == 0 for item in range(31)),
      estimates=(Estimate(0.0, np.inf, 0),) + (Estimate(0.1, 0.5, 1),) * 31,
    )
    ax = plot.session_progress(session, item_bank)
    ax.set_xlim(10, 13)

    _check_ticks_count(ax, item_bank.names)

  def test_session_progress_of_a_session_without_items_has_no_x_ticks(self) -> None:
    """A session that has given no item has no step to name or count, so its x axis has no ticks.

    Fails if the lone initial estimate gets ticks at fractions of an item around 0, such as -0.06 and 0.015.
    """
    ax = plot.session_progress(Session(estimates=self.SESSION.estimates[:1]), self.ITEM_BANK)

    assert _x_tick_labels(ax) == []

  def test_session_progress_rejects_items_outside_the_bank_before_drawing(self) -> None:
    """A session whose items are not in the item bank raises before any figure exists.

    The session is built without the bank, so it can hold an index past the bank's last item. Fails if a bad call
    leaves an empty figure open behind the error, or reads past the bank's parameters.
    """
    with pytest.raises(ValueError, match=r"administered_items\[0\] is 2; item indices must be between 0 and 1"):
      plot.session_progress(self.SESSION, ItemBank(self.ITEM_BANK.items[:2]))
    assert plt.get_fignums() == []


class TestBankInformation:
  """Tests for bank_information()."""

  # Two-parameter items: a = 1 at b = 0 and b = 1, and a = 2 at b = 0. A 2PL item's information is a^2 P (1 - P).
  ITEM_BANK = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))

  @pytest.mark.parametrize(
    ("items", "expected"),
    [
      pytest.param(None, 0.25 + _logistic(-1) * _logistic(1) + 1.0, id="whole-bank"),
      pytest.param([0, 1], 0.25 + _logistic(-1) * _logistic(1), id="two-items"),
    ],
  )
  def test_bank_information_sums_the_information_of_its_items(self, items: list[int] | None, expected: float) -> None:
    """At theta = 0 the information is the sum of a^2 P (1 - P) over the items, and the standard error 1 / sqrt of it.

    Fails if the information is averaged rather than summed, if ``items`` is ignored or reads other items, or if the
    standard error is not the inverse square root of the information.
    """
    ax = plot.bank_information(self.ITEM_BANK, items=items)
    (information,) = ax.get_lines()
    (second,) = [axes for axes in ax.figure.axes if axes is not ax]
    (standard_error,) = second.get_lines()

    information_at_0 = np.interp(0.0, np.asarray(information.get_xdata()), np.asarray(information.get_ydata()))
    assert information_at_0 == pytest.approx(expected)
    se_at_0 = np.interp(0.0, np.asarray(standard_error.get_xdata()), np.asarray(standard_error.get_ydata()))
    assert se_at_0 == pytest.approx(1 / math.sqrt(expected))

  def test_bank_information_without_standard_error_draws_one_scale(self) -> None:
    """With ``se=False`` the plot has one axes, so several item sets can share it.

    Fails if a second y axis is made anyway: each call would stack another scale on the first.
    """
    ax = plot.bank_information(self.ITEM_BANK, se=False)

    assert ax.figure.axes == [ax]

  def test_bank_information_leaves_out_an_infinite_standard_error(self) -> None:
    """An item of discrimination 200 has information that underflows to 0 far from its difficulty, where the standard
    error is infinite: the curve leaves those abilities out.

    Fails if 1 / sqrt(0) is computed, whose divide-by-zero warning the test settings turn into an error, or if an
    infinite standard error is drawn as a number.
    """
    ax = plot.bank_information(ItemBank(np.array([[200.0, 4.0, 0.0, 1.0]])))
    (second,) = [axes for axes in ax.figure.axes if axes is not ax]
    (standard_error,) = second.get_lines()
    values = np.asarray(standard_error.get_ydata())

    assert np.isnan(values[0])
    assert np.isfinite(values[-1])

  @pytest.mark.parametrize(
    ("items", "error", "message"),
    [
      pytest.param([], ValueError, "items is empty; pass the positions of at least one item", id="empty"),
      pytest.param([0, 3], ValueError, r"items\[1\] is 3; item indices must be between 0 and 2", id="outside"),
      pytest.param([0, 0], ValueError, r"items\[1\] is 0 \('item_0'\); item indices must not repeat", id="repeated"),
      pytest.param([0.5], TypeError, r"items\[0\] is 0.5; item indices must be integers", id="float"),
      pytest.param([0, "x"], TypeError, r"items\[1\] is 'x'; item indices must be integers", id="mixed"),
      pytest.param(2, ValueError, r"items has shape \(\); it must be a sequence of item positions", id="scalar"),
      pytest.param([[0, 1], [2, 0]], ValueError, r"items has shape \(2, 2\)", id="matrix"),
    ],
  )
  def test_bank_information_rejects_items_that_are_not_a_set_of_the_banks_before_drawing(
    self, items: npt.ArrayLike, error: type[Exception], message: str
  ) -> None:
    """Fails if an empty set draws a flat zero, an item outside the bank is read past its end, an item counts twice, a
    single number reads as one item (``items=2`` would plot item 2 alone, not two items), a matrix is flattened, a bad
    entry is blamed on a good one, or a figure is left open behind the error."""
    with pytest.raises(error, match=message):
      plot.bank_information(self.ITEM_BANK, items=items)
    assert plt.get_fignums() == []


class TestExpectedScore:
  """Tests for expected_score()."""

  def test_expected_score_is_the_mean_of_the_items_characteristic_curves(self) -> None:
    """At theta = 0, items at b = 0 and b = 1 answer correctly with 0.5 and logistic(-1): the curve is their mean.

    Fails if the curves are summed, which would put a two-item set on another scale than a one-item set, or if
    ``items`` is ignored.
    """
    item_bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [3.0, -2.0, 0.0, 1.0]]))

    (line,) = plot.expected_score(item_bank, items=[0, 1]).get_lines()

    at_0 = np.interp(0.0, np.asarray(line.get_xdata()), np.asarray(line.get_ydata()))
    assert at_0 == pytest.approx((0.5 + _logistic(-1)) / 2)


class TestItemPersonMap:
  """Tests for item_person_map()."""

  # A 3PL item with a = 0.3, b = -1 and c = 0.5 is most informative at b + log((1 + sqrt(1 + 8c)) / 2) / a.
  ITEM_BANK = ItemBank(np.array([[0.3, -1.0, 0.5, 1.0]]))
  LOCATION = -1.0 + math.log((1 + math.sqrt(5)) / 2) / 0.3
  THETAS = np.linspace(-3, 3, 601)

  @staticmethod
  def _bars(ax) -> list[Rectangle]:
    """Return the bars of the plot."""
    return [patch for patch in ax.patches if isinstance(patch, Rectangle)]

  def test_item_person_map_places_items_where_they_are_most_informative(self) -> None:
    """The item's bar, below the axis, covers the ability where it is most informative, about 0.60, not its b of -1.

    Fails if items are placed at their difficulty, which for three-parameter items lies left of where they measure.
    """
    (item_bar,) = [bar for bar in self._bars(plot.item_person_map(self.ITEM_BANK, self.THETAS)) if bar.get_height() < 0]

    assert item_bar.get_x() <= self.LOCATION < item_bar.get_x() + item_bar.get_width()
    assert not item_bar.get_x() <= -1.0 < item_bar.get_x() + item_bar.get_width()

  def test_item_person_map_draws_shares_so_that_sizes_compare(self) -> None:
    """The examinees' bars sum to 1 above the axis and the items' bars to 1 below it.

    Fails if the bars count examinees and items, which puts a population of 601 and a bank of 1 on scales 600 times
    apart.
    """
    heights = np.array([bar.get_height() for bar in self._bars(plot.item_person_map(self.ITEM_BANK, self.THETAS))])

    assert heights[heights > 0].sum() == pytest.approx(1.0)
    assert heights[heights < 0].sum() == pytest.approx(-1.0)

  @pytest.mark.parametrize(
    ("thetas", "message"),
    [
      pytest.param([], r"thetas has shape \(0,\); it must hold one ability per examinee, at least one", id="empty"),
      pytest.param([0.0, math.nan], r"thetas\[1\] is nan; abilities must be finite numbers", id="nan"),
      pytest.param([[0.0, 1.0]], r"thetas has shape \(1, 2\)", id="matrix"),
    ],
  )
  def test_item_person_map_rejects_abilities_that_are_not_finite_numbers_before_drawing(
    self, thetas: npt.ArrayLike, message: str
  ) -> None:
    """Fails if NaN abilities are dropped silently from the shares, or a figure is left open behind the error."""
    with pytest.raises(ValueError, match=message):
      plot.item_person_map(self.ITEM_BANK, thetas)
    assert plt.get_fignums() == []


class TestLikelihood:
  """Tests for likelihood()."""

  # Two 2PL items at b = 0: answering one right and one wrong has likelihood P (1 - P), whose maximum, 1/4, is at 0.
  ITEM_BANK = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0]]))
  SESSION = Session(
    estimates=(Estimate(0.0, np.inf, 0), Estimate(0.5, 1.0, 1), Estimate(0.1, 0.9, 1)),
    administered_items=(0, 1),
    responses=(True, False),
  )

  def test_likelihood_is_relative_to_its_maximum_and_marks_the_estimate(self) -> None:
    """The curve is 4 P (1 - P): 1 at theta = 0 and 4 logistic(1) logistic(-1) at 1, with lines at 0.1 and 0.4.

    Fails if the curve is the log-likelihood or is not divided by its maximum, if the dashed line marks another estimate
    than the session's final one, or if the true ability given is not marked.
    """
    ax = plot.likelihood(self.SESSION, self.ITEM_BANK, true_theta=0.4)
    curve, estimate, true_ability = ax.get_lines()
    x, y = np.asarray(curve.get_xdata()), np.asarray(curve.get_ydata())

    assert np.interp(0.0, x, y) == pytest.approx(1.0)
    assert np.interp(1.0, x, y) == pytest.approx(4 * _logistic(1) * _logistic(-1))
    assert (x[0], x[-1]) == (-6.0, 6.0)
    assert np.asarray(estimate.get_xdata()).tolist() == [0.1, 0.1]
    assert np.asarray(true_ability.get_xdata()).tolist() == [0.4, 0.4]

  def test_likelihood_of_correct_responses_rises_to_the_edge(self) -> None:
    """Two correct responses have no maximum inside the range, so the curve rises to 1 at theta = 6.

    Fails if the curve is divided by its value at the estimate, which puts it above 1 past the estimate.
    """
    session = Session(estimates=self.SESSION.estimates, administered_items=(0, 1), responses=(True, True))

    (curve, _) = plot.likelihood(session, self.ITEM_BANK).get_lines()
    y = np.asarray(curve.get_ydata())

    assert (np.diff(y) > 0).all()
    assert y[-1] == pytest.approx(1.0)

  def test_likelihood_rejects_items_outside_the_bank_before_drawing(self) -> None:
    """Fails if a bad call reads past the bank's parameters or leaves a figure open behind the error."""
    with pytest.raises(ValueError, match=r"administered_items\[1\] is 1; item indices must be between 0 and 0"):
      plot.likelihood(self.SESSION, ItemBank(self.ITEM_BANK.items[:1]))
    assert plt.get_fignums() == []


class TestMetricByAbility:
  """Tests for metric_by_ability()."""

  # Twenty examinees of abilities 0 to 19, two per decile. In decile k, the examinee of ability 2k ends at an error of
  # 0.1k + 0.05 and the one of 2k + 1 at 0.1k - 0.05, so the decile's bias is 0.1k with a Monte Carlo error of 0.05.
  RESULT = _run(
    [
      _ended((k,), (2 * k + 0.1 * k + 0.05,))
      if examinee % 2 == 0
      else _ended((k, 10), (0.0, 2 * k + 1 + 0.1 * k - 0.05))
      for examinee in range(20)
      for k in [examinee // 2]
    ],
    true_thetas=np.arange(20.0),
  )

  def test_metric_by_ability_draws_each_deciles_metric_at_its_mean_ability(self) -> None:
    """Decile k's bias, 0.1k, is drawn at its mean ability, 2k + 0.5, with bars of 0.05 each way, over a line at 0.

    Fails if a decile's metric is drawn at another decile's ability, if the bars span another width than one Monte Carlo
    error each way, or if the bias has no line at 0 to be read against.
    """
    ax = plot.metric_by_ability(self.RESULT, "bias")
    (container,) = ax.containers
    assert isinstance(container, ErrorbarContainer)
    points, _, (bars,) = container.lines
    k = np.arange(10)

    np.testing.assert_allclose(np.asarray(points.get_xdata()), 2 * k + 0.5)
    np.testing.assert_allclose(np.asarray(points.get_ydata()), 0.1 * k, atol=1e-12)
    np.testing.assert_allclose(
      [segment[:, 1] for segment in bars.get_segments()], np.c_[0.1 * k - 0.05, 0.1 * k + 0.05], atol=1e-12
    )
    assert any(np.asarray(line.get_ydata()).tolist() == [0, 0] for line in ax.get_lines())

  def test_metric_by_ability_with_levels_draws_each_distinct_ability(self) -> None:
    """Two examinees at each of -1, 0 and 1 give three points, with RMSEs sqrt(0.05), 0.2 and sqrt(0.08).

    Fails if ``levels`` still groups by decile, which with six examinees puts the six in five groups.
    """
    sessions = [_ended((0,), (theta,)) for theta in (-0.9, -0.7, -0.2, 0.2, 1.4, 1.0)]
    ax = plot.metric_by_ability(_run(sessions, true_thetas=[-1, -1, 0, 0, 1, 1]), "rmse", levels=True)
    (container,) = ax.containers
    assert isinstance(container, ErrorbarContainer)
    points = container.lines[0]

    np.testing.assert_allclose(np.asarray(points.get_xdata()), [-1.0, 0.0, 1.0])
    np.testing.assert_allclose(np.asarray(points.get_ydata()), [math.sqrt(0.05), 0.2, math.sqrt(0.08)])

  @pytest.mark.parametrize(
    ("result", "metric", "message"),
    [
      pytest.param(RESULT, "evaluations", "Unsupported metric 'evaluations'", id="unknown-metric"),
      pytest.param(
        _run([_ended((0,), (0.1,))]), "rmse", "deciles needs the examinees' true abilities", id="no-true-abilities"
      ),
    ],
  )
  def test_metric_by_ability_rejects_what_it_cannot_plot_before_drawing(
    self, result: SimulationResult, metric: str, message: str
  ) -> None:
    """Fails if an unknown metric reads a column that is not there, or a figure is left open behind the error."""
    with pytest.raises(ValueError, match=message):
      plot.metric_by_ability(result, metric)
    assert plt.get_fignums() == []


class TestMetricByLength:
  """Tests for metric_by_length()."""

  # Examinee A (true 0) takes three items and estimates 0.5, 0.3 and 0.1, with standard errors 0.9, 0.7 and 0.5;
  # examinee B (true 1) stops after one item at 1.4, with 0.8.
  SESSIONS = (_ended((0, 1, 2), (0.5, 0.3, 0.1), (0.9, 0.7, 0.5)), _ended((3,), (1.4,), (0.8,)))

  def test_metric_by_length_counts_a_stopped_test_at_its_final_estimate(self) -> None:
    """After 1, 2 and 3 items, B's error stays 0.4: the RMSEs are sqrt((0.25 + 0.16) / 2), then 0.09 and 0.01 for 0.25.

    Fails if only the tests still running count, which after 2 items gives A's 0.3 alone and judges a variable-length
    run on its longest tests, or if the x axis counts from 0.
    """
    (line,) = plot.metric_by_length(_run(list(self.SESSIONS), true_thetas=[0.0, 1.0]), "rmse").get_lines()

    np.testing.assert_array_equal(np.asarray(line.get_xdata()), [1, 2, 3])
    expected = np.sqrt([(0.25 + 0.16) / 2, (0.09 + 0.16) / 2, (0.01 + 0.16) / 2])
    np.testing.assert_allclose(np.asarray(line.get_ydata()), expected)

  def test_metric_by_length_draws_the_mean_standard_error_without_true_abilities(self) -> None:
    """The mean standard errors are 0.85, 0.75 and 0.65, from recorded sessions whose true abilities are unknown.

    Fails if the standard error, which needs no true ability, raises for a result without them, or a stopped test drops
    out of the mean.
    """
    (line,) = plot.metric_by_length(_run(list(self.SESSIONS)), "se").get_lines()

    np.testing.assert_allclose(np.asarray(line.get_ydata()), [0.85, 0.75, 0.65])

  @pytest.mark.parametrize(
    ("true_thetas", "metric", "message"),
    [
      pytest.param([0.0, 1.0], "test_length", "Unsupported metric 'test_length'", id="unknown-metric"),
      pytest.param(None, "bias", "the bias needs the examinees' true abilities", id="no-true-abilities"),
    ],
  )
  def test_metric_by_length_rejects_what_it_cannot_plot_before_drawing(
    self, true_thetas: list[float] | None, metric: str, message: str
  ) -> None:
    """Fails if the bias is computed against missing abilities, or a figure is left open behind the error."""
    with pytest.raises(ValueError, match=message):
      plot.metric_by_length(_run(list(self.SESSIONS), true_thetas=true_thetas), metric)
    assert plt.get_fignums() == []


class TestRecovery:
  """Tests for recovery()."""

  # Examinees of true abilities 0 and 1 end at 0.1 and 1.2: errors 0.1 and 0.2, so a bias of 0.15 with an error of
  # 0.05, and an RMSE of sqrt(0.025) = 0.158 whose error is sd(0.01, 0.04) / sqrt(2) / (2 * 0.158) = 0.047.
  RESULT = _run([_ended((0,), (0.1,)), _ended((1, 2), (0.0, 1.2))], true_thetas=[0.0, 1.0])

  def test_recovery_pairs_each_final_estimate_with_its_true_ability(self) -> None:
    """The points are (0, 0.1) and (1, 1.2) under the identity line, and the box gives the RMSE and bias by hand.

    Fails if an estimate is paired with another examinee's ability, if an estimate before the last is drawn, or if the
    box reports other numbers than the run's.
    """
    ax = plot.recovery(self.RESULT)
    points, identity = ax.get_lines()
    assert isinstance(identity, AxLine)
    (box,) = ax.texts

    np.testing.assert_allclose(np.asarray(points.get_xdata()), [0.0, 1.0])
    np.testing.assert_allclose(np.asarray(points.get_ydata()), [0.1, 1.2])
    assert (identity.get_xy1(), identity.get_slope()) == ((0, 0), 1)
    assert box.get_text() == "RMSE = 0.158 ± 0.047\nbias = 0.150 ± 0.050"

  def test_recovery_without_true_abilities_raises_before_drawing(self) -> None:
    """Fails if estimates are drawn against missing abilities, or a figure is left open behind the error."""
    with pytest.raises(ValueError, match="rmse needs the examinees' true abilities"):
      plot.recovery(_run([_ended((0,), (0.1,))]))
    assert plt.get_fignums() == []


class TestLengthDistribution:
  """Tests for length_distribution()."""

  def test_length_distribution_gives_each_length_its_own_bin(self) -> None:
    """Tests of 1, 3, 3 and 4 items fill bins centred on 1, 2, 3 and 4 with 1, 0, 2 and 1 sessions.

    Fails if the bins are matplotlib's default ten, which split or merge lengths, or have edges on whole numbers, where
    a test length falls between two bins.
    """
    sessions = [_ended(tuple(range(length)), (0.0,) * length) for length in (1, 3, 3, 4)]
    (outline,) = plot.length_distribution(_run(sessions)).patches
    assert isinstance(outline, Polygon)
    vertices = np.asarray(outline.get_xy())
    steps = [
      (float(x0), float(x1), float(y0)) for (x0, y0), (x1, y1) in itertools.pairwise(vertices) if y0 == y1 and x1 > x0
    ]

    assert steps == [(0.5, 1.5, 1.0), (1.5, 2.5, 0.0), (2.5, 3.5, 2.0), (3.5, 4.5, 1.0)]

  def test_length_distribution_of_one_length_ticks_that_length_alone(self) -> None:
    """A run whose tests all take 20 items ticks 20 and no fraction of an item around it.

    Fails if matplotlib's integer locator, which needs two whole numbers in view, falls back to ticks such as 19.65.
    """
    sessions = [_ended(tuple(range(20)), (0.0,) * 20) for _ in range(3)]

    ax = plot.length_distribution(_run(sessions, n_items=20))

    low, high = ax.get_xlim()
    assert [position for position, _, _ in _x_tick_labels(ax) if low <= position <= high] == [20.0]


@pytest.mark.parametrize(
  "draw",
  [
    lambda ax, title: plot.item_curve(1, TestItemCurve.ITEM_BANK, ax=ax, title=title),
    lambda ax, title: plot.item_exposure(TestItemExposure.RATES, TestItemExposure.ITEM_BANK, ax=ax, title=title),
    lambda ax, title: plot.session_progress(
      TestSessionProgress.SESSION, TestSessionProgress.ITEM_BANK, ax=ax, title=title
    ),
    lambda ax, title: plot.bank_information(TestBankInformation.ITEM_BANK, ax=ax, title=title),
    lambda ax, title: plot.expected_score(TestBankInformation.ITEM_BANK, ax=ax, title=title),
    lambda ax, title: plot.item_person_map(TestItemPersonMap.ITEM_BANK, TestItemPersonMap.THETAS, ax=ax, title=title),
    lambda ax, title: plot.likelihood(TestLikelihood.SESSION, TestLikelihood.ITEM_BANK, ax=ax, title=title),
    lambda ax, title: plot.metric_by_ability(TestMetricByAbility.RESULT, ax=ax, title=title),
    lambda ax, title: plot.metric_by_length(TestMetricByAbility.RESULT, ax=ax, title=title),
    lambda ax, title: plot.recovery(TestMetricByAbility.RESULT, ax=ax, title=title),
    lambda ax, title: plot.length_distribution(TestMetricByAbility.RESULT, ax=ax, title=title),
  ],
  ids=[
    "item_curve",
    "item_exposure",
    "session_progress",
    "bank_information",
    "expected_score",
    "item_person_map",
    "likelihood",
    "metric_by_ability",
    "metric_by_length",
    "recovery",
    "length_distribution",
  ],
)
def test_plot_draws_on_the_given_axes_and_returns_them(draw) -> None:
  """A plot function draws on the axes it is given, sets the title there in the style's size and returns those axes.

  Fails if a function opens a figure of its own despite ``ax``, ignores ``title``, sets the title at a size of its own
  rather than the style's, or returns other axes than the caller's.
  """
  figure, ax = plt.subplots()

  with plt.style.context({"axes.titlesize": 9}):
    assert draw(ax, "A title") is ax
  assert ax.get_title() == "A title"
  assert ax.title.get_fontsize() == 9
  assert ax.get_lines() or ax.patches
  assert plt.get_fignums() == [figure.number]


# Every kind of plot catsim draws, each on the axes it is given.
_PLOTS: dict[str, Callable[[Axes], Axes]] = {
  "icc": lambda ax: plot.item_curve(1, TestItemCurve.ITEM_BANK, kind=PlotType.ICC, ax=ax),
  "iic": lambda ax: plot.item_curve(1, TestItemCurve.ITEM_BANK, kind=PlotType.IIC, ax=ax),
  "both": lambda ax: plot.item_curve(1, TestItemCurve.ITEM_BANK, kind=PlotType.BOTH, ax=ax),
  "exposure": lambda ax: plot.item_exposure(TestItemExposure.RATES, TestItemExposure.ITEM_BANK, ax=ax),
  "exposure-by-b": lambda ax: plot.item_exposure(TestItemExposure.RATES, TestItemExposure.ITEM_BANK, par="b", ax=ax),
  "exposure-histogram": lambda ax: plot.item_exposure(
    TestItemExposure.RATES, TestItemExposure.ITEM_BANK, hist=True, ax=ax
  ),
  "session": lambda ax: plot.session_progress(TestSessionProgress.SESSION, TestSessionProgress.ITEM_BANK, ax=ax),
  "bank_information": lambda ax: plot.bank_information(TestBankInformation.ITEM_BANK, ax=ax),
  "expected_score": lambda ax: plot.expected_score(TestBankInformation.ITEM_BANK, ax=ax),
  "item_person_map": lambda ax: plot.item_person_map(TestItemPersonMap.ITEM_BANK, TestItemPersonMap.THETAS, ax=ax),
  "likelihood": lambda ax: plot.likelihood(TestLikelihood.SESSION, TestLikelihood.ITEM_BANK, ax=ax),
  "metric_by_ability": lambda ax: plot.metric_by_ability(TestMetricByAbility.RESULT, ax=ax),
  "metric_by_length": lambda ax: plot.metric_by_length(TestMetricByAbility.RESULT, ax=ax),
  "recovery": lambda ax: plot.recovery(TestMetricByAbility.RESULT, ax=ax),
  "length_distribution": lambda ax: plot.length_distribution(TestMetricByAbility.RESULT, ax=ax),
}


@pytest.mark.parametrize("grid", [True, False])
@pytest.mark.parametrize("name", list(_PLOTS))
def test_plot_has_a_grid_exactly_when_the_style_has_one(name: str, grid: bool) -> None:
  """A plot shows a grid when the matplotlib style in effect has one, and none when it has none.

  Fails if a function switches the grid itself, as ``ax.grid()`` with no arguments does: it toggles the grid, which
  takes a style's grid away and gives a plain plot one.
  """
  with plt.style.context({"axes.grid": grid}):
    _, ax = plt.subplots()
    _PLOTS[name](ax)

    assert _shows_grid(ax) is grid


@pytest.mark.parametrize("name", ["both", "bank_information"])
def test_plot_right_hand_scale_has_a_spine_and_no_grid(name: str) -> None:
  """The second y axis of a plot with two scales draws its spine and no grid, whatever the style.

  Fails if the second axes take the style's grid, whose lines follow the right-hand scale and cross the first grid's,
  or a style's hidden right spine, which leaves the right-hand ticks floating beside the plot.
  """
  with plt.style.context({"axes.grid": True, "axes.spines.right": False}):
    _, ax = plt.subplots()
    _PLOTS[name](ax)
    (second,) = [axes for axes in ax.figure.axes if axes is not ax]

    assert not _shows_grid(second)
    assert second.spines.right.get_visible()


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
