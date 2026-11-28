"""Plots of items, item banks, CAT sessions and simulation results, drawn with matplotlib.

The module needs matplotlib, which the plot extra installs: ``pip install 'catsim[plot]'``. Each
function takes its data first, as plain catsim objects, and its options as keyword-only arguments,
draws on the ``ax`` it is given or on the axes of a new figure, and returns the axes it drew on
(ADR-0022).

Plots label items by their names in the item bank (ADR-0004), drawn as given: a dollar sign in a
name does not start mathtext. An x axis with one tick per item names each item while it has at most
30, the most names that stay apart at matplotlib's default figure width; a longer one ticks whole
numbers, as each function's description says.

Plots follow the matplotlib style, set with ``plt.style.use`` or ``plt.style.context``: its fonts,
colours, title size and grid. matplotlib fixes most of an axes' look when the axes are made, so
make ``ax`` under the style the plot should take. A second y axis, on the right, draws its spine
and no grid.

A plot that takes a ``label`` draws one run, item set or session in the next colour of the axes:
drawing several on the same axes, each with its ``label``, and calling ``ax.legend()`` compares
them. A plot that compares estimates with true abilities raises ValueError for a result without
them, before any figure is created.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import assert_never

import numpy
import numpy.typing as npt

try:
  import matplotlib.pyplot as plt
  from matplotlib.axes import Axes
  from matplotlib.colors import to_rgba
  from matplotlib.ticker import FuncFormatter, MaxNLocator
  from matplotlib.typing import ColorType
except ImportError as error:
  _missing = "catsim.plot needs matplotlib; install it with: pip install 'catsim[plot]'"
  raise ImportError(_missing) from error

from irtlib import ItemBank, irt
from irtlib.typing import FloatArray, IntArray

from . import metrics
from .checks import _check_in_bank, _item, _item_indices
from .session import Session
from .simulation import SimulationResult

# The most items an x axis names, one tick each: at matplotlib's default figure width, more names overlap.
_MAX_NAMED_TICKS = 30
# The abilities the curves of an item bank span.
_BANK_THETAS = numpy.linspace(-4, 4, 161)
# The abilities a session's likelihood spans: the range catsim's estimators keep their estimates in (ADR-0018).
_LIKELIHOOD_THETAS = numpy.linspace(-6, 6, 481)


class PlotType(StrEnum):
  """The curves :func:`item_curve` draws.

  Attributes
  ----------
  ICC
      The item characteristic curve, the probability of a correct response against ability.
  IIC
      The item information curve, the item's information against ability.
  BOTH
      Both curves on one plot, each with its own y axis.
  """

  ICC = "icc"
  IIC = "iic"
  BOTH = "both"


def item_curve(
  item: int,
  item_bank: ItemBank,
  *,
  kind: PlotType | str = PlotType.ICC,
  max_info: bool = True,
  ax: Axes | None = None,
  title: str | None = None,
) -> Axes:
  r"""Plot an item's characteristic curve, its information curve, or both.

  The curves span four units of ability on each side of the item's difficulty, and a box in the
  upper left corner gives the item's name and parameters. With ``PlotType.BOTH`` the information
  curve has its own y axis, on the right, and the two curves take the first two colours of the
  style.

  .. plot::
      :caption: One item's characteristic and information curves, apart and together. The dot marks its peak.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from catsim.plot import PlotType, item_curve
      item_bank = ItemBank.generate(1, rng=7)
      fig, axes = plt.subplots(3, 1, figsize=(7, 15))
      item_curve(0, item_bank, kind=PlotType.ICC, ax=axes[0])
      item_curve(0, item_bank, kind=PlotType.IIC, ax=axes[1])
      item_curve(0, item_bank, kind=PlotType.BOTH, ax=axes[2])
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  item : int
      The item's index in ``item_bank``; ``item_bank.index(name)`` gives the index of a named item.
  item_bank : ItemBank
      The item bank that holds the item's name and parameters.
  kind : PlotType or str, optional
      Which curves to draw, as a :class:`PlotType` or its value (``"icc"``, ``"iic"`` or
      ``"both"``). Default is ``PlotType.ICC``.
  max_info : bool, optional
      Whether to mark the ability at which the item gives the most information with a dot on the
      information curve. Default is True.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the characteristic curve is on, or those of the information curve for
      ``PlotType.IIC``.

  Raises
  ------
  TypeError
      If ``item`` is not an integer, before any figure is created.
  ValueError
      If ``item`` is not an index of ``item_bank`` or ``kind`` is neither a :class:`PlotType` nor
      one of its values, before any figure is created.
  """
  item = _item("item", item, item_bank.n_items)
  kind = PlotType(kind)
  parameters = item_bank.items[item]
  ax = _prepare_axes(ax, title)
  thetas = numpy.linspace(parameters[1] - 4, parameters[1] + 4, 161)

  match kind:
    case PlotType.ICC:
      _draw_icc(ax, thetas, parameters)
      box_axes = ax
    case PlotType.IIC:
      _draw_information(ax, thetas, parameters, max_info=max_info)
      box_axes = ax
    case PlotType.BOTH:
      box_axes = _right_hand_axes(ax)
      icc_color, information_color = _cycle_color(0), _cycle_color(1)
      for axes, color in ((ax, icc_color), (box_axes, information_color)):
        axes.yaxis.label.set_color(color)
        axes.tick_params(axis="y", colors=color)
      _draw_icc(ax, thetas, parameters, color=icc_color)
      _draw_information(box_axes, thetas, parameters, max_info=max_info, color=information_color)
    case _:
      assert_never(kind)

  ax.set_xlabel(r"$\theta$")
  _annotate_item(box_axes, item_bank.names[item], parameters)
  return ax


def _prepare_axes(ax: Axes | None, title: str | None) -> Axes:
  """Return ``ax``, or the axes of a new figure, with ``title`` set when one is given."""
  if ax is None:
    _, ax = plt.subplots()
  if title is not None:
    ax.set_title(title)
  return ax


def _right_hand_axes(ax: Axes) -> Axes:
  """Return twin axes that share ``ax``'s x axis and draw a second y scale on the right, with its spine and no grid."""
  twin = ax.twinx()
  # The twin's grid lines would follow the right-hand scale and cross ax's, and a style that hides the right spine would
  # leave the scale's ticks floating.
  twin.grid(visible=False)
  twin.spines.right.set_visible(True)
  return twin


def _cycle_color(index: int) -> tuple[float, float, float, float]:
  """Return the ``index``-th colour of the current style's cycle, as RGBA."""
  # matplotlib looks a reference such as "C1" up when the figure is drawn, so a figure drawn after its style ended would
  # take another style's colour.
  return to_rgba(f"C{index}")


def _label_items(ax: Axes, positions: IntArray, names: Sequence[str]) -> None:
  """Tick the x axis at ``positions`` with the ``names`` of the items there, or at whole numbers if there are too many.

  Names are drawn as given, vertically so that long ones fit. With more than ``_MAX_NAMED_TICKS`` names, matplotlib
  picks the ticks among the whole numbers, which also holds in a view zoomed in to a few positions.
  """
  if len(names) <= _MAX_NAMED_TICKS:
    ax.set_xticks(positions, names, rotation=90, parse_math=False)
  else:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _draw_icc(ax: Axes, thetas: FloatArray, parameters: FloatArray, *, color: ColorType | None = None) -> None:
  """Draw the item's characteristic curve, in ``color`` or the next colour of the axes' cycle."""
  ax.plot(thetas, irt.icc(thetas, parameters), color=color, label=r"$P(\theta)$")
  ax.set_ylabel(r"$P(\theta)$")


def _draw_information(
  ax: Axes, thetas: FloatArray, parameters: FloatArray, *, max_info: bool, color: ColorType | None = None
) -> None:
  """Draw the item's information curve and, with ``max_info``, a dot of the curve's colour at its peak."""
  (line,) = ax.plot(thetas, irt.information(thetas, parameters), color=color, label=r"$I(\theta)$")
  ax.set_ylabel(r"$I(\theta)$")
  if max_info:
    peak = irt.max_info_theta(parameters)
    ax.plot(peak, irt.information(peak, parameters), "o", color=line.get_color(), label="Maximum information")


def _annotate_item(ax: Axes, name: str, parameters: FloatArray) -> None:
  """Write the item's name and parameters, to three significant digits, in a box in the upper left corner of the axes.

  The curves of typical items keep clear of that corner: at the low end of the ability range the ICC is near its
  lower asymptote and the IIC near zero. A very flat item (a of 0.2 or less) with c = 0 and d < 1 is the exception:
  its information curve falls from the left edge of the plot, under the box.
  """
  a, b, c, d = parameters
  # The parameters are mathtext, so a dollar sign in the name would pair up with theirs.
  shown_name = name.replace("$", r"\$")
  ax.annotate(
    f"{shown_name}\n$a = {a:.3g}$\n$b = {b:.3g}$\n$c = {c:.3g}$\n$d = {d:.3g}$",
    bbox={"facecolor": "white", "alpha": 1},
    xy=(0.02, 0.97),
    xycoords="axes fraction",
    ha="left",
    va="top",
  )


def bank_information(
  item_bank: ItemBank,
  *,
  items: npt.ArrayLike | None = None,
  se: bool = True,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot the information of an item bank, or of some of its items, against ability, with its standard error.

  The information :math:`I(\theta)` is the sum of the items' information, from -4 to 4. For a whole
  bank it bounds what a test drawn from it can measure; with ``items`` it is the information of that
  fixed set of items. With ``se``, the standard error :math:`1/\sqrt{I(\theta)}` goes on a second y
  axis, on the right, and each axis takes its curve's colour. To compare sets of items on one axes,
  pass ``se=False`` and a ``label`` with each.

  .. plot::
      :caption: The information of a 100-item bank, with its standard error.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from catsim import plot

      plot.bank_information(ItemBank.generate(100, "3PL", rng=0))
      plt.show()

  Parameters
  ----------
  item_bank : ItemBank
      The item bank.
  items : array-like or None, optional
      The positions of the items to sum, distinct integers of ``item_bank``. Default is None, which
      sums every item of the bank.
  se : bool, optional
      Whether to draw the standard error on a second y axis. Default is True.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The information curve's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes of the information. The standard error is drawn on their twin, which shares the x axis.

  Raises
  ------
  TypeError
      If an entry of ``items`` is not an integer, before any figure is created.
  ValueError
      If ``items`` is empty, repeats an item or names one outside the bank, before any figure is created.
  """
  information = irt.total_information(_BANK_THETAS, _bank_items(item_bank, items))
  ax = _prepare_axes(ax, title)
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel(r"Information $I(\theta)$")
  if not se:
    ax.plot(_BANK_THETAS, information, label=label)
    return ax
  information_color, se_color = _cycle_color(0), _cycle_color(1)
  ax.plot(_BANK_THETAS, information, color=information_color, label=label)
  se_axes = _right_hand_axes(ax)
  # Where the information underflows to 0 the standard error is infinite, and the curve leaves those abilities out.
  standard_errors = numpy.full_like(information, numpy.nan)
  numpy.divide(1, numpy.sqrt(information), out=standard_errors, where=information > 0)
  se_axes.plot(_BANK_THETAS, standard_errors, color=se_color, label="Standard error")
  se_axes.set_ylabel(r"Standard error $1/\sqrt{I(\theta)}$")
  for axes, color in ((ax, information_color), (se_axes, se_color)):
    axes.yaxis.label.set_color(color)
    axes.tick_params(axis="y", colors=color)
  return ax


def expected_score(
  item_bank: ItemBank,
  *,
  items: npt.ArrayLike | None = None,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot the expected proportion of correct responses to an item bank, or to some of its items, against ability.

  The proportion is the mean of the items' characteristic curves, from -4 to 4: the test
  characteristic curve divided by the number of items, so that sets of any size share one scale,
  from 0 to 1.

  .. plot::
      :caption: The expected proportion correct of a 100-item bank and of its first 20 items.

      import matplotlib.pyplot as plt
      from irtlib import ItemBank
      from catsim import plot

      item_bank = ItemBank.generate(100, "3PL", rng=0)
      ax = plot.expected_score(item_bank, label="All 100 items")
      plot.expected_score(item_bank, items=range(20), ax=ax, label="The first 20 items")
      ax.legend()
      plt.show()

  Parameters
  ----------
  item_bank : ItemBank
      The item bank.
  items : array-like or None, optional
      The positions of the items to average, distinct integers of ``item_bank``. Default is None,
      which averages every item of the bank.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  TypeError
      If an entry of ``items`` is not an integer, before any figure is created.
  ValueError
      If ``items`` is empty, repeats an item or names one outside the bank, before any figure is created.
  """
  expected = irt.icc(_BANK_THETAS, _bank_items(item_bank, items)).mean(axis=1)
  ax = _prepare_axes(ax, title)
  ax.plot(_BANK_THETAS, expected, label=label)
  ax.set_ylim(0, 1)
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel("Expected proportion correct")
  return ax


def item_person_map(
  item_bank: ItemBank, thetas: npt.ArrayLike, *, ax: Axes | None = None, title: str | None = None
) -> Axes:
  r"""Plot examinees' abilities above the axis and, below it, the abilities where the bank's items inform most.

  Both histograms share one :math:`\theta` scale and its bins, and each shows shares, so a bank and a
  population of any sizes compare: the examinees' bars sum to 1 above the axis and the items' bars
  to 1 below it. An item sits at ``ItemBank.max_info_thetas``, which is its difficulty
  for one- and two-parameter items.

  .. plot::
      :caption: A 100-item bank against a population of 500 examinees.

      import matplotlib.pyplot as plt
      import numpy
      from irtlib import ItemBank
      from catsim import plot

      thetas = numpy.random.default_rng(1).normal(0, 1, 500)
      plot.item_person_map(ItemBank.generate(100, "3PL", rng=0), thetas)
      plt.show()

  Parameters
  ----------
  item_bank : ItemBank
      The item bank.
  thetas : array-like
      The examinees' abilities, true or estimated, as finite numbers.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  ValueError
      If ``thetas`` is not a non-empty one-dimensional array of finite numbers, before any figure is
      created.
  """
  abilities = _abilities(thetas)
  locations = item_bank.max_info_thetas
  edges = numpy.histogram_bin_edges(numpy.concatenate([abilities, locations]), bins="auto")
  examinees, _ = numpy.histogram(abilities, edges)
  items, _ = numpy.histogram(locations, edges)
  widths = numpy.diff(edges)
  ax = _prepare_axes(ax, title)
  ax.bar(edges[:-1], examinees / abilities.size, widths, align="edge", label=f"Examinees ({abilities.size})")
  ax.bar(edges[:-1], -items / locations.size, widths, align="edge", label=f"Items ({locations.size})")
  ax.axhline(0, color="black", linewidth=0.8)
  # The items' shares are drawn downwards, as negative heights, and labelled as the positive shares they are.
  ax.yaxis.set_major_formatter(FuncFormatter(lambda share, _: f"{abs(share):g}"))
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel("Share of examinees (up) and items (down)")
  ax.legend(loc="upper right")
  return ax


def _bank_items(item_bank: ItemBank, items: npt.ArrayLike | None) -> FloatArray:
  """Return the parameters of the items at positions ``items`` of ``item_bank``, or of every item when it is None."""
  if items is None:
    return item_bank.items
  # An object array keeps each entry as given, so a check blames the bad entry, not a good one coerced to its type.
  values = numpy.asarray(items, dtype=object)
  if values.ndim != 1:
    msg = f"items has shape {values.shape}; it must be a sequence of item positions"
    raise ValueError(msg)
  indices = _item_indices("items", values.tolist(), item_bank.names)
  if not indices:
    msg = "items is empty; pass the positions of at least one item, or None for the whole bank"
    raise ValueError(msg)
  return item_bank.items[list(indices)]


def _abilities(thetas: npt.ArrayLike) -> FloatArray:
  """Return the abilities as a float array, checking that they are a non-empty 1-D array of finite numbers."""
  abilities = numpy.asarray(thetas, dtype=float)
  if abilities.ndim != 1 or abilities.size == 0:
    msg = f"thetas has shape {abilities.shape}; it must hold one ability per examinee, at least one"
    raise ValueError(msg)
  if not numpy.isfinite(abilities).all():
    position = int(numpy.flatnonzero(~numpy.isfinite(abilities))[0])
    msg = f"thetas[{position}] is {abilities[position]}; abilities must be finite numbers"
    raise ValueError(msg)
  return abilities


# The item parameter each value of item_exposure's ``par`` reads: its column in the item matrix and the axis label.
_ITEM_PARAMETERS = {
  "a": (0, "Item discrimination"),
  "b": (1, "Item difficulty"),
  "c": (2, "Item guessing parameter"),
  "d": (3, "Item upper asymptote"),
}


def item_exposure(  # noqa: PLR0913 - the rates and their bank, two views (par, hist), and every plot's ax, title, label
  exposure_rates: FloatArray,
  item_bank: ItemBank,
  *,
  par: str | None = None,
  hist: bool = False,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  """Plot the items' exposure rates: from the most to the least exposed, against an item parameter, or as a histogram.

  By default the items are sorted from the most to the least exposed, which shows how evenly a run
  uses its bank and where an exposure cap holds the rates; the x axis names each item, or counts them
  in a bank of more than 30 items. With ``par``, each item is a point at its value of that
  parameter, which shows which items a selector favours. With ``hist=True``, the plot counts the
  items in each range of exposure rates.

  .. plot::
      :caption: The exposure rates of a simulation, sorted and against the item difficulty.

      import matplotlib.pyplot as plt
      import numpy
      from catsim import CatEngine, simulate
      from irtlib import ItemBank
      from catsim import plot
      from catsim.estimation import EAPEstimator
      from catsim.initialization import FixedPointInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

      engine = CatEngine(
          ItemBank.generate(100, rng=0),
          FixedPointInitializer(0.0),
          MaxInfoSelector(),
          EAPEstimator(),
          StopRule(StandardErrorStopper(0.4), MaxLengthStopper(20)),
      )
      result = simulate(engine, true_thetas=numpy.random.default_rng(1).normal(0, 1, 200), rng=0)
      fig, axes = plt.subplots(2, 1, figsize=(7, 10))
      rates = result.exposure_rates().value
      plot.item_exposure(rates, result.item_bank, ax=axes[0])
      plot.item_exposure(rates, result.item_bank, par="b", ax=axes[1])
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  exposure_rates : numpy.ndarray
      The exposure rate of each item, in item bank order, such as the value of a
      :class:`~catsim.SimulationResult`'s :meth:`~catsim.SimulationResult.exposure_rates`.
  item_bank : ItemBank
      The item bank the rates belong to, which holds the items' names and parameters.
  par : str or None, optional
      The item parameter to plot the rates against: ``"a"``, ``"b"``, ``"c"`` or ``"d"``. Default is
      None, which sorts the items from the most to the least exposed.
  hist : bool, optional
      Whether to draw a histogram of the rates. Default is False.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The rates' label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  ValueError
      If ``exposure_rates`` is not one rate per item of ``item_bank``, ``par`` is not one of
      ``"a"``, ``"b"``, ``"c"`` and ``"d"``, or ``par`` is given with ``hist=True``; always before
      any figure is created.
  """
  exposure_rates = numpy.asarray(exposure_rates, dtype=float)
  _check_exposure_inputs(exposure_rates, item_bank, par, hist=hist)
  ax = _prepare_axes(ax, title)

  if hist:
    ax.hist(exposure_rates, max(len(exposure_rates) // 10, 3), label=label)
    ax.set_xlabel("Exposure rate")
    ax.set_ylabel("Items")
    return ax

  if par is None:
    order = numpy.argsort(-exposure_rates, kind="stable")
    ranks = numpy.arange(1, item_bank.n_items + 1)
    ax.plot(ranks, exposure_rates[order], drawstyle="steps-mid", label=label)
    _label_items(ax, ranks, [item_bank.names[item] for item in order])
    ax.set_xlabel("Items, from the most to the least exposed")
  else:
    column, xlabel = _ITEM_PARAMETERS[par]
    ax.plot(item_bank.items[:, column], exposure_rates, "o", label=label)
    ax.set_xlabel(xlabel)
  ax.set_ylabel("Exposure rate")
  return ax


def _check_exposure_inputs(exposure_rates: FloatArray, item_bank: ItemBank, par: str | None, *, hist: bool) -> None:
  """Raise ValueError for inputs of :func:`item_exposure` that cannot give a true plot."""
  if exposure_rates.shape != (item_bank.n_items,):
    msg = (
      f"exposure_rates has shape {exposure_rates.shape}; there must be one exposure rate per item of the bank "
      f"({item_bank.n_items})"
    )
    raise ValueError(msg)
  if par is None:
    return
  if par not in _ITEM_PARAMETERS:
    msg = f"Unsupported parameter {par!r} for 'par'. Supported parameters are: {', '.join(_ITEM_PARAMETERS)}."
    raise ValueError(msg)
  if hist:
    msg = f"A histogram of exposure rates has no parameter axis: pass par={par!r} or hist=True, not both."
    raise ValueError(msg)


def session_progress(
  session: Session,
  item_bank: ItemBank,
  *,
  true_theta: float | None = None,
  ax: Axes | None = None,
  title: str | None = None,
) -> Axes:
  r"""Plot how a session's ability estimate and its standard error change as items are administered.

  The x axis counts the items administered. While the session has at most 30 items, its ticks name
  the item given at each step instead, and the initial estimate, at 0, has none, so a session with
  no items has no ticks. The y axis shows the ability estimate after each response, with a band of
  one standard error on each side: the error each estimate was recorded with, the one
  precision-based stoppers judge. An infinite standard error, such as the initial estimate's, leaves
  the band out. The difficulty of each administered item and, when it is given, the true ability
  share the axis.

  .. plot::
      :caption: One simulated session: the ability estimate after each item, with its standard error.

      import matplotlib.pyplot as plt
      import numpy
      from catsim import CatEngine, simulate
      from irtlib import ItemBank
      from catsim import plot
      from catsim.estimation import NumericalSearchEstimator
      from catsim.initialization import UniformInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

      engine = CatEngine(
          ItemBank.generate(100, rng=0),
          UniformInitializer(),
          MaxInfoSelector(),
          NumericalSearchEstimator(),
          StopRule(StandardErrorStopper(0.4), MaxLengthStopper(20)),
      )
      true_thetas = numpy.random.default_rng(1).normal(0, 1, 10)
      result = simulate(engine, true_thetas=true_thetas, rng=0)
      fig, ax = plt.subplots(figsize=(8, 5))
      plot.session_progress(result.sessions[0], result.item_bank, true_theta=result.true_thetas[0], ax=ax)
      plt.show()

  Parameters
  ----------
  session : Session
      The session to plot, such as an entry of ``SimulationResult.sessions``.
  item_bank : ItemBank
      The item bank the session's items come from.
  true_theta : float or None, optional
      The examinee's true ability, drawn as a horizontal line, such as an entry of
      ``SimulationResult.true_thetas``. Default is None, which draws none.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  ValueError
      If an administered item is not an item of ``item_bank``, before any figure is created.
  """
  # The session checked its indices when it was built, but it does not know the bank.
  _check_in_bank("administered_items", session.administered_items, item_bank.n_items)
  administered = numpy.asarray(session.administered_items, dtype=int)
  difficulties = item_bank.difficulty[administered]
  thetas = numpy.array(session.theta_history)
  ses = numpy.array([estimate.se for estimate in session.estimates])
  xs = numpy.arange(len(thetas))
  finite = numpy.isfinite(ses)
  ax = _prepare_axes(ax, title)

  estimate_color, difficulty_color = _cycle_color(0), _cycle_color(1)
  ax.plot(xs, thetas, marker=".", color=estimate_color, label=r"Ability estimate $\hat{\theta}$")
  ax.fill_between(
    xs[finite],
    (thetas - ses)[finite],
    (thetas + ses)[finite],
    color=estimate_color,
    alpha=0.2,
    label=r"$\hat{\theta} \pm SE$",
  )
  ax.plot(xs[1:], difficulties, "o", color=difficulty_color, label="Item difficulty")
  if true_theta is not None:
    ax.axhline(true_theta, color="black", linestyle="--", label=r"True ability $\theta$")

  ax.set_xlabel("Items administered")
  _label_items(ax, xs[1:], [item_bank.names[item] for item in session.administered_items])
  ax.set_ylabel(r"$\theta$")
  ax.legend(loc="best")
  return ax


def likelihood(
  session: Session,
  item_bank: ItemBank,
  *,
  true_theta: float | None = None,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot the likelihood of a session's responses across abilities, relative to its maximum, with its final estimate.

  The curve is the likelihood of the responses to the administered items, divided by its maximum,
  from -6 to 6, the range catsim's estimators keep their estimates in. A dashed line in the curve's
  colour marks the session's final estimate and, when it is given, a dotted black line the true
  ability. Responses that are all correct, or all incorrect, have no maximum inside the range, so
  their curve rises to its edge.

  .. plot::
      :caption: The likelihood of one simulated session's responses.

      import matplotlib.pyplot as plt
      import numpy
      from catsim import CatEngine, simulate
      from irtlib import ItemBank
      from catsim import plot
      from catsim.estimation import EAPEstimator
      from catsim.initialization import FixedPointInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.stopping import MaxLengthStopper

      engine = CatEngine(
          ItemBank.generate(100, rng=0),
          FixedPointInitializer(0.0),
          MaxInfoSelector(),
          EAPEstimator(),
          MaxLengthStopper(10),
      )
      result = simulate(engine, true_thetas=[0.7], rng=0)
      plot.likelihood(result.sessions[0], result.item_bank, true_theta=0.7)
      plt.show()

  Parameters
  ----------
  session : Session
      The session whose responses to plot, such as an entry of ``SimulationResult.sessions``.
  item_bank : ItemBank
      The item bank the session's items come from.
  true_theta : float or None, optional
      The examinee's true ability, marked by a dotted line. Default is None, which marks none.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The curve's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  ValueError
      If an administered item is not an item of ``item_bank``, before any figure is created.
  """
  # The session checked its indices when it was built, but it does not know the bank.
  _check_in_bank("administered_items", session.administered_items, item_bank.n_items)
  items = item_bank.items[list(session.administered_items)]
  log_likelihood = irt.log_likelihood(_LIKELIHOOD_THETAS, [session.responses], items)[0]
  ax = _prepare_axes(ax, title)
  (line,) = ax.plot(_LIKELIHOOD_THETAS, numpy.exp(log_likelihood - numpy.max(log_likelihood)), label=label)
  ax.axvline(session.current_theta, color=line.get_color(), linestyle="--", linewidth=1)
  if true_theta is not None:
    ax.axvline(true_theta, color="black", linestyle=":", linewidth=1)
  ax.set_xlabel(r"$\theta$")
  ax.set_ylabel("Likelihood relative to its maximum")
  return ax


# The metrics of a simulation run that metric_by_ability can plot, each with its axis label. metric_by_length plots the
# first four.
_METRICS = {
  "bias": r"Bias of $\hat{\theta}$",
  "mse": r"MSE of $\hat{\theta}$",
  "rmse": r"RMSE of $\hat{\theta}$",
  "se": r"Mean standard error of $\hat{\theta}$",
  "test_length": "Mean test length",
  "overlap_rate": "Test overlap rate",
}
_METRICS_BY_LENGTH = ("bias", "mse", "rmse", "se")


def _check_metric(metric: str, supported: Sequence[str]) -> None:
  """Raise ValueError unless ``metric`` is one of ``supported``."""
  if metric not in supported:
    msg = f"Unsupported metric {metric!r}. Supported metrics are: {', '.join(supported)}."
    raise ValueError(msg)


def metric_by_ability(
  result: SimulationResult,
  metric: str = "rmse",
  *,
  levels: bool = False,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot a run's metric in each decile of the true ability, or at each distinct true ability, with its error bars.

  Each point is a group's metric at its mean true ability, from the result's
  :meth:`~catsim.SimulationResult.deciles`, or with ``levels=True`` from its
  :meth:`~catsim.SimulationResult.ability_levels`, which suits designs of a few fixed abilities. The
  bars span one Monte Carlo standard error on each side, and the bias gets a line at 0.

  .. plot::
      :caption: The RMSE in each decile of the true ability, for two selectors.

      import matplotlib.pyplot as plt
      import numpy
      from catsim import CatEngine, simulate
      from irtlib import ItemBank
      from catsim import plot
      from catsim.estimation import EAPEstimator
      from catsim.initialization import FixedPointInitializer
      from catsim.selection import MaxInfoSelector, RandomesqueSelector
      from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

      item_bank = ItemBank.generate(100, rng=0)
      true_thetas = numpy.random.default_rng(1).normal(0, 1, 300)
      stopper = StopRule(StandardErrorStopper(0.4), MaxLengthStopper(20))
      fig, ax = plt.subplots()
      for name, selector in [("Maximum information", MaxInfoSelector()), ("Randomesque", RandomesqueSelector(5))]:
          engine = CatEngine(item_bank, FixedPointInitializer(0.0), selector, EAPEstimator(), stopper)
          result = simulate(engine, true_thetas=true_thetas, rng=2)
          plot.metric_by_ability(result, "rmse", ax=ax, label=name)
      ax.legend()
      plt.show()

  Parameters
  ----------
  result : SimulationResult
      The run, with its examinees' true abilities.
  metric : str, optional
      The metric: ``"bias"``, ``"mse"``, ``"rmse"``, ``"se"`` (the mean standard error),
      ``"test_length"`` or ``"overlap_rate"``. Default is ``"rmse"``.
  levels : bool, optional
      Whether to group the sessions by distinct true ability rather than by decile. Default is False.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The run's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  ValueError
      If ``metric`` is not one of the metrics above, or the result has no true abilities, before any
      figure is created.
  """
  _check_metric(metric, tuple(_METRICS))
  table = result.ability_levels() if levels else result.deciles()
  ax = _prepare_axes(ax, title)
  ax.errorbar(table["true_theta"], table[metric], yerr=table[f"{metric}_mc_se"], marker="o", capsize=2, label=label)
  if metric == "bias":
    ax.axhline(0, color="black", linewidth=0.8)
  ax.set_xlabel(r"True ability $\theta$")
  ax.set_ylabel(_METRICS[metric])
  return ax


def metric_by_length(
  result: SimulationResult,
  metric: str = "rmse",
  *,
  ax: Axes | None = None,
  title: str | None = None,
  label: str | None = None,
) -> Axes:
  r"""Plot a run's metric after each number of items administered, with a band of one Monte Carlo standard error.

  Every session counts after every number of items: one that stopped earlier counts at its final
  estimate, so a variable-length run is not judged on its longest tests alone. The metric is the
  bias, MSE or RMSE of the estimates, which need the true abilities, or their mean standard error.

  .. plot::
      :caption: The RMSE and the mean standard error after each item.

      import matplotlib.pyplot as plt
      import numpy
      from catsim import CatEngine, simulate
      from irtlib import ItemBank
      from catsim import plot
      from catsim.estimation import EAPEstimator
      from catsim.initialization import FixedPointInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

      engine = CatEngine(
          ItemBank.generate(100, rng=0),
          FixedPointInitializer(0.0),
          MaxInfoSelector(),
          EAPEstimator(),
          StopRule(StandardErrorStopper(0.4), MaxLengthStopper(20)),
      )
      result = simulate(engine, true_thetas=numpy.random.default_rng(1).normal(0, 1, 300), rng=2)
      ax = plot.metric_by_length(result, "rmse", label="RMSE")
      plot.metric_by_length(result, "se", ax=ax, label="Mean standard error")
      ax.set_ylabel(r"Error of $\hat{\theta}$")
      ax.legend()
      plt.show()

  Parameters
  ----------
  result : SimulationResult
      The run.
  metric : str, optional
      The metric: ``"bias"``, ``"mse"``, ``"rmse"`` or ``"se"`` (the mean standard error). Default
      is ``"rmse"``.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The run's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  ValueError
      If ``metric`` is not one of the metrics above, or is the bias, MSE or RMSE of a result without
      true abilities, before any figure is created.
  """
  _check_metric(metric, _METRICS_BY_LENGTH)
  thetas, ses = _estimates_by_length(result)
  if metric == "se":
    values = [metrics.mean(column) for column in ses.T]
  else:
    true_thetas = result.true_thetas
    if true_thetas is None:
      msg = f"the {metric} needs the examinees' true abilities, and this result has none"
      raise ValueError(msg)
    function = {"bias": metrics.bias, "mse": metrics.mse, "rmse": metrics.rmse}[metric]
    values = [function(true_thetas, column) for column in thetas.T]
  lengths = numpy.arange(1, thetas.shape[1] + 1)
  value = numpy.array([metric_.value for metric_ in values], dtype=float)
  mc_se = numpy.array([metric_.mc_se for metric_ in values], dtype=float)
  ax = _prepare_axes(ax, title)
  (line,) = ax.plot(lengths, value, label=label)
  ax.fill_between(lengths, value - mc_se, value + mc_se, color=line.get_color(), alpha=0.2, linewidth=0)
  ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
  ax.set_xlabel("Items administered")
  ax.set_ylabel(_METRICS[metric])
  return ax


def _estimates_by_length(result: SimulationResult) -> tuple[FloatArray, FloatArray]:
  """Return each session's estimate and standard error after 1, 2, ... items: a row per session, a column per length."""
  # The columns run to the longest test, and a session that stopped keeps its final estimate in the later ones.
  lengths = [len(session.administered_items) for session in result.sessions]
  steps = numpy.arange(1, max(lengths) + 1)
  thetas = numpy.empty((len(lengths), steps.size))
  ses = numpy.empty((len(lengths), steps.size))
  for row, (session, length) in enumerate(zip(result.sessions, lengths, strict=True)):
    estimates = [session.estimates[step] for step in numpy.minimum(steps, length)]
    thetas[row] = [estimate.theta for estimate in estimates]
    ses[row] = [estimate.se for estimate in estimates]
  return thetas, ses


def recovery(result: SimulationResult, *, ax: Axes | None = None, title: str | None = None) -> Axes:
  r"""Plot each examinee's final ability estimate against their true ability, with the run's RMSE and bias.

  The dashed line marks estimates equal to the true ability, so points above it are overestimates. A
  box in the upper left corner gives the RMSE and the bias, each with its Monte Carlo standard error.

  .. plot::
      :caption: The final estimates of 300 simulated examinees against their true abilities.

      import matplotlib.pyplot as plt
      import numpy
      from catsim import CatEngine, simulate
      from irtlib import ItemBank
      from catsim import plot
      from catsim.estimation import EAPEstimator
      from catsim.initialization import FixedPointInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

      engine = CatEngine(
          ItemBank.generate(100, rng=0),
          FixedPointInitializer(0.0),
          MaxInfoSelector(),
          EAPEstimator(),
          StopRule(StandardErrorStopper(0.4), MaxLengthStopper(20)),
      )
      result = simulate(engine, true_thetas=numpy.random.default_rng(1).normal(0, 1, 300), rng=2)
      plot.recovery(result)
      plt.show()

  Parameters
  ----------
  result : SimulationResult
      The run, with its examinees' true abilities.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.

  Raises
  ------
  ValueError
      If the result has no true abilities, before any figure is created.
  """
  rmse, bias = result.rmse(), result.bias()
  table = result.examinees()
  ax = _prepare_axes(ax, title)
  ax.plot(table["true_theta"], table["theta"], "o", markersize=3, alpha=0.5)
  ax.axline((0, 0), slope=1, color="black", linestyle="--", linewidth=0.8)
  ax.set_xlabel(r"True ability $\theta$")
  ax.set_ylabel(r"Final estimate $\hat{\theta}$")
  ax.annotate(
    f"RMSE = {rmse.value:.3f} ± {rmse.mc_se:.3f}\nbias = {bias.value:.3f} ± {bias.mc_se:.3f}",
    bbox={"facecolor": "white", "alpha": 1},
    xy=(0.02, 0.97),
    xycoords="axes fraction",
    ha="left",
    va="top",
  )
  return ax


def length_distribution(
  result: SimulationResult, *, ax: Axes | None = None, title: str | None = None, label: str | None = None
) -> Axes:
  """Plot how many of a run's sessions administered each number of items.

  Each test length has its own bin, drawn as an outline, so several runs drawn on the same axes stay
  readable over each other. In a variable-length run, the bin at the length cap counts the tests
  that reached the cap.

  .. plot::
      :caption: The test lengths of a variable-length run capped at 30 items.

      import matplotlib.pyplot as plt
      import numpy
      from catsim import CatEngine, simulate
      from irtlib import ItemBank
      from catsim import plot
      from catsim.estimation import EAPEstimator
      from catsim.initialization import FixedPointInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule

      engine = CatEngine(
          ItemBank.generate(100, rng=0),
          FixedPointInitializer(0.0),
          MaxInfoSelector(),
          EAPEstimator(),
          StopRule(StandardErrorStopper(0.5), MaxLengthStopper(30)),
      )
      result = simulate(engine, true_thetas=numpy.random.default_rng(1).normal(0, 1, 300), rng=2)
      plot.length_distribution(result)
      plt.show()

  Parameters
  ----------
  result : SimulationResult
      The run.
  ax : matplotlib.axes.Axes or None, optional
      The axes to draw on. Default is None, which draws on the axes of a new figure.
  title : str or None, optional
      The plot title. Default is None, which sets none.
  label : str or None, optional
      The run's label, for a legend. Default is None, which sets none.

  Returns
  -------
  matplotlib.axes.Axes
      The axes the plot is on.
  """
  lengths = result.examinees()["test_length"]
  edges = numpy.arange(lengths.min(), lengths.max() + 2) - 0.5
  ax = _prepare_axes(ax, title)
  ax.hist(lengths, edges, histtype="step", linewidth=plt.rcParams["lines.linewidth"], label=label)
  # One whole number in view is enough: the default of two makes a run of one length fall back to fractional ticks.
  ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
  ax.set_xlabel("Test length (items)")
  ax.set_ylabel("Sessions")
  return ax
