"""Plots of items, exposure rates and CAT sessions, drawn with matplotlib.

The module needs matplotlib, which the plot extra installs: ``pip install 'catsim[plot]'``. Each
function takes its data first, as plain catsim objects, and its options as keyword-only arguments,
draws on the ``ax`` it is given or on the axes of a new figure, and returns the axes it drew on
(ADR-0022).

Plots label items by their names in the item bank (ADR-0004), drawn as given: a dollar sign in a
name does not start mathtext. An x axis with one tick per item names each item while it has at most
30, the most names that stay apart at matplotlib's default figure width; a longer one ticks whole
numbers, as each function's description says.
"""

from collections.abc import Sequence
from enum import StrEnum
from typing import assert_never

import numpy

try:
  import matplotlib.pyplot as plt
  from matplotlib.axes import Axes
  from matplotlib.ticker import MaxNLocator
except ImportError as error:
  _missing = "catsim.plot needs matplotlib; install it with: pip install 'catsim[plot]'"
  raise ImportError(_missing) from error

from . import irt
from .checks import _check_in_bank, _item
from .item_bank import ItemBank
from .session import Session
from .typing import FloatArray, IntArray

# The most items an x axis names, one tick each: at matplotlib's default figure width, more names overlap.
_MAX_NAMED_TICKS = 30


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
  curve has its own y axis, on the right, and the plot has no grid.

  .. plot::
      :caption: One item's characteristic and information curves, apart and together. The dot marks its peak.

      import matplotlib.pyplot as plt
      from catsim import ItemBank
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
      ax.grid()
      box_axes = ax
    case PlotType.IIC:
      _draw_information(ax, thetas, parameters, max_info=max_info)
      ax.grid()
      box_axes = ax
    case PlotType.BOTH:
      box_axes = ax.twinx()
      for axes, color in ((ax, "C0"), (box_axes, "C3")):
        axes.yaxis.label.set_color(color)
        axes.tick_params(axis="y", colors=color)
      _draw_icc(ax, thetas, parameters, color="C0")
      _draw_information(box_axes, thetas, parameters, max_info=max_info, color="C3")
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
    ax.set_title(title, size=18)
  return ax


def _label_items(ax: Axes, positions: IntArray, names: Sequence[str]) -> None:
  """Tick the x axis at ``positions`` with the ``names`` of the items there, or at whole numbers if there are too many.

  Names are drawn as given, vertically so that long ones fit. With more than ``_MAX_NAMED_TICKS`` names, matplotlib
  picks the ticks among the whole numbers, which also holds in a view zoomed in to a few positions.
  """
  if len(names) <= _MAX_NAMED_TICKS:
    ax.set_xticks(positions, names, rotation=90, parse_math=False)
  else:
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def _draw_icc(ax: Axes, thetas: FloatArray, parameters: FloatArray, *, color: str | None = None) -> None:
  """Draw the item's characteristic curve, in ``color`` or the next colour of the axes' cycle."""
  ax.plot(thetas, irt.icc(thetas, parameters), color=color, label=r"$P(\theta)$")
  ax.set_ylabel(r"$P(\theta)$")


def _draw_information(
  ax: Axes, thetas: FloatArray, parameters: FloatArray, *, max_info: bool, color: str | None = None
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


# The item parameter each value of item_exposure's ``par`` reads: its column in the item matrix and the axis label.
_ITEM_PARAMETERS = {
  "a": (0, "Item discrimination"),
  "b": (1, "Item difficulty"),
  "c": (2, "Item guessing parameter"),
  "d": (3, "Item upper asymptote"),
}


def item_exposure(
  exposure_rates: FloatArray,
  item_bank: ItemBank,
  *,
  par: str | None = None,
  hist: bool = False,
  ax: Axes | None = None,
  title: str | None = None,
) -> Axes:
  """Plot the exposure rate of each item: in item order, against an item parameter, or as a histogram.

  In item order, the x axis names each item, or ticks item indices in a bank of more than 30 items.
  With ``par``, the items are sorted by that parameter and each item's exposure rate is drawn at its
  value, which shows which items a selector favours; items of close parameters crowd together there,
  so their names are left out. With ``hist=True``, the plot counts the items in each range of
  exposure rates.

  .. plot::
      :caption: The exposure rates of a simulation, as a histogram and against the item difficulty.

      import matplotlib.pyplot as plt
      import numpy
      from catsim import CatEngine, ItemBank, SimulationRunner
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
      result = SimulationRunner(engine).run(true_thetas, rng=0)
      fig, axes = plt.subplots(2, 1, figsize=(7, 12))
      plot.item_exposure(result.exposure_rates, result.item_bank, hist=True, ax=axes[0], title="Exposure rates")
      plot.item_exposure(result.exposure_rates, result.item_bank, par="b", ax=axes[1])
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  exposure_rates : numpy.ndarray
      The exposure rate of each item, in item bank order, such as ``SimulationResult.exposure_rates``.
  item_bank : ItemBank
      The item bank the rates belong to, which holds the items' names and parameters.
  par : str or None, optional
      The item parameter to plot the rates against: ``"a"``, ``"b"``, ``"c"`` or ``"d"``. Default is
      None, which plots the rates in item bank order.
  hist : bool, optional
      Whether to draw a histogram of the rates. Default is False.
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
      If ``exposure_rates`` is not one rate per item of ``item_bank``, ``par`` is not one of
      ``"a"``, ``"b"``, ``"c"`` and ``"d"``, or ``par`` is given with ``hist=True``; always before
      any figure is created.
  """
  exposure_rates = numpy.asarray(exposure_rates, dtype=float)
  _check_exposure_inputs(exposure_rates, item_bank, par, hist=hist)
  ax = _prepare_axes(ax, title)

  if hist:
    ax.hist(exposure_rates, max(len(exposure_rates) // 10, 3))
    ax.set_xlabel("Exposure rate")
    ax.set_ylabel("Items")
    return ax

  if par is None:
    indices = numpy.arange(item_bank.n_items)
    ax.plot(indices, exposure_rates, marker="o")
    _label_items(ax, indices, item_bank.names)
    ax.set_xlabel("Items")
  else:
    column, xlabel = _ITEM_PARAMETERS[par]
    values = item_bank.items[:, column]
    order = values.argsort(kind="stable")
    ax.plot(values[order], exposure_rates[order], marker="o")
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
  no items has no ticks. The left y axis shows the ability estimate after each response, with a
  band of one standard error on each side, the difficulty of each administered item and, when it is
  given, the true ability. The right y axis shows the standard error each estimate was recorded
  with, the one precision-based stoppers judge. Infinite standard errors, such as the initial
  estimate's, are left out.

  .. plot::
      :caption: One simulated session: the ability estimate after each item, with its standard error.

      import matplotlib.pyplot as plt
      import numpy
      from catsim import CatEngine, ItemBank, SimulationRunner
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
      result = SimulationRunner(engine).run(true_thetas, rng=0)
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
      The axes of the ability scale. The standard error is drawn on their twin, which shares the x
      axis.

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
  se_axes = ax.twinx()
  # The legend is on ``ax``, which must be drawn over the twin's line to stay readable.
  ax.set_zorder(se_axes.get_zorder() + 1)
  ax.patch.set_visible(False)

  ax.plot(xs, thetas, marker=".", color="C0", label=r"Ability estimate $\hat{\theta}$")
  ax.fill_between(
    xs[finite], (thetas - ses)[finite], (thetas + ses)[finite], color="C0", alpha=0.2, label=r"$\hat{\theta} \pm SE$"
  )
  ax.plot(xs[1:], difficulties, "o", color="C1", label="Item difficulty")
  if true_theta is not None:
    ax.axhline(true_theta, color="black", linestyle="--", label=r"True ability $\theta$")
  se_axes.plot(xs[finite], ses[finite], color="C2", label="Standard error")

  ax.set_xlabel("Items administered")
  _label_items(ax, xs[1:], [item_bank.names[item] for item in session.administered_items])
  ax.set_ylabel(r"$\theta$")
  se_axes.set_ylabel("Standard error", color="C2")
  se_axes.tick_params(axis="y", colors="C2")
  ax.grid()
  handles, labels = ax.get_legend_handles_labels()
  se_handles, se_labels = se_axes.get_legend_handles_labels()
  ax.legend(handles + se_handles, labels + se_labels, loc="best")
  return ax
