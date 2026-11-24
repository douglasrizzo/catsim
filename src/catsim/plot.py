"""Plots of items, exposure rates and CAT sessions, drawn with matplotlib.

The module needs matplotlib, which the plot extra installs: ``pip install 'catsim[plot]'``. Each
function takes its data first and its options as keyword-only arguments, draws on the ``ax`` it is
given or on the axes of a new figure, and returns the axes it drew on.
"""

from enum import StrEnum
from typing import assert_never

import numpy

try:
  import matplotlib.pyplot as plt
  from matplotlib.axes import Axes
except ImportError as error:
  _missing = "catsim.plot needs matplotlib; install it with: pip install 'catsim[plot]'"
  raise ImportError(_missing) from error

from . import irt
from .item_bank import ItemBank
from .simulation import SimulationResult
from .state import CatSessionState
from .typing import FloatArray


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
  item: FloatArray,
  *,
  kind: PlotType | str = PlotType.ICC,
  max_info: bool = True,
  ax: Axes | None = None,
  title: str | None = None,
) -> Axes:
  r"""Plot an item's characteristic curve, its information curve, or both.

  The curves span four units of ability on each side of the item's difficulty, and a box in the
  upper left corner gives the item's parameters. With ``PlotType.BOTH`` the information curve has
  its own y axis, on the right, and the plot has no grid.

  .. plot::
      :caption: One item's characteristic and information curves, apart and together. The dot marks its peak.

      import matplotlib.pyplot as plt
      from catsim import ItemBank
      from catsim.plot import PlotType, item_curve
      item = ItemBank.generate(1, rng=7).items[0]
      fig, axes = plt.subplots(3, 1, figsize=(7, 15))
      item_curve(item, kind=PlotType.ICC, ax=axes[0])
      item_curve(item, kind=PlotType.IIC, ax=axes[1])
      item_curve(item, kind=PlotType.BOTH, ax=axes[2])
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  item : numpy.ndarray
      One item's parameters ``[a, b, c, d]``, such as a row of ``ItemBank.items``.
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
  ValueError
      If ``item`` is not one row of four parameters or ``kind`` is not a :class:`PlotType`,
      before any figure is created.
  """
  item = numpy.asarray(item, dtype=float)
  if item.shape != (4,):
    msg = f"item must be one row of four parameters [a, b, c, d], got shape {item.shape}."
    raise ValueError(msg)
  kind = PlotType(kind)
  ax = _prepare_axes(ax, title)
  thetas = numpy.linspace(item[1] - 4, item[1] + 4, 161)

  match kind:
    case PlotType.ICC:
      _draw_icc(ax, thetas, item)
      ax.grid()
      box_axes = ax
    case PlotType.IIC:
      _draw_information(ax, thetas, item, max_info=max_info)
      ax.grid()
      box_axes = ax
    case PlotType.BOTH:
      box_axes = ax.twinx()
      for axes, color in ((ax, "C0"), (box_axes, "C3")):
        axes.yaxis.label.set_color(color)
        axes.tick_params(axis="y", colors=color)
      _draw_icc(ax, thetas, item, color="C0")
      _draw_information(box_axes, thetas, item, max_info=max_info, color="C3")
    case _:
      assert_never(kind)

  ax.set_xlabel(r"$\theta$")
  _annotate_parameters(box_axes, item)
  return ax


def _prepare_axes(ax: Axes | None, title: str | None) -> Axes:
  """Return ``ax``, or the axes of a new figure, with ``title`` set when one is given."""
  if ax is None:
    _, ax = plt.subplots()
  if title is not None:
    ax.set_title(title, size=18)
  return ax


def _draw_icc(ax: Axes, thetas: FloatArray, item: FloatArray, color: str | None = None) -> None:
  """Draw the item's characteristic curve, in ``color`` or the next colour of the axes' cycle."""
  ax.plot(thetas, irt.icc(thetas, item), color=color, label=r"$P(\theta)$")
  ax.set_ylabel(r"$P(\theta)$")


def _draw_information(
  ax: Axes, thetas: FloatArray, item: FloatArray, *, max_info: bool, color: str | None = None
) -> None:
  """Draw the item's information curve and, with ``max_info``, a dot of the curve's colour at its peak."""
  (line,) = ax.plot(thetas, irt.information(thetas, item), color=color, label=r"$I(\theta)$")
  ax.set_ylabel(r"$I(\theta)$")
  if max_info:
    peak = irt.max_info_theta(item)
    ax.plot(peak, irt.information(peak, item), "o", color=line.get_color(), label="Maximum information")


def _annotate_parameters(ax: Axes, item: FloatArray) -> None:
  """Write the item parameters, to three significant digits, in a box in the upper left corner of the axes."""
  a, b, c, d = item
  ax.annotate(
    f"$a = {a:.3g}$\n$b = {b:.3g}$\n$c = {c:.3g}$\n$d = {d:.3g}$",
    bbox={"facecolor": "white", "alpha": 1},
    xy=(0.02, 0.97),
    xycoords="axes fraction",
    ha="left",
    va="top",
  )


def gen3d_dataset_scatter(
  item_bank: ItemBank | FloatArray,
  title: str | None = None,
  figsize: tuple[float, float] | None = None,
) -> Axes:
  """Generate a 3D scatter plot of item parameters.

  Creates a three-dimensional visualization of item parameters (a, b, c) to help
  understand the distribution of item characteristics in the item bank.

  .. plot::

      import matplotlib.pyplot as plt
      from catsim import plot
      from catsim.item_bank import ItemBank
      item_bank = ItemBank.generate(100)
      plot.gen3d_dataset_scatter(item_bank)
      plt.show()

  Parameters
  ----------
  item_bank : ItemBank or numpy.ndarray
      An ItemBank or item matrix containing item parameters.
      If a numpy array is provided, it will be converted to an ItemBank.
  title : str or None, optional
      The scatter plot title. Default is None.
  figsize : tuple or None, optional
      Figure size (width, height) in inches. Default is None.

  Returns
  -------
  matplotlib.axes.Axes
      The matplotlib 3D axes object containing the plot.
  """
  if isinstance(item_bank, numpy.ndarray):
    item_bank = ItemBank(item_bank)

  fig = plt.figure(figsize=figsize)
  ax = fig.add_subplot(111, projection="3d")
  ax.scatter(
    list(item_bank.discrimination), list(item_bank.difficulty), zs=list(item_bank.pseudo_guessing), s=10, c="b"
  )

  if title is not None:
    ax.set_title(title, size=18)

  ax.set_xlabel("a")
  ax.set_ylabel("b")
  ax.set_zlabel("c")

  return ax


# The item parameter each value of item_exposure's ``par`` reads: its column in the item matrix and the axis label.
_ITEM_PARAMETERS = {
  "a": (0, "Item discrimination"),
  "b": (1, "Item difficulty"),
  "c": (2, "Item guessing parameter"),
  "d": (3, "Item upper asymptote"),
}


def item_exposure(
  exposure_rates: FloatArray,
  *,
  item_bank: ItemBank | None = None,
  par: str | None = None,
  hist: bool = False,
  ax: Axes | None = None,
  title: str | None = None,
) -> Axes:
  """Plot the exposure rate of each item: in item order, against an item parameter, or as a histogram.

  With ``par``, the items are sorted by that parameter and each item's exposure rate is drawn at its
  value, which shows which items a selector favours. With ``hist=True``, the plot counts the items
  in each range of exposure rates.

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
      result = SimulationRunner(engine).run(true_thetas)
      fig, axes = plt.subplots(2, 1, figsize=(7, 12))
      plot.item_exposure(result.exposure_rates, hist=True, ax=axes[0], title="Exposure rates")
      plot.item_exposure(result.exposure_rates, item_bank=result.item_bank, par="b", ax=axes[1])
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  exposure_rates : numpy.ndarray
      The exposure rate of each item, in item bank order, such as ``SimulationResult.exposure_rates``.
  item_bank : ItemBank or None, optional
      The item bank the rates belong to, needed with ``par``. Default is None.
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
      If ``exposure_rates`` is not one-dimensional, ``par`` is not one of ``"a"``, ``"b"``, ``"c"``
      and ``"d"``, ``par`` is given with ``hist=True`` or without ``item_bank``, or the rates and
      the item bank differ in length; always before any figure is created.
  """
  exposure_rates = numpy.asarray(exposure_rates, dtype=float)
  _check_exposure_inputs(exposure_rates, item_bank, par, hist=hist)
  ax = _prepare_axes(ax, title)

  if hist:
    ax.hist(exposure_rates, max(len(exposure_rates) // 10, 3))
    ax.set_xlabel("Exposure rate")
    ax.set_ylabel("Items")
    return ax

  if par is None or item_bank is None:
    values = numpy.arange(len(exposure_rates))
    xlabel = "Items"
  else:
    column, xlabel = _ITEM_PARAMETERS[par]
    values = item_bank.items[:, column]
  order = values.argsort(kind="stable")
  ax.plot(values[order], exposure_rates[order], marker="o")
  ax.set_xlabel(xlabel)
  ax.set_ylabel("Exposure rate")
  return ax


def _check_exposure_inputs(
  exposure_rates: FloatArray, item_bank: ItemBank | None, par: str | None, *, hist: bool
) -> None:
  """Raise ValueError for inputs of :func:`item_exposure` that cannot give a true plot."""
  if exposure_rates.ndim != 1:
    msg = f"exposure_rates must hold one rate per item, got an array of shape {exposure_rates.shape}."
    raise ValueError(msg)
  if item_bank is not None and len(exposure_rates) != item_bank.n_items:
    msg = f"exposure_rates has {len(exposure_rates)} values but the item bank has {item_bank.n_items} items."
    raise ValueError(msg)
  if par is None:
    return
  if par not in _ITEM_PARAMETERS:
    msg = f"Unsupported parameter {par!r} for 'par'. Supported parameters are: {', '.join(_ITEM_PARAMETERS)}."
    raise ValueError(msg)
  if hist:
    msg = f"A histogram of exposure rates has no parameter axis: pass par={par!r} or hist=True, not both."
    raise ValueError(msg)
  if item_bank is None:
    msg = f"item_bank must be passed to plot exposure rates against parameter {par!r}."
    raise ValueError(msg)


def test_progress(
  ax: Axes | None = None,
  title: str | None = None,
  simulation: SimulationResult | None = None,
  index: int | None = None,
  session: CatSessionState | None = None,
  thetas: list[float] | None = None,
  administered_items: FloatArray | None = None,
  true_theta: float | None = None,
  info: bool = False,
  var: bool = False,
  see: bool = False,
  reliability: bool = False,
  marker: str | int | None = None,
  figsize: tuple[float, float] | None = None,
) -> Axes:
  """Generate a plot representing an examinee's test progress over time.

  The plot shows how the ability estimate, item difficulties, and measurement quality
  metrics evolve as items are administered during the test.

  Note that, while some functions increase or decrease monotonically (like test
  information and standard error of estimation), the plot calculates these values using
  the examinee's ability estimated at that given time of the test. This means that a test
  that was thought to be informative at a given point may not be as informative after new
  estimates are made. The value at :math:`x` items uses the estimate made after those
  :math:`x` items, so these curves start at the first item.

  .. plot::

      import matplotlib.pyplot as plt
      import numpy
      from catsim.item_bank import ItemBank
      from catsim import plot
      from catsim.initialization import UniformInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.estimation import NumericalSearchEstimator
      from catsim.stopping import MaxLengthStopper, StandardErrorStopper, StopRule
      from catsim.engine import CatEngine
      from catsim.simulation import SimulationRunner

      fig, axes = plt.subplots(2, 1, figsize=(7, 12))
      engine = CatEngine(
          ItemBank.generate(100),
          UniformInitializer(),
          MaxInfoSelector(),
          NumericalSearchEstimator(),
          StopRule(StandardErrorStopper(0.4), MaxLengthStopper(20)),
      )
      true_thetas = numpy.random.default_rng(1).normal(0, 1, 10)
      result = SimulationRunner(engine).run(true_thetas)
      plot.test_progress(simulation=result, index=0, ax=axes[0])
      plot.test_progress(simulation=result, index=0, info=True, var=True, see=True, ax=axes[1])
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  ax : matplotlib.axes.Axes or None, optional
      Axis to use. If None, a figure with the necessary axis will be created. Default is None.
  title : str or None, optional
      The plot title. Default is None.
  simulation : SimulationResult or None, optional
      Simulation result containing session traces. Default is None.
  index : int or None, optional
      The index of the session to be plotted when ``simulation`` is provided.
  session : CatSessionState or None, optional
      Not plotted on its own: passing it without ``simulation`` raises ``ValueError``, and with
      ``simulation`` the session at ``index`` is plotted instead. Default is None.
  thetas : list[float] or None, optional
      If a simulation result is not passed, then a list of ability estimations can be manually
      passed to the function. Default is None.
  administered_items : numpy.ndarray or None, optional
      If a simulation result is not passed, then a matrix of administered items, represented
      by their parameters, can be manually passed to the function. Default is None.
  true_theta : float or None, optional
      The value of the examinee's true ability. If it is passed, it will be shown on
      the plot, otherwise not. Default is None.
  info : bool, optional
      Plot test information. It only works if both abilities and administered items are
      passed. Default is False.
  var : bool, optional
      Plot the estimation variance during the test. It only works if both abilities
      and administered items are passed. Default is False.
  see : bool, optional
      Plot the standard error of estimation during the test. It only works if both
      abilities and administered items are passed. Default is False.
  reliability : bool, optional
      Plot the test reliability. It only works if both abilities and administered items
      are passed. Default is False.
  marker : str or int or None, optional
      Matplotlib marker style for the plots. Default is None.
  figsize : tuple or None, optional
      Size of the figure to be created, in case no axis is passed. Default is None.

  Returns
  -------
  matplotlib.axes.Axes
      The matplotlib axes object containing the plot.

  Raises
  ------
  ValueError
      If ``simulation`` is passed without ``index``, ``session`` without ``simulation``, or
      neither ``simulation`` nor both ``thetas`` and ``administered_items``; or if ``thetas``
      does not hold one more estimate than there are administered items.
  """
  if simulation is None and session is None and thetas is None and administered_items is None:
    msg = "One of: simulation, session, thetas, administered_items must be passed."
    raise ValueError(msg)

  if ax is None:
    _, ax = plt.subplots(figsize=figsize)

  if title is not None:
    ax.set_title(title, size=18)

  if simulation is not None:
    if index is None:
      msg = "index must be provided when simulation is passed"
      raise ValueError(msg)
    session = simulation.sessions[index]
    thetas = list(session.theta_history)
    administered_items = simulation.item_bank.get_items(session.administered_items)
    true_theta = session.true_theta

  if session is not None and simulation is None:
    msg = "session plotting currently requires the parent simulation result"
    raise ValueError(msg)

  if thetas is None:
    msg = "thetas must be provided"
    raise ValueError(msg)
  if administered_items is None:
    msg = "administered_items must be provided"
    raise ValueError(msg)

  if len(thetas) - 1 != len(administered_items[:, 1]):
    msg = "Number of estimated thetas and administered items is not the same."
    raise ValueError(msg)

  # x counts the items answered: thetas[0] comes from the initializer and thetas[x] follows the
  # response to item x, which is plotted at x.
  xs = list(range(len(thetas)))
  ax.plot(xs, thetas, label=r"$\hat{\theta}$", marker=marker)
  ax.plot(xs[1:], administered_items[:, 1], label="Item difficulty", marker=marker)
  if true_theta is not None:
    ax.hlines(true_theta, 0, len(xs) - 1, label=r"$\theta$", colors="black", linestyles="dashed")

  # Each quality measure at x uses thetas[x] and the first x items. The curves start at x = 1:
  # with no items the information is 0 and the variance, SEE and reliability are infinite.
  quality_xs = xs[1:]
  for enabled, measure, label in (
    (info, irt.total_information, r"$I(\theta)$"),
    (var, irt.var, r"$Var$"),
    (see, irt.see, r"$SEE$"),
    (reliability, irt.reliability, "Reliability"),
  ):
    if enabled:
      values = [measure(thetas[x], administered_items[:x]) for x in quality_xs]
      ax.plot(quality_xs, values, label=label, marker=marker)
  ax.set_xlabel("Items")
  ax.grid()
  ax.legend(loc="best")

  return ax
