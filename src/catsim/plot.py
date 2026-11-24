"""Module with functions for plotting IRT-related results."""

from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy
from matplotlib.axes import Axes

from . import irt
from .item_bank import ItemBank
from .simulation import SimulationResult
from .state import CatSessionState
from .typing import FloatArray


class PlotType(Enum):
  """Enum with the available item plot types.

  Attributes
  ----------
  ICC : auto
      Item Characteristic Curve - plots probability of correct response vs ability.
  IIC : auto
      Item Information Curve - plots item information vs ability.
  BOTH : auto
      Both ICC and IIC plotted together.
  """

  ICC = auto()
  IIC = auto()
  BOTH = auto()


def item_curve(
  a: float = 1,
  b: float = 0,
  c: float = 0,
  d: float = 1,
  ax: Axes | None = None,
  title: str | None = None,
  ptype: PlotType = PlotType.ICC,
  max_info: bool = True,
  figsize: tuple[float, float] | None = None,
) -> Axes:
  """Plot Item Response Theory-related item plots.

  .. plot::
      :caption: Item characteristic and information functions for a given item. Last plot contains both curves together.

      import matplotlib.pyplot as plt
      from catsim.item_bank import ItemBank
      from catsim.plot import item_curve, PlotType
      item_bank = ItemBank.generate(1)
      item = item_bank.items[0]
      fig, axes = plt.subplots(3, 1, figsize=(7, 15))
      item_curve(item[0], item[1], item[2], item[3], ptype=PlotType.ICC, ax=axes[0])
      item_curve(item[0], item[1], item[2], item[3], ptype=PlotType.IIC, ax=axes[1])
      item_curve(item[0], item[1], item[2], item[3], ptype=PlotType.BOTH, ax=axes[2])
      plt.tight_layout()
      plt.show()

  When both curves are plotted in the same figure, the figure has no grid, since each
  curve has a different scale.

  Parameters
  ----------
  a : float, optional
      Item discrimination parameter. Default is 1.
  b : float, optional
      Item difficulty parameter. Default is 0.
  c : float, optional
      Item pseudo-guessing parameter. Default is 0.
  d : float, optional
      Item upper asymptote. Default is 1.
  ax : matplotlib.axes.Axes or None, optional
      Matplotlib axes object to plot on. If None, a new figure is created. Default is None.
  title : str or None, optional
      Plot title. Default is None.
  ptype : PlotType, optional
      Type of plot: PlotType.ICC for item characteristic curve, PlotType.IIC for item
      information curve, PlotType.BOTH for both curves. Default is PlotType.ICC.
  max_info : bool, optional
      Whether the point of maximum information should be shown in the plot. Default is True.
  figsize : tuple or None, optional
      Figure size (width, height) in inches. Default is None.

  Returns
  -------
  matplotlib.axes.Axes
      The matplotlib axes object containing the plot.
  """
  if ax is None:
    _, ax = plt.subplots(figsize=figsize)

  item = numpy.array([a, b, c, d])
  thetas = numpy.arange(b - 4, b + 4, 0.1)
  p_thetas = irt.icc(thetas, item)
  i_thetas = irt.information(thetas, item)

  if ptype in {PlotType.ICC, PlotType.IIC}:
    if title is not None:
      ax.set_title(title, size=18)

    _annotate_parameters(ax, a, b, c, d)
    ax.set_xlabel(r"$\theta$")
    ax.grid()

    if ptype == PlotType.ICC:
      ax.set_ylabel(r"$P(\theta)$")
      ax.plot(thetas, p_thetas, label=r"$P(\theta)$")

    elif ptype == PlotType.IIC:
      ax.set_ylabel(r"$I(\theta)$")
      ax.plot(thetas, i_thetas, label=r"$I(\theta)$")
      if max_info:
        aux = irt.max_info_theta(item)
        ax.plot(aux, irt.information(aux, item), "o")

  elif ptype == PlotType.BOTH:
    ax.set_xlabel(r"$\theta$", size=16)
    ax.set_ylabel(r"$P(\theta)$", color="b", size=16)
    ax.plot(thetas, p_thetas, "b-", label=r"$P(\theta)$")
    # Make the y-axis label and tick labels match the line color.
    for tl in ax.get_yticklabels():
      tl.set_color("b")

    ax2 = ax.twinx()
    ax2.set_ylabel(r"$I(\theta)$", color="r", size=16)
    ax2.plot(thetas, i_thetas, "r-", label=r"$I(\theta)$")
    for tl in ax2.get_yticklabels():
      tl.set_color("r")

    if max_info:
      aux = irt.max_info_theta(item)
      ax2.plot(aux, irt.information(aux, item), "o")

    if title is not None:
      ax.set_title(title, size=18)

    _annotate_parameters(ax2, a, b, c, d)

  return ax


def _annotate_parameters(ax: Axes, a: float, b: float, c: float, d: float) -> None:
  """Write the item parameters, to three significant digits, in a box in the upper left corner of the axes.

  The curves of typical items keep clear of that corner: at the low end of the ability range the ICC is
  near its lower asymptote and the IIC near zero. A very flat item with d < 1 (a around 0.1) is the
  exception, as its information curve leans left into the corner.
  """
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


def item_exposure(
  ax: Axes | None = None,
  title: str | None = None,
  simulation: SimulationResult | None = None,
  item_bank: ItemBank | FloatArray | None = None,
  exposure_rates: FloatArray | None = None,
  par: str | None = None,
  hist: bool = False,
) -> Axes:
  """Generate a plot showing item bank exposure rates.

  The plot visualizes how frequently each item was administered during a simulation,
  which is important for assessing item security and test balance.

  .. plot::
      :caption: Item exposure rates for a given item bank, after a simulation has been performed.

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
      plot.item_exposure(title='Exposures', simulation=result, hist=True, ax=axes[0])
      plot.item_exposure(title='Exposures', simulation=result, par='b', ax=axes[1])
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  ax : matplotlib.axes.Axes or None, optional
      Matplotlib axes object to plot on. If None, a new figure is created. Default is None.
  title : str or None, optional
      The plot title. Default is None.
  simulation : SimulationResult or None, optional
      Simulation result containing exposure data and the item bank. Pass either this or
      ``exposure_rates``. Default is None.
  item_bank : ItemBank or numpy.ndarray or None, optional
      An ItemBank, or an item matrix of [a, b, c, d] parameters, which is converted to an ItemBank.
      Only used with ``exposure_rates``, and required when ``par`` is given. Default is None.
  exposure_rates : numpy.ndarray or None, optional
      The exposure rate of each item, in item bank order. Pass either this or ``simulation``.
      Default is None.
  par : str or None, optional
      A string representing one of the item parameters ('a', 'b', 'c', 'd') to plot on
      the x axis, or None to plot the exposures in item bank order. If `hist=True`, no
      sorting will be done. Default is None.
  hist : bool, optional
      If True, plots a histogram of item exposures. Otherwise, plots a line chart of
      the exposures against the parameter chosen in `par`. Default is False.

  Returns
  -------
  matplotlib.axes.Axes
      The matplotlib axes object containing the plot.

  Raises
  ------
  ValueError
      If neither or both of ``simulation`` and ``exposure_rates`` are given, if ``item_bank`` is
      passed with ``simulation``, if ``par`` is not one of 'a', 'b', 'c', 'd' or None, if ``par``
      is given with ``exposure_rates`` but no ``item_bank``, or if ``exposure_rates`` and
      ``item_bank`` differ in length.
  """
  if simulation is not None:
    if item_bank is not None or exposure_rates is not None:
      msg = "Pass either simulation or exposure_rates (with item_bank), not both; simulation already holds them."
      raise ValueError(msg)
    item_bank = simulation.item_bank
    exposure_rates = simulation.exposure_rates
  if exposure_rates is None:
    msg = "Exposure data must be passed, either as simulation or as exposure_rates."
    raise ValueError(msg)

  supported_parameters = ("a", "b", "c", "d")
  if par is not None and par not in supported_parameters:
    msg = f"Unsupported parameter {par!r} for 'par'. Supported parameters are: {', '.join(supported_parameters)}."
    raise ValueError(msg)

  if isinstance(item_bank, numpy.ndarray):
    item_bank = ItemBank(item_bank)
  if item_bank is None:
    if par is not None:
      msg = f"item_bank must be passed with exposure_rates to plot exposure against parameter {par!r}."
      raise ValueError(msg)
  elif len(exposure_rates) != item_bank.n_items:
    msg = f"exposure_rates has {len(exposure_rates)} values but the item bank has {item_bank.n_items} items."
    raise ValueError(msg)

  if ax is None:
    _, ax = plt.subplots()

  if title is not None:
    ax.set_title(title, size=18)

  if item_bank is None or par is None:
    parameter = numpy.arange(len(exposure_rates))
    xlabel = "Items"
  elif par == "a":
    parameter = item_bank.discrimination
    xlabel = "Item discrimination"
  elif par == "b":
    parameter = item_bank.difficulty
    xlabel = "Item difficulty"
  elif par == "c":
    parameter = item_bank.pseudo_guessing
    xlabel = "Item Guessing"
  else:
    parameter = item_bank.upper_asymptote
    xlabel = "Item upper asymptote"

  if hist:
    ax.hist(exposure_rates, max(len(exposure_rates) // 10, 3))
    ax.set_xlabel("Item exposure")
    ax.set_ylabel("Items")
  else:
    indexes = parameter.argsort(kind="stable")
    ax.plot(parameter[indexes], exposure_rates[indexes], marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Item exposure")

  return ax


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
