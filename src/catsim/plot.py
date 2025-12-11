"""Module with functions for plotting IRT-related results."""

from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy
import numpy.typing as npt
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

from . import irt
from .item_bank import ItemBank
from .simulation import Simulator


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
  figsize: tuple | None = None,
) -> Axes:
  """Plot Item Response Theory-related item plots.

  .. plot::
      :caption: Item characteristic and information functions for a given item. Last plot contains both curves together.

      import matplotlib.pyplot as plt
      from catsim.item_bank import ItemBank
      from catsim.plot import item_curve, PlotType
      item_bank = ItemBank.generate_item_bank(1)
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

  thetas = numpy.arange(b - 4, b + 4, 0.1, "double")
  p_thetas = []
  i_thetas = []
  for theta in thetas:
    p_thetas.append(irt.icc(theta, a, b, c, d))
    i_thetas.append(irt.inf(theta, a, b, c, d))

  if ptype in {PlotType.ICC, PlotType.IIC}:
    if title is not None:
      ax.set_title(title, size=18)

    ax.annotate(
      "$a = " + format(a) + "$\n$b = " + format(b) + "$\n$c = " + format(c) + "$\n$d = " + format(d) + "$",
      bbox={"facecolor": "white", "alpha": 1},
      xy=(0.75, 0.05),
      xycoords="axes fraction",
    )
    ax.set_xlabel(r"$\theta$")
    ax.grid()

    if ptype == PlotType.ICC:
      ax.set_ylabel(r"$P(\theta)$")
      ax.plot(thetas, p_thetas, label=r"$P(\theta)$")

    elif ptype == PlotType.IIC:
      ax.set_ylabel(r"$I(\theta)$")
      ax.plot(thetas, i_thetas, label=r"$I(\theta)$")
      if max_info:
        aux = irt.max_info(a, b, c, d)
        ax.plot(aux, irt.inf(aux, a, b, c, d), "o")

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
      aux = irt.max_info(a, b, c, d)
      ax2.plot(aux, irt.inf(aux, a, b, c, d), "o")

    if title is not None:
      ax.set_title(title, size=18)

    ax2.annotate(
      "$a = " + format(a) + "$\n$b = " + format(b) + "$\n$c = " + format(c) + "$\n$d = " + format(d) + "$",
      bbox={"facecolor": "white", "alpha": 1},
      xy=(0.75, 0.05),
      xycoords="axes fraction",
    )

  return ax


def gen3d_dataset_scatter(
  item_bank: ItemBank | npt.NDArray[numpy.floating],
  title: str | None = None,
  figsize: tuple | None = None,
) -> Axes:
  """Generate a 3D scatter plot of item parameters.

  Creates a three-dimensional visualization of item parameters (a, b, c) to help
  understand the distribution of item characteristics in the item bank.

  .. plot::

      import matplotlib.pyplot as plt
      from catsim import plot
      from catsim.item_bank import ItemBank
      item_bank = ItemBank.generate_item_bank(100)
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
  assert Axes3D
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
  simulator: Simulator | None = None,
  item_bank: ItemBank | npt.NDArray[numpy.floating] | None = None,
  par: str | None = None,
  hist: bool = False,
) -> Axes:
  """Generate a plot showing item bank exposure rates.

  The plot visualizes how frequently each item was administered during a simulation,
  which is important for assessing item security and test balance.

  .. plot::
      :caption: Item exposure rates for a given item bank, after a simulation has been performed.

      import matplotlib.pyplot as plt
      from catsim.item_bank import ItemBank
      from catsim import plot
      from catsim.initialization import RandomInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.estimation import NumericalSearchEstimator
      from catsim.stopping import MinErrorStopper
      from catsim.simulation import Simulator

      fig, axes = plt.subplots(2, 1, figsize=(7, 12))

      s = Simulator(ItemBank.generate_item_bank(100), 10)
      s.simulate(RandomInitializer(), MaxInfoSelector(), NumericalSearchEstimator(), MinErrorStopper(0.4, max_items=20))
      plot.item_exposure(title='Exposures', simulator=s, hist=True, ax=axes[0])
      plot.item_exposure(title='Exposures', simulator=s, par='b', ax=axes[1])
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  ax : matplotlib.axes.Axes or None, optional
      Matplotlib axes object to plot on. If None, a new figure is created. Default is None.
  title : str or None, optional
      The plot title. Default is None.
  simulator : Simulator or None, optional
      A simulator which has already simulated a series of CATs, containing estimations
      to the examinees' abilities and a list of administered items for each examinee.
      Default is None.
  item_bank : ItemBank or numpy.ndarray or None, optional
      An ItemBank or item matrix containing item parameters and their exposure rate in the last column.
      If a numpy array is provided, it will be converted to an ItemBank.
      Default is None.
  par : str or None, optional
      A string representing one of the item parameters ('a', 'b', 'c', 'd') to order
      the items by and use on the x axis, or None to use the default order of the item
      bank. If `hist=True`, no sorting will be done. Default is None.
  hist : bool, optional
      If True, plots a histogram of item exposures. Otherwise, plots a line chart of
      the exposures, sorted in the x-axis by the parameter chosen in `par`. Default is False.

  Returns
  -------
  matplotlib.axes.Axes
      The matplotlib axes object containing the plot.

  Raises
  ------
  ValueError
      If neither simulator nor item_bank is provided, or if par is not one of 'a', 'b', 'c', 'd', or None.
  """
  if simulator is None and item_bank is None:
    msg = "Not a single plottable object was passed. Either 'simulator' or 'item_bank' must be passed."
    raise ValueError(msg)

  if ax is None:
    _, ax = plt.subplots()

  if title is not None:
    ax.set_title(title, size=18)

  if simulator is not None:
    item_bank = simulator.item_bank
  elif isinstance(item_bank, numpy.ndarray):
    item_bank = ItemBank(item_bank)

  assert item_bank is not None

  supported_parameters = {"a", "b", "c", "d"}
  if par is not None and par not in supported_parameters:
    msg = "Unsupported parameter 'par'. Supported parameters are: " + ", ".join(supported_parameters) + "."
    raise ValueError(msg)

  if par == "a":
    parameter = item_bank.discrimination
    xlabel = "Item discrimination"
  elif par == "b":
    parameter = item_bank.difficulty
    xlabel = "Item difficulty"
  elif par == "c":
    parameter = item_bank.pseudo_guessing
    xlabel = "Item Guessing"
  elif par == "d":
    parameter = item_bank.upper_asymptote
    xlabel = "Item upper asymptote"
  else:
    parameter = numpy.array(range(item_bank.n_items))
    xlabel = "Items"

  if hist:
    ax.hist(item_bank.exposure_rates, max(int(item_bank.n_items / 10), 3))
    ax.set_xlabel("Item exposure")
    ax.set_ylabel("Items")
  else:
    indexes = parameter.argsort()
    ax.plot(item_bank.exposure_rates[indexes], marker="o")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Item exposure")

  # Only show legend if there are labeled artists
  handles, _ = ax.get_legend_handles_labels()
  if handles:
    plt.legend(loc="best")

  return ax


def test_progress(
  ax: Axes | None = None,
  title: str | None = None,
  simulator: Simulator | None = None,
  index: int | None = None,
  thetas: list[float] | None = None,
  administered_items: npt.NDArray[numpy.floating] | None = None,
  true_theta: float | None = None,
  info: bool = False,
  var: bool = False,
  see: bool = False,
  reliability: bool = False,
  marker: str | int | None = None,
  figsize: tuple | None = None,
) -> Axes:
  """Generate a plot representing an examinee's test progress over time.

  The plot shows how the ability estimate, item difficulties, and measurement quality
  metrics evolve as items are administered during the test.

  Note that, while some functions increase or decrease monotonically (like test
  information and standard error of estimation), the plot calculates these values using
  the examinee's ability estimated at that given time of the test. This means that a test
  that was thought to be informative at a given point may not be as informative after new
  estimates are made.

  .. plot::

      import matplotlib.pyplot as plt
      from catsim.item_bank import ItemBank
      from catsim import plot
      from catsim.initialization import RandomInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.estimation import NumericalSearchEstimator
      from catsim.stopping import MinErrorStopper
      from catsim.simulation import Simulator

      fig, axes = plt.subplots(2, 1, figsize=(7, 12))
      s = Simulator(ItemBank.generate_item_bank(100), 10)
      s.simulate(RandomInitializer(), MaxInfoSelector(), NumericalSearchEstimator(), MinErrorStopper(0.4, max_items=20))
      plot.test_progress(simulator=s, index=0, ax=axes[0])
      plot.test_progress(simulator=s, index=0, info=True, var=True, see=True, ax=axes[1])
      plt.tight_layout()
      plt.show()

  Parameters
  ----------
  ax : matplotlib.axes.Axes or None, optional
      Axis to use. If None, a figure with the necessary axis will be created. Default is None.
  title : str or None, optional
      The plot title. Default is None.
  simulator : Simulator or None, optional
      A simulator which has already simulated a series of CATs, containing estimations
      to the examinees' abilities and a list of administered items for each examinee.
      Default is None.
  index : int or None, optional
      The index of the examinee in the simulator whose plot is to be done. Default is None.
  thetas : list[float] or None, optional
      If a Simulator is not passed, then a list of ability estimations can be manually
      passed to the function. Default is None.
  administered_items : numpy.ndarray or None, optional
      If a Simulator is not passed, then a matrix of administered items, represented
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
      If neither simulator nor the required manual parameters are provided, or if thetas
      and administered_items have mismatched lengths.
  """
  if simulator is None and thetas is None and administered_items is None:
    msg = "Not a single plottable object was passed. One of: simulator, thetas, administered_items must be passed."
    raise ValueError(msg)

  if ax is None:
    _, ax = plt.subplots(figsize=figsize)

  if title is not None:
    ax.set_title(title, size=18)

  if simulator is not None and index is not None:
    thetas = simulator.estimations[index]
    administered_items = simulator.items[simulator.administered_items[index]]
    true_theta = simulator.examinees[index]

  assert thetas is not None
  assert administered_items is not None
  assert true_theta is not None

  if thetas is not None and administered_items is not None and len(thetas) - 1 != len(administered_items[:, 1]):
    msg = "Number of estimated thetas and administered items is not the same."
    raise ValueError(msg)

  # len(thetas) - 1 because the first item is made by the initializer
  xs = list(range(len(thetas))) if thetas is not None else list(range(len(administered_items[:, 1])))

  if thetas is not None:
    ax.plot(xs, thetas, label=r"$\hat{\theta}$", marker=marker)
  if administered_items is not None:
    difficulties = administered_items[:, 1]
    ax.plot(xs[1:], difficulties, label="Item difficulty", marker=marker)
  if true_theta is not None:
    ax.hlines(true_theta, 0, len(xs) - 1, label=r"$\theta$", colors="black", linestyles="dashed")
  if thetas is not None and administered_items is not None:
    # calculate and plot test information, var, standard error and reliability
    if info:
      infos = [
        irt.test_info(
          thetas[x],
          administered_items[: x + 1,],
        )
        for x in xs
      ]
      ax.plot(xs, infos, label=r"$I(\theta)$", marker=marker)

    if var:
      varss = [
        irt.var(
          thetas[x],
          administered_items[: x + 1,],
        )
        for x in xs
      ]
      ax.plot(xs, varss, label=r"$Var$", marker=marker)

    if see:
      sees = [
        irt.see(
          thetas[x],
          administered_items[: x + 1,],
        )
        for x in xs
      ]
      ax.plot(xs, sees, label=r"$SEE$", marker=marker)

    if reliability:
      reliabilities = [
        irt.reliability(
          thetas[x],
          administered_items[: x + 1,],
        )
        for x in xs
      ]
      ax.plot(xs, reliabilities, label="Reliability", marker=marker)
  ax.set_xlabel("Items")
  ax.grid()
  ax.legend(loc="best")

  return ax
