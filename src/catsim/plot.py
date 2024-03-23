"""Module with functions for plotting IRT-related results."""

import pathlib as pl
from enum import Enum, auto

import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D

from . import irt
from .simulation import Simulator


class PlotType(Enum):
  """Enum with the available item plots."""

  ICC = auto()
  IIC = auto()
  BOTH = auto()


def item_curve(
  a: float = 1,
  b: float = 0,
  c: float = 0,
  d: float = 1,
  title: str | None = None,
  ptype: PlotType = PlotType.ICC,
  max_info: bool = True,
  filepath: str | None = None,
  show: bool = True,
  figsize: tuple | None = None,
) -> None:
  """Plot 'Item Response Theory'-related item plots.

  .. plot::

      from catsim.cat import generate_item_bank
      from catsim import plot
      item = generate_item_bank(1)[0]
      plot.item_curve(item[0], item[1], item[2], item[3], ptype='icc')
      plot.item_curve(item[0], item[1], item[2], item[3], ptype='iic')
      plot.item_curve(item[0], item[1], item[2], item[3], ptype='both')

  When both curves are plotted in the same figure, the figure has no grid, since each curve has a different scale.

  :param a: item discrimination parameter
  :param b: item difficulty parameter
  :param c: item pseudo-guessing parameter
  :param d: item upper asymptote
  :param title: plot title
  :param ptype: PlotType.ICC for the item characteristic curve, PlotType.IIC for the item information curve.
  :param max_info: whether the point of maximum information should be shown in the plot
  :param filepath: saves the plot in the given path
  :param show: whether the generated plot is to be shown
  """
  thetas = numpy.arange(b - 4, b + 4, 0.1, "double")
  p_thetas = []
  i_thetas = []
  for theta in thetas:
    p_thetas.append(irt.icc(theta, a, b, c, d))
    i_thetas.append(irt.inf(theta, a, b, c, d))

  if ptype in {PlotType.ICC, PlotType.IIC}:
    plt.figure(figsize=figsize)

    if title is not None:
      plt.title(title, size=18)

    plt.annotate(
      "$a = " + format(a) + "$\n$b = " + format(b) + "$\n$c = " + format(c) + "$\n$d = " + format(d) + "$",
      bbox={"facecolor": "white", "alpha": 1},
      xy=(0.75, 0.05),
      xycoords="axes fraction",
    )
    plt.xlabel(r"$\theta$")
    plt.grid()

    if ptype == "icc":
      plt.ylabel(r"$P(\theta)$")
      plt.plot(thetas, p_thetas, label=r"$P(\theta)$")

    elif ptype == "iic":
      plt.ylabel(r"$I(\theta)$")
      plt.plot(thetas, i_thetas, label=r"$I(\theta)$")
      if max_info:
        aux = irt.max_info(a, b, c, d)
        plt.plot(aux, irt.inf(aux, a, b, c, d), "o")

  elif ptype == PlotType.BOTH:
    _, ax1 = plt.subplots(figsize=figsize)

    ax1.set_xlabel(r"$\theta$", size=16)
    ax1.set_ylabel(r"$P(\theta)$", color="b", size=16)
    ax1.plot(thetas, p_thetas, "b-", label=r"$P(\theta)$")
    # Make the y-axis label and tick labels match the line color.
    for tl in ax1.get_yticklabels():
      tl.set_color("b")

    ax2 = ax1.twinx()
    ax2.set_ylabel(r"$I(\theta)$", color="r", size=16)
    ax2.plot(thetas, i_thetas, "r-", label=r"$I(\theta)$")
    for tl in ax2.get_yticklabels():
      tl.set_color("r")
    if max_info:
      aux = irt.max_info(a, b, c, d)
      plt.plot(aux, irt.inf(aux, a, b, c, d), "o")

    if title is not None:
      ax1.set_title(title, size=18)

    ax2.annotate(
      "$a = " + format(a) + "$\n$b = " + format(b) + "$\n$c = " + format(c) + "$\n$d = " + format(d) + "$",
      bbox={"facecolor": "white", "alpha": 1},
      xy=(0.75, 0.05),
      xycoords="axes fraction",
    )

  if filepath is not None:
    filepath = pl.Path(filepath)
    # if os.path.dirname(filepath) is empty, it means the user passed the name
    # of the file instead of a path, e.g. 'plot.pdf' instead '~/Downloads/plot.pdf'
    if len(filepath.parent) > 0:
      filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)

  if show:
    plt.show()


def gen3d_dataset_scatter(
  items: numpy.ndarray,
  title: str | None = None,
  filepath: str | None = None,
  show: bool = True,
  figsize: tuple | None = None,
) -> None:
  """Generate the item matrix tridimensional dataset scatter plot.

  .. plot::

      from catsim.cat import generate_item_bank
      from catsim import plot
      items = generate_item_bank(100)
      plot.gen3d_dataset_scatter(items)

  :param items: the item matrix
  :param title: the scatter plot title
  :param filepath: the path to save the scatter plot
  :param show: whether the generated plot is to be shown
  """
  assert Axes3D
  irt.validate_item_bank(items)

  fig = plt.figure(figsize=figsize)
  ax = fig.add_subplot(111, projection="3d")
  ax.scatter(list(items[:, 0]), list(items[:, 1]), list(items[:, 2]), s=10, c="b")

  if title is not None:
    plt.title(title, size=18)

  ax.set_xlabel("a")
  ax.set_ylabel("b")
  ax.set_zlabel("c")

  if filepath is not None:
    filepath = pl.Path(filepath)
    if len(filepath.parent) > 0:
      filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)

  if show:
    plt.show()


def item_exposure(
  title: str | None = None,
  simulator: Simulator | None = None,
  items: numpy.ndarray | None = None,
  par: str | None = None,
  hist: bool = False,
  filepath: str | None = None,
  show: bool = True,
  figsize: tuple | None = None,
) -> None:
  """Generate a bar chart for the item bank exposure rate.

  The `x` axis represents one of the item parameters, while the `y` axis represents their exposure rates. an examinee's
  test progress.

  .. plot::

      from catsim.cat import generate_item_bank
      from catsim import plot
      from catsim.initialization import RandomInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.estimation import NumericalSearchEstimator
      from catsim.stopping import MaxItemStopper
      from catsim.simulation import Simulator

      s = Simulator(generate_item_bank(100), 10)
      s.simulate(RandomInitializer(), MaxInfoSelector(), NumericalSearchEstimator(), MaxItemStopper(20))
      plot.item_exposure(title='Exposures', simulator=s, hist=True)
      plot.item_exposure(title='Exposures', simulator=s, par='b')

  :param title: the plot title.
  :param simulator: a simulator which has already simulated a series of CATs, containing estimations to the examinees'
                    abilities and a list of administered items for each examinee.
  :param items: an item matrix containing item parameters and their exposure rate in the last column.
  :param par: a string representing one of the item parameters to order the items by and use on the x axis, or `None`
              to use the default order of the item bank. Please note that, if `hist=True`, no sorting will be done.
  :param hist: if True, plots a histogram of item exposures. Otherwise, plots a dotted line chart of the exposures,
               sorted in the x-axis by the parameter chosen in `par`.
  :param filepath: the path to save the plot.
  :param show: whether the generated plot is to be shown.
  """
  if simulator is None and items is None:
    msg = "Not a single plottable object was passed. Either 'simulator' or 'items' must be passed."
    raise ValueError(msg)

  plt.figure(figsize=figsize)

  if title is not None:
    plt.title(title, size=18)

  if simulator is not None:
    items = simulator.items

  assert items is not None

  if items.shape[1] != 5:  # noqa: PLR2004
    msg = "The item matrix is supposed to have 5 columns, the last one representing item exposure rates."
    raise ValueError(msg)

  supported_parameters = {"a", "b", "c", "d"}
  if par is not None and par not in supported_parameters:
    msg = "Unsupported parameter 'par'. Supported parameters are: " + ", ".join(supported_parameters) + "."
    raise ValueError(msg)

  if par == "a":
    parameter = items[:, 0]
    xlabel = "Item discrimination"
  elif par == "b":
    parameter = items[:, 1]
    xlabel = "Item difficulty"
  elif par == "c":
    parameter = items[:, 2]
    xlabel = "Item Guessing"
  elif par == "d":
    parameter = items[:, 3]
    xlabel = "Item upper asymptote"
  else:
    parameter = numpy.array(range(items.shape[0]))
    xlabel = "Items"

  if hist:
    plt.hist(items[:, 4], max(int(items.shape[0] / 10), 3))
    plt.xlabel("Item exposure")
    plt.ylabel("Items")
  else:
    indexes = parameter.argsort()
    plt.plot(items[:, 4][indexes], marker="o")
    plt.xlabel(xlabel)
    plt.ylabel("Item exposure")

  plt.legend(loc="best")

  if filepath is not None:
    filepath = pl.Path(filepath)
    if len(filepath.parent) > 0:
      filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)

  if show:
    plt.show()


def test_progress(
  title: str | None = None,
  simulator: Simulator = None,
  index: int | None = None,
  thetas: list[float] | None = None,
  administered_items: numpy.ndarray = None,
  true_theta: float | None = None,
  info: bool = False,
  var: bool = False,
  see: bool = False,
  reliability: bool = False,
  filepath: str | None = None,
  show: bool = True,
  figsize: tuple | None = None,
) -> None:
  """Generates a plot representing an examinee's test progress.

  Note that, while some functions increase or decrease monotonically, like test information and standard error of
  estimation, the plot calculates these values using the examinee's ability estimated at that given time of the test.
  This means that a test that was tought to be informative at a given point may not be as informative after new
  estimates are done.

  .. plot::

      from catsim.cat import generate_item_bank
      from catsim import plot
      from catsim.initialization import RandomInitializer
      from catsim.selection import MaxInfoSelector
      from catsim.estimation import NumericalSearchEstimator
      from catsim.stopping import MaxItemStopper
      from catsim.simulation import Simulator

      s = Simulator(generate_item_bank(100), 10)
      s.simulate(RandomInitializer(), MaxInfoSelector(), NumericalSearchEstimator(), MaxItemStopper(20))
      plot.test_progress(simulator=s, index=0)
      plot.test_progress(simulator=s, index=0, info=True, var=True, see=True)

  :param title: the plot title.
  :param simulator: a simulator which has already simulated a series of CATs,
                    containing estimations to the examinees' abilities and
                    a list of administered items for each examinee.
  :param index: the index of the examinee in the simulator whose plot is to be done.
  :param thetas: if a :py:class:`Simulator` is not passed, then a list of ability
                 estimations can be manually passed to the function.
  :param administered_items: if a :py:class:`Simulator` is not passed, then a
                             matrix of administered items, represented by their
                             parameters, can be manually passed to the function.
  :param true_theta: the value of the examinee's true ability. If it is passed,
                     it will be shown on the plot, otherwise not.
  :param info: plot test information. It only works if both abilities and
               administered items are passed.
  :param var: plot the estimation variance during the test. It only
              works if both abilities and administered items are passed.
  :param see: plot the standard error of estimation during the test. It only
             works if both abilities and administered items are passed.
  :param reliability: plot the test reliability. It only works if both abilities
                      and administered items are passed.
  :param filepath: the path to save the plot
  :param show: whether the generated plot is to be shown
  """
  if simulator is None and thetas is None and administered_items is None:
    msg = "Not a single plottable object was passed. One of: simulator, thetas, administered_items must be passed."
    raise ValueError(msg)

  plt.figure(figsize=figsize)

  if title is not None:
    plt.title(title, size=18)

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
    plt.plot(xs, thetas, label=r"$\hat{\theta}$")
  if administered_items is not None:
    difficulties = administered_items[:, 1]
    plt.plot(xs[1:], difficulties, label="Item difficulty")
  if true_theta is not None:
    plt.hlines(true_theta, 0, len(xs), label=r"$\theta$")
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
      plt.plot(xs, infos, label=r"$I(\theta)$")

    if var:
      varss = [
        irt.var(
          thetas[x],
          administered_items[: x + 1,],
        )
        for x in xs
      ]
      plt.plot(xs, varss, label=r"$Var$")

    if see:
      sees = [
        irt.see(
          thetas[x],
          administered_items[: x + 1,],
        )
        for x in xs
      ]
      plt.plot(xs, sees, label=r"$SEE$")

    if reliability:
      reliabilities = [
        irt.reliability(
          thetas[x],
          administered_items[: x + 1,],
        )
        for x in xs
      ]
      plt.plot(xs, reliabilities, label="Reliability")
  plt.xlabel("Items")
  plt.grid()
  plt.legend(loc="best")

  if filepath is not None:
    filepath = pl.Path(filepath)
    # if os.path.dirname(filepath) is empty, it means the user passed the name
    # of the file instead of a path, e.g. 'plot.pdf' instead '~/Downloads/plot.pdf'
    if len(filepath.parent) > 0:
      filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)

  if show:
    plt.show()


def param_dist(
  items: numpy.ndarray, filepath: str | None = None, show: bool = True, figsize: tuple | None = None
) -> None:
  """Plot histograms for the item parameters.

  :param items: Item parameter matrix.
  :type items: numpy.ndarray
  :param filepath: Optional filepath to save the plot, defaults to None
  :type filepath: str | None, optional
  :param show: Whether to show the plot after generating it, defaults to True
  :type show: bool, optional
  :param figsize: Optional figure size, defaults to None
  :type figsize: tuple | None, optional
  """
  _, axes = plt.subplots(2, 2, figsize=figsize)
  _ = axes[0, 0].hist(items[:, 0], bins=100)
  _ = axes[0, 1].hist(items[:, 1], bins=100)
  _ = axes[1, 0].hist(items[:, 2], bins=100)
  _ = axes[1, 1].hist(items[:, 3], bins=100)

  if filepath is not None:
    filepath = pl.Path(filepath)
    # if os.path.dirname(filepath) is empty, it means the user passed the name
    # of the file instead of a path, e.g. 'plot.pdf' instead '~/Downloads/plot.pdf'
    if len(filepath.parent) > 0:
      filepath.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)

  if show:
    plt.show()
