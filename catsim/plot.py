"""Module with functions for plotting IRT-related results."""

import os
import numpy
from catsim import irt
import matplotlib.pyplot as plt
from catsim.simulation import Simulator
from mpl_toolkits.mplot3d import Axes3D


def __column(matrix, i):
    """Returns columns from a bidimensional Python list (a list of lists)"""
    return [row[i] for row in matrix]


def item_curve(
    a: float=1,
    b: float=0,
    c: float=0,
    d: float=1,
    title: str=None,
    ptype: str='icc',
    max_info=True,
    filepath: str=None,
    show: bool=True
):
    """Plots 'Item Response Theory'-related item plots

    .. plot::

        from catsim.cat import generate_item_bank
        from catsim import plot
        item = generate_item_bank(1)[0]
        plot.item_curve(item[0], item[1], item[2], ptype='icc')
        plot.item_curve(item[0], item[1], item[2], ptype='iic')
        plot.item_curve(item[0], item[1], item[2], ptype='both')

    When both curves are plotted in the same figure, the figure has no grid,
    since each curve has a different scale.

    :param a: item discrimination parameter
    :param b: item difficulty parameter
    :param c: item pseudo-guessing parameter
    :param title: plot title
    :param ptype: 'icc' for the item characteristic curve, 'iic' for the item
                  information curve or 'both' for both curves in the same plot
    :param filepath: saves the plot in the given path
    :param show: whether the generated plot is to be shown
    """
    available_types = ['icc', 'iic', 'both']

    if ptype not in available_types:
        raise ValueError('\'{0}\' not in available plot types: {1}'.format(ptype, available_types))

    thetas = numpy.arange(b - 4, b + 4, .1, 'double')
    p_thetas = []
    i_thetas = []
    for theta in thetas:
        p_thetas.append(irt.icc(theta, a, b, c, d))
        i_thetas.append(irt.inf(theta, a, b, c, d))

    if ptype in ['icc', 'iic']:
        plt.figure()

        if title is not None:
            plt.title(title, size=18)

        plt.annotate(
            '$a = ' + format(a) + '$\n$b = ' + format(
                b
            ) + '$\n$c = ' + format(c) + '$',
            bbox=dict(
                facecolor='white',
                alpha=1
            ),
            xy=(.75, .05),
            xycoords='axes fraction'
        )
        plt.xlabel(r'$\theta$')
        plt.grid()
        plt.legend(loc='best')

        if ptype == 'icc':
            plt.ylabel(r'$P(\theta)$')
            plt.plot(thetas, p_thetas)

        elif ptype == 'iic':
            plt.ylabel(r'$I(\theta)$')
            plt.plot(thetas, i_thetas)
            if max_info:
                aux = irt.max_info(a, b, c, d)
                plt.plot(aux, irt.inf(aux, a, b, c, d), 'o')

    elif ptype == 'both':
        _, ax1 = plt.subplots()

        ax1.plot(thetas, p_thetas, 'b-')
        ax1.set_xlabel(r'$\theta$', size=16)
        # Make the y-axis label and tick labels match the line color.
        ax1.set_ylabel(r'$P(\theta)$', color='b', size=16)
        for tl in ax1.get_yticklabels():
            tl.set_color('b')

        ax2 = ax1.twinx()
        ax2.plot(thetas, i_thetas, 'r-')
        ax2.set_ylabel(r'$I(\theta)$', color='r', size=16)
        for tl in ax2.get_yticklabels():
            tl.set_color('r')
        if max_info:
            aux = irt.max_info(a, b, c, d)
            plt.plot(aux, irt.inf(aux, a, b, c, d), 'o')

        if title is not None:
            ax1.set_title(title, size=18)

        ax2.annotate(
            '$a = ' + format(a) + '$\n$b = ' + format(
                b
            ) + '$\n$c = ' + format(c) + '$\n$d = ' + format(d) + '$',
            bbox=dict(
                facecolor='white',
                alpha=1
            ),
            xy=(.75, .05),
            xycoords='axes fraction'
        )
        ax2.legend(loc='best', framealpha=0)

    if filepath is not None:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        plt.savefig(filepath, bbox_inches='tight')

    if show:
        plt.show()


def gen3D_dataset_scatter(
    items: numpy.ndarray,
    title: str=None,
    filepath: str=None,
    show: bool=True
):
    """Generate the item matrix tridimensional dataset scatter plot

    .. plot::

        from catsim.cat import generate_item_bank
        from catsim import plot
        items = generate_item_bank(100)
        plot.gen3D_dataset_scatter(items)

    :param items: the item matrix
    :param title: the scatter plot title
    :param filepath: the path to save the scatter plot
    :param show: whether the generated plot is to be shown
    """
    irt.validate_item_bank(items)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(list(items[:, 0]), list(items[:, 1]), list(items[:, 2]), s=10, c='b')

    if title is not None:
        plt.title(title, size=18)

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')

    if filepath is not None:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        plt.savefig(filepath, bbox_inches='tight')

    if show:
        plt.show()


def test_progress(
    title: str=None,
    simulator: Simulator=None,
    index: int=None,
    thetas: list=None,
    administered_items: numpy.ndarray=None,
    true_theta: float=None,
    info: bool=False,
    var: bool=False,
    see: bool=False,
    reliability: bool=False,
    filepath: str=None,
    show: bool=True
):
    """Generates a plot representing an examinee's test progress

    .. plot::

        from catsim.cat import generate_item_bank
        from catsim import plot
        from catsim.initialization import RandomInitializer
        from catsim.selection import MaxInfoSelector
        from catsim.estimation import HillClimbingEstimator
        from catsim.stopping import MaxItemStopper
        from catsim.simulation import Simulator

        initializer = RandomInitializer()
        selector = MaxInfoSelector()
        estimator = HillClimbingEstimator()
        stopper = MaxItemStopper(20)
        s = Simulator(generate_item_bank(100), 10)
        s.simulate(initializer, selector, estimator, stopper)
        plot.test_progress(simulator=s, index=0)
        plot.test_progress(simulator=s, index=0, true_theta=s.examinees[0], info=True, var=True, see=True)

    :param title: the plot title.
    :param simulator: a simulator which has already simulated a series of CATs,
                      containing estimations to the examinees' proficiencies and
                      a list of administered items for each examinee.
    :param index: the index of the examinee in the simulator whose plot is to be done.
    :param thetas: if a :py:class:`Simulator` is not passed, then a list of proficiency
                   estimations can be manually passed to the function.
    :param administered_items: if a :py:class:`Simulator` is not passed, then a
                               matrix of administered items, represented by their
                               parameters, can be manually passed to the function.
    :param true_theta: the value of the examinee's true proficiency. If it is passed,
                       it will be shown on the plot, otherwise not.
    :param info: plot test information. It only works if both proficiencies and
                 administered items are passed.
    :param var: plot the estimation variance during the test. It only
                works if both proficiencies and administered items are passed.
    :param see: plot the standard error of estimation during the test. It only
               works if both proficiencies and administered items are passed.
    :param reliability: plot the test reliability. It only works if both proficiencies
                        and administered items are passed.
    :param filepath: the path to save the plot
    :param show: whether the generated plot is to be shown
    """
    if simulator is None and thetas is None and administered_items is None:
        raise ValueError('Not a single plottable object was passed.')

    plt.figure()

    if title is not None:
        plt.title(title, size=18)

    if simulator is not None and index is not None:
        thetas = simulator.all_estimations[index]
        administered_items = simulator.items[simulator.administered_items[index]]
        true_theta = simulator.examinees[index]

    if thetas is not None and administered_items is not None and len(thetas) != len(
        administered_items[:, 1]
    ):
        raise ValueError(
            'Number of estimations and administered items is not the same. They should be.'
        )

    xs = range(len(thetas)) if thetas is not None else range(len(administered_items[:, 1]))

    if thetas is not None:
        plt.plot(xs, thetas, label=r'$\hat{\theta}$')
    if administered_items is not None:
        difficulties = administered_items[:, 1]
        plt.plot(xs, difficulties, label='Item difficulty')
    if true_theta is not None:
        plt.hlines(true_theta, 0, len(xs), label=r'$\theta$')
    if thetas is not None and administered_items is not None:

        # calculate and plot test information, var, standard error and reliability
        if info:
            sees = [irt.test_info(thetas[x], administered_items[:x + 1, ]) for x in xs]
            plt.plot(xs, sees, label=r'$I(\theta)$')

        if var:
            varss = [irt.var(thetas[x], administered_items[:x + 1, ]) for x in xs]
            plt.plot(xs, varss, label=r'$Var$')

        if see:
            sees = [irt.see(thetas[x], administered_items[:x + 1, ]) for x in xs]
            plt.plot(xs, sees, label=r'$SEE$')

        if reliability:
            reliabilities = [irt.reliability(thetas[x], administered_items[:x + 1, ]) for x in xs]
            plt.plot(xs, reliabilities, label='Reliability')

    plt.xlabel('Items')
    plt.grid()
    plt.legend(loc='best')

    if filepath is not None:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        plt.savefig(filepath, bbox_inches='tight')

    if show:
        plt.show()
