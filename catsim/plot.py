"""Module with functions for plotting IRT-related results."""

import os
from catsim import irt
import numpy
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
    title: str=None,
    ptype: str='icc',
    filepath: str=None
):
    """Plots 'Item Response Theory'-related item plots

    :param a: item discrimination parameter
    :param b: item difficulty parameter
    :param c: item pseudo-guessing parameter
    :param title: plot title
    :param ptype: 'icc' for the item characteristic curve, 'iif' for the item
                  information curve or 'both' for both curves in the same plot
    :param filepath: saves the plot in the given path
    """
    available_types = ['icc', 'iif', 'both']

    if not os.path.exists(os.path.dirname(filepath)):
        os.makedirs(os.path.dirname(filepath))

    if ptype not in available_types:
        raise ValueError('\'{0}\' not in available plot types: {1}'.format(ptype, available_types))

    thetas = numpy.arange(b - 4, b + 4, .1, 'double')
    p_thetas = []
    i_thetas = []
    for theta in thetas:
        p_thetas.append(irt.tpm(theta, a, b, c))
        i_thetas.append(irt.inf(theta, a, b, c))

    if ptype in ['icc', 'iif']:
        plt.figure()
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
            if title is not None:
                title = 'Item characteristic curve'

        elif ptype == 'iif':
            plt.ylabel(r'$I(\theta)$')
            plt.plot(thetas, i_thetas)
            if title is not None:
                title = 'Item information curve'

        plt.title(title, size=18)

    elif ptype == 'both':
        fig, ax1 = plt.subplots()

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

        if title is not None:
            ax1.set_title(title, size=18)

        ax2.annotate(
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
        ax2.legend(loc='best', framealpha=0)

    if filepath is not None:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()
    plt.close()


def gen3D_dataset_scatter(items: numpy.ndarray, title: str='Item parameters', filepath: str=None):
    """Generate the item matrix tridimensional dataset scatter plot

    :param items: the item matrix
    :param title: the scatter plot title
    :param filepath: the path to save the scatter plot
    """
    irt.validate_item_bank(items)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(
        __column(items.tolist(), 0),
        __column(items.tolist(), 1),
        __column(items.tolist(), 2),
        s=10
    )

    plt.title(title, size=18)

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')

    if filepath is not None:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()
    plt.close()


def test_progress(
    title: str=None,
    simulator: Simulator=None,
    index: int=None,
    thetas: list=None,
    administered_items: numpy.ndarray=None,
    true_theta: float=None,
    filepath: str=None
):
    """Generates a plot representing an examinee's test progress

    :param title: the plot title
    :param simulator: a simulator which has already simulated a series of CATs, containing estimations to the examinees' proficiencies and a list of administered items for each examinee
    :param index: the index of the examinee in the simulator whose plot is to be done
    :param thetas: if a :py:class:`Simulator` is not passed, then a list of proficiency estimations can be manually passed to the function
    :param administered_items: if a :py:class:`Simulator` is not passed, then a matrix of administered items, represented by their parameters, can be manually passed to the function
    :param true_theta: the value of the examinee's true proficiency. If it is passed, it will be shown on the plot, otherwise not
    :param filepath: the path to save the plot
    """
    if simulator is None and thetas is None and administered_items is None:
        raise ValueError('Not a single plottable object was passed.')

    plt.figure()
    plt.title(title, size=18)
    if simulator is not None and index is not None:
        thetas = simulator.estimations[index]
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
        plt.plot(xs, thetas, label=r'$\theta$')
    if administered_items is not None:
        difficulties = administered_items[:, 1]
        plt.plot(xs, difficulties, label='Item difficulty')
    if true_theta is not None:
        plt.hlines(true_theta, 0, len(xs), label='True theta')

    plt.xlabel('Items')
    plt.grid()
    plt.legend(loc='best')

    if filepath is not None:
        if not os.path.exists(os.path.dirname(filepath)):
            os.makedirs(os.path.dirname(filepath))
        plt.savefig(filepath, bbox_inches='tight')

    plt.show()
    plt.close()
