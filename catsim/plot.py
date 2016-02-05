"""Module with functions for plotting IRT-related results."""

import os
from catsim import irt
import numpy as np
import matplotlib.pyplot as plt


def __column(matrix, i):
    """Returns columns from a bidimensional Python list (a list of lists)"""
    return [row[i] for row in matrix]


def plot_irt(a=1, b=0, c=0, title=None, ptype='icc', filepath=None):
    """Plots 'Item Response Theory'-related item plots

    :param a: item discrimination parameter
    :type a: float
    :param b: item difficulty parameter
    :type b: float
    :param c: item pseudo-guessing parameter
    :type c: float
    :param title: plot title
    :type title: string
    :param ptype: 'icc' for the item characteristic curve, 'iif' for the item
                  information curve or 'both' for both curves in the same plot
    :type ptype: string
    :param filepath: saves the plot in the given path
    :type filepath: string
    """
    available_types = ['icc', 'iif', 'both']

    if ptype not in available_types:
        raise ValueError('\'{0}\' not in available plot types: {1}'.format(ptype, available_types))

    thetas = np.arange(b - 4, b + 4, .1, 'double')
    p_thetas = []
    i_thetas = []
    for theta in thetas:
        p_thetas.append(irt.tpm(theta, a, b, c))
        i_thetas.append(irt.inf(theta, a, b, c))

    if ptype in ['icc', 'iif']:
        plt.figure()
        if title is not None:
            plt.title(title, size=18)
        plt.annotate('$a = ' + format(a) + '$\n$b = ' + format(
            b) + '$\n$c = ' + format(c) + '$',
            bbox=dict(facecolor='white',
                      alpha=1),
            xy=(.75, .05),
            xycoords='axes fraction')
        plt.xlabel(r'$\theta$')
        plt.grid()
        plt.legend(loc='best')

        if ptype == 'icc':
            plt.ylabel(r'$P(\theta)$')
            plt.plot(thetas, p_thetas)

        elif ptype == 'iif':
            plt.ylabel(r'$I(\theta)$')
            plt.plot(thetas, i_thetas)

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

        ax1.set_title(title, size=18)

        ax2.annotate('$a = ' + format(a) + '$\n$b = ' + format(
            b) + '$\n$c = ' + format(c) + '$',
            bbox=dict(facecolor='white',
                      alpha=1),
            xy=(.75, .05),
            xycoords='axes fraction')
        ax2.legend(loc='best', framealpha=0)

    if filepath is not None:
        plt.savefig(filepath)

    plt.close()


def gen3D_dataset_scatter(title, items, path):
    """Generate the item matrix tridimensional dataset scatter plot

    :param title: the scatter plot title, which will also be used as the file name
    :type title: string
    :param items: the item matrix
    :type items: numpy.ndarray
    :param path: the path to save the scatter plot
    :type path: string
    """
    irt.validate_item_bank(items)

    if not os.path.exists(path):
        os.makedirs(path)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(__column(items.tolist(), 0), __column(items.tolist(), 1),
               __column(items.tolist(), 2),
               s=10)
    ax.set_title(title)
    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')

    plt.savefig(path + title + '.pdf',
                bbox_inches='tight')
