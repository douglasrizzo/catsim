"""Module with functions for plotting clustering and IRT-related results."""

import os
import numpy as np
import matplotlib.pyplot as plt
import catsim.cat.irt


def column(matrix, i):
    """Returns columns from a bidimensional Python list (a list of lists)"""
    return [row[i] for row in matrix]


def plot3D(points, clusters, title, path, centers=None):
    """Plots 3D cluster charts

       :param points: a matrix with the 3D locations of the data points
       :type points: numpy.ndarray
       :param clusters: a list with the cluster memberships for each data point
       :type clusters: numpy.ndarray
       :param title: The title for the plot
       :type title: string
       :param centers: a matrix with the positions of the cluster centers,
                       if they exist
       :type centers: numpy.ndarray
    """
    # plota gráficos
    # variáveis utilizadas no plot
    colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = np.hstack([colors] * 20)
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    if title is not None:
        ax.set_title(title)

    ax.scatter(column(points.tolist(), 0), column(points.tolist(), 1),
               column(points.tolist(), 2),
               c=colors[clusters].tolist(),
               s=10)

    if centers is not None:
        center_colors = colors[:len(centers)]
        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                   s=100,
                   c=center_colors)

    ax.set_xlabel('a')
    ax.set_ylabel('b')
    ax.set_zlabel('c')

    if not os.path.exists(path):
        os.makedirs(path)

    plt.savefig(path + title + '.pdf', bbox_inches='tight')


def plotIRT(a=1, b=0, c=0, title=None, ptype='icc', filepath=None):
    """Plots 'Item Response Theory'-related item plots

       :param a: item discrimination parameter
       :type a: float
       :param b: item difficulty parameter
       :type b: float
       :param c: item pseudo-guessing parameter
       :type c: float
       :param title: plot title
       :type title: string
       :param ptype: 'icc' for the item characteristic curve, 'iif' for the
                     item information curve or 'both' for both curve in the
                     same plot
       :type ptype: string
       :param filepath: saves the plot in the given path
       :type filepath: string
    """
    thetas = np.arange(b - 4, b + 4, .1, 'double')
    p_thetas = []
    i_thetas = []
    for theta in thetas:
        p_thetas.append(catsim.cat.irt.tpm(theta, a, b, c))
        i_thetas.append(catsim.cat.irt.inf(theta, a, b, c))

    if ptype in ['icc', 'iif']:
        plt.figure()
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


def gen3DDatasetGraphs():
    dados_graphdir = dissertacao + '/img/3d/'
    if not os.path.exists(dados_graphdir):
        os.makedirs(dados_graphdir)
    datasets = loadDatasets()

    for dataset_name, x, x_scaled in datasets:
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(column(x_scaled.tolist(), 0), column(x_scaled.tolist(), 1),
                   column(x_scaled.tolist(), 2),
                   s=10)
        ax.set_title(dataset_name + ' normalizado')
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_zlabel('c')

        plt.savefig(dados_graphdir + dataset_name + '_scaled.pdf',
                    bbox_inches='tight')

        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(column(x.tolist(), 0), column(x.tolist(), 1),
                   column(x.tolist(), 2),
                   s=10)
        ax.set_title(dataset_name)
        ax.set_xlabel('a')
        ax.set_ylabel('b')
        ax.set_zlabel('c')

        plt.savefig(dados_graphdir + dataset_name + '.pdf',
                    bbox_inches='tight')
