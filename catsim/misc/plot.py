"""Module with functions for plotting IRT-related results."""

import os

import matplotlib.pyplot as plt


def column(matrix, i):
    """Returns columns from a bidimensional Python list (a list of lists)"""
    return [row[i] for row in matrix]


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
