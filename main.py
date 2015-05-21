# -*- coding: utf-8 -*-

import os
import time
from datetime import timedelta
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from numpy import random

from sklearn import cluster as skcluster, preprocessing
from sklearn.metrics import silhouette_score

import catsim.cat.simulate
import catsim.cluster.stats
import catsim.cluster.distances
import catsim.cat.irt
import catsim.misc.plot
import catsim.misc.results
import catsim.misc.stats


def column(matrix, i):
    """retorna colunas de uma lista bidimensional do Python"""
    return [row[i] for row in matrix]


def medias_tpm(x, c):
    """médias dos parâmetros a, b e c para cada cluster"""
    medias = np.zeros((np.max(c) + 1, 3))

    for i in np.arange(0, np.size(c)):
        if c[i] == -1:
            continue
        medias[c[i], 0] += x[i, 0]
        medias[c[i], 1] += x[i, 1]
        medias[c[i], 2] += x[i, 2]

    ocorrencias = np.bincount(np.delete(c, np.where(c == -1)))

    for counter, i in enumerate(ocorrencias):
        medias[counter, 0] = medias[counter, 0] / i
        medias[counter, 1] = medias[counter, 1] / i
        medias[counter, 2] = medias[counter, 2] / i

    return medias


def loadDatasets():
    """carrega datasets utilizados no experimento"""
    mt = np.genfromtxt(dissertacao_datadir + 'mt_params.csv', dtype='double',
                       delimiter=',', skip_header=True)
    ch = np.genfromtxt(dissertacao_datadir + 'ch_params.csv', dtype='double',
                       delimiter=',', skip_header=True)
    cn = np.genfromtxt(dissertacao_datadir + 'cn_params.csv', dtype='double',
                       delimiter=',', skip_header=True)
    lc = np.genfromtxt(dissertacao_datadir + 'lc_params.csv', dtype='double',
                       delimiter=',', skip_header=True)
    enem = np.concatenate((ch, cn, lc, mt))

    # base sintética gerada usando as distribuições utilizadas pelo barrada
    # nos artigos de 2008 e 2009 dele
    if not os.path.exists(dissertacao_datadir + 'synth_params.csv'):
        np.savetxt(dissertacao_datadir + 'synth_params.csv',
                   np.array([random.normal(1.2, .25, 500),
                             random.normal(0, 1, 500),
                             random.normal(.25, .02, 500)],
                            dtype='double').T,
                   delimiter=',')

    sintetico = np.genfromtxt(dissertacao_datadir + 'synth_params.csv',
                              dtype='double',
                              delimiter=',')

    return [
            # ['CH', ch],
            # ['CH normalizado', preprocessing.scale(ch)],
            # ['CN', cn],
            # ['CN normalizado', preprocessing.scale(cn)],
            # ['LC', lc],
            # ['LC normalizado', preprocessing.scale(lc)],
            # ['MT', mt],
            # ['MT normalizado', preprocessing.scale(mt)],
            ['Enem', enem],
            ['Enem normalizado', preprocessing.scale(enem)],
            ['Sintético', sintetico],
            ['Sintético normalizado', preprocessing.scale(sintetico)]
           ]


def testDistances(dataset):
    name = dataset[0].lower()

    ds = []
    d_names = ['Manhattan', 'Euclidean', 'Minskowski - p = 3',
               'Minskowski - p = 4', 'Minskowski - p = 5', 'Chebyshev']

    for p in np.arange(4):
        ds.append(distances.pnorm(dataset[1], p=p + 1))

    ds.append(distances.pnorm(dataset[1], p=float('inf')))

    # for i in distances:
    #     np.savetxt(dissertacao_datadir + name + '_' + i[0] + '.csv', i[1],
    #                delimiter=',')


def gen3DDatasetGraphs():
    """gera os gráficos 3D dos parâmetros da TRI"""
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


def sklearnTests(plots, videos=False):
    """Agrupa os itens em clusters"""

    datasets = loadDatasets()

    for dataset_name, x in datasets:
        # distâncias euclideanas

        algorithms = []

        # cria estimadores de clustering
        for n_clusters in np.arange(2, (np.size(x, 0) / 2) - 1):
            n_clusters = int(n_clusters)
            # k-Means, ligação média, ligação completa, Ward, espectral
            algorithms.extend(
                [('K-Means', 'K-Means (k = ' + format(n_clusters) + ')',
                  n_clusters, skcluster.KMeans(n_clusters=n_clusters,
                                               init='random')),
                 ('Hier. ligação média',
                  'Hier. ligação média (k = ' + format(n_clusters) + ')',
                  n_clusters,
                  skcluster.AgglomerativeClustering(linkage='average',
                                                    affinity='euclidean',
                                                    n_clusters=n_clusters)),
                 ('Hier. ligação completa',
                  'Hier. ligação completa (k = ' + format(n_clusters) + ')',
                  n_clusters,
                  skcluster.AgglomerativeClustering(linkage='complete',
                                                    affinity='euclidean',
                                                    n_clusters=n_clusters)),
                 ('Hier. Ward', 'Hier. Ward (k = ' + format(n_clusters) + ')',
                  n_clusters,
                  skcluster.AgglomerativeClustering(linkage='ward',
                                                    n_clusters=n_clusters)),
                 ('Espectral', 'Espectral (k = ' + format(n_clusters) + ')',
                  n_clusters,
                  skcluster.SpectralClustering(n_clusters=n_clusters,
                                               eigen_solver='arpack',
                                               affinity="nearest_neighbors"))])

        min_samples = 4
        for eps in np.arange(.1, 1, .02):
            algorithms.extend([('DBSCAN', 'DBSCAN (eps = ' + format(eps) + ')',
                                eps, skcluster.DBSCAN(
                                  eps=eps,
                                  min_samples=min_samples))])

        algorithms.extend([('Aff. Propagation', 'Affinity Propagation', eps,
                            skcluster.AffinityPropagation(
                              damping=.9,
                              preference=-200))])

        t0 = time.time()
        for counter, algorithm_package in enumerate(algorithms):
            [algorithm_id, algorithm_name,
             algorithm_variable, algorithm] = algorithm_package

            # roda algoritmos
            t1 = time.time()
            algorithm.fit(x)
            t2 = time.time()

            print(
                format(timedelta(seconds=(t1 - t0) / (counter + 1) * (np.size(
                    algorithms, 0) - counter))) + '\t' + dataset_name + '\t' +
                format(round(100 / np.size(algorithms, 0) * counter, 2)) +
                '%\t' + algorithm_name + '            \r', end='\r')

            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(x)

            # calcula menor e maior clusters
            if len(set(y_pred)) > 1:
                cluster_bins = np.bincount(np.delete(y_pred, np.where(y_pred ==
                                                                      -1)))
                min_c = min(cluster_bins)
                max_c = max(cluster_bins)

                var = cluster.stats.mean_variance(y_pred, distances)
                dun = cluster.stats.dunn(y_pred, distances)
                silhouette = silhouette_score(x, y_pred)

                cluster_mediasdir = outdir + 'medias/'
                if not os.path.exists(cluster_mediasdir):
                    os.makedirs(cluster_mediasdir)

                np.savetxt(cluster_mediasdir + algorithm_name + '_' +
                           dataset_name + '_medias.csv', medias_tpm(x, y_pred),
                           delimiter=',')

                results.saveResults([algorithm_id, dataset_name,
                                     algorithm_variable,
                                     np.size(x, 0), len(set(y_pred)),
                                     t2 - t1, min_c, max_c, var, dun,
                                     silhouette, str(y_pred.tolist()).strip(
                                      '[]').replace(',', '')])

                if plots:
                    # plota gráficos
                    # variáveis utilizadas no plot
                    colors = np.array(
                        [x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
                    colors = np.hstack([colors] * 20)
                    fig = plt.figure(figsize=(8, 6))
                    plt.title(dataset_name + ' - ' + algorithm_name, size=18)
                    ax = fig.add_subplot(111, projection='3d')
                    ax.scatter(column(x_scaled.tolist(), 0),
                               column(x_scaled.tolist(), 1),
                               column(x_scaled.tolist(), 2),
                               c=colors[y_pred].tolist(),
                               s=10)

                    if hasattr(algorithm, 'cluster_centers_'):
                        centers = algorithm.cluster_centers_
                        center_colors = colors[:len(centers)]
                        ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2],
                                   s=100,
                                   c=center_colors)

                    ax.set_title(algorithm_name + ' - ' + dataset_name)
                    ax.set_xlabel('a')
                    ax.set_ylabel('b')
                    ax.set_zlabel('c')

                    cluster_graphdir = outdir + 'clusters/'

                    if not os.path.exists(cluster_graphdir):
                        os.makedirs(cluster_graphdir)

                    plt.savefig(cluster_graphdir + algorithm_name + ' - ' +
                                dataset_name + '.pdf',
                                bbox_inches='tight')

                if videos:
                    for ii in range(0, 360, 1):
                        ax.view_init(elev=10., azim=ii)
                        print('Gerando imagens para renderizar vídeo: ' +
                              format(round(100 / 360 * ii, 2)) + '%')
                        plt.savefig(cluster_graphdir + format(ii) + '.jpg',
                                    dpi=500,
                                    bbox_inches='tight')

                    if os.path.isfile(cluster_graphdir + 'video.avi'):
                        os.remove(cluster_graphdir + 'video.avi')

                    call('ffmpeg -i \'' + cluster_graphdir +
                         '%d.jpg\' -vcodec mpeg4 \'' + cluster_graphdir +
                         'video.mp4\'',
                         shell=True)
                    filelist = [f for f in os.listdir(cluster_graphdir)
                                if f.endswith(".jpg")]
                    for f in filelist:
                        os.remove(cluster_graphdir + f)


if __name__ == '__main__':
    dissertacao = '/home/douglas/repos/dissertacao/'
    dissertacao_imgdir = dissertacao + 'img/'
    dissertacao_datadir = dissertacao + 'dados/'
    outdir = '/home/douglas/Desktop/out/'

    sklearnTests(False)
