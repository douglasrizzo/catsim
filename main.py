# -*- coding: utf-8 -*-

import os
import sys
import time
from datetime import timedelta
from subprocess import call

import matplotlib.pyplot as plt
import numpy as np
import pandas
from mpl_toolkits.mplot3d import Axes3D
from numpy import random
from pandas import DataFrame

from sklearn import cluster as skcluster, preprocessing
from sklearn.metrics import silhouette_score

import cat
import cluster
import distances
import irt
import plot
import stats


def column(matrix, i):
    """retorna colunas de uma lista bidimensional do Python"""
    return [row[i] for row in matrix]


def loadResults():
    """
    Caso o arquivo cluster_results.csv já existe, ele pode ser carregado usando
    esta função
    """
    if not os.path.exists:
        raise FileNotFoundError('Arquivo com resultados não existe.')

    df = pandas.read_csv(dissertacao + 'dados/cluster_results.csv',
                         header=0,
                         encoding='latin_1')
    df[['t (segundos)', 'Dunn', 'Silhueta', 'Nº itens', 'Variável',
        'Nº clusters', 'Menor cluster', 'Maior cluster'
        ]] = df[['t (segundos)', 'Dunn', 'Silhueta', 'Nº itens',
                 'Variável', 'Nº clusters', 'Menor cluster',
                 'Maior cluster']].astype(float)
    df['Sem classificação'] = df['Classificações'].apply(lambda x:
                                                         x.count('-1'))
    df['Classificações'] = df['Classificações'].apply(lambda x:
                                                      np.array(x.split(' '),
                                                               dtype='int'))
    df['pct. sem classificação'] = df[['Sem classificação', 'Classificações'
                                       ]].apply(lambda x: 100 / np.size(x[1]) *
                                                x[0],
                                                axis=1)
    return df


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
    caminho = '/home/douglas/Desktop/dissertacao/dados/'
    mt = np.genfromtxt(caminho + 'mt_params.csv',
                       dtype='double',
                       delimiter=',',
                       skip_header=True)
    ch = np.genfromtxt(caminho + 'ch_params.csv',
                       dtype='double',
                       delimiter=',',
                       skip_header=True)
    cn = np.genfromtxt(caminho + 'cn_params.csv',
                       dtype='double',
                       delimiter=',',
                       skip_header=True)
    lc = np.genfromtxt(caminho + 'lc_params.csv',
                       dtype='double',
                       delimiter=',',
                       skip_header=True)
    enem = np.concatenate((ch, cn, lc, mt))

    # base sintética gerada usando as distribuições utilizadas pelo barrada
    # nos artigos de 2008 e 2009 dele
    if not os.path.exists(caminho + 'synth_params.csv'):
        np.savetxt(caminho + 'synth_params.csv',
                   np.array([random.normal(1.2, .25, 500),
                             random.normal(0, 1, 500),
                             random.normal(.25, .02, 500)],
                            dtype='double').T,
                   delimiter=',')

    sintetico = np.genfromtxt(caminho + 'synth_params.csv',
                              dtype='double',
                              delimiter=',')

    return [['CH', ch],
            ['CN', cn],
            ['LC', lc],
            ['MT', mt],
            ['Enem', enem],
            ['Sintético', sintetico],
            ['CH normalizado', preprocessing.scale(ch)],
            ['CN normalizado', preprocessing.scale(cn)],
            ['LC normalizado', preprocessing.scale(lc)],
            ['MT normalizado', preprocessing.scale(mt)],
            ['Enem normalizado', preprocessing.scale(enem)],
            ['Sintético normalizado', preprocessing.scale(sintetico)]]


def loadDistances(dataset):
    name = dataset[0].lower()

    distances = []
    d_names = ['Manhattan', 'Euclidean', 'Minskowski - p = 3',
               'Minskowski - p = 4', 'Minskowski - p = 5']

    for p in np.arange(4):
        distances[0, p] = d_names[p]
        distances[1, p] = distances.pnorm(dataset[1], p)

    distances[0, 6] = 'Chebyshev'
    distances[1, 6] = distances.pnorm(dataset[1], float('inf'))

    for i in distances:
        np.savetxt(dissertacao_datadir + name + '_' + i[0] + '.csv', i[1],
                   delimiter=',')


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


def genClusters(plots, videos=False):
    """Agrupa os itens em clusters"""

    datasets = loadDatasets()

    for dataset in datasets:
        distances = loadDistances()

    algorithms = loadAlgorithms()

    individuais = np.array(
        ['Algoritmo', 'Dataset', 'Variável', 'Nº itens', 'Nº clusters',
         't (segundos)', 'Menor cluster', 'Maior cluster', 'Variância', 'Dunn',
         'Silhueta', 'Classificações'])

    for dataset_name, x_scaled, x in datasets:
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
                                eps, cluster.DBSCAN(eps=eps,
                                                    min_samples=min_samples))])

        algorithms.extend([('Aff. Propagation', 'Affinity Propagation', eps,
                            cluster.AffinityPropagation(damping=.9,
                                                        preference=-200))])

        t0 = time.time()
        for counter, algorithm_package in enumerate(algorithms):
            algorithm_id, algorithm_name, algorithm_variable, algorithm = algorithm_package

            # roda algoritmos
            t1 = time.time()
            algorithm.fit(x_scaled)
            t2 = time.time()

            print(
                format(timedelta(seconds=(t1 - t0) / (counter + 1) * (np.size(
                    algorithms, 0) - counter))) + '\t' + dataset_name + '\t' +
                format(round(100 / np.size(algorithms, 0) * counter,
                             2)) + '%\t' + algorithm_name + '            \r',
                end='\r')

            if hasattr(algorithm, 'labels_'):
                y_pred = algorithm.labels_.astype(np.int)
            else:
                y_pred = algorithm.predict(x_scaled)

            # calcula menor e maior clusters
            if len(set(y_pred)) > 1:
                cluster_bins = np.bincount(np.delete(y_pred, np.where(y_pred ==
                                                                      -1)))
                min_c = min(cluster_bins)
                max_c = max(cluster_bins)

                var = cluster.stats.mean_variance(y_pred, distances)
                dun = cluster.stats.dunn(y_pred, distances)
                silhouette = silhouette_score(x_scaled, y_pred)

                cluster_mediasdir = outdir + 'medias/'
                if not os.path.exists(cluster_mediasdir):
                    os.makedirs(cluster_mediasdir)

                np.savetxt(cluster_mediasdir + algorithm_name + '_' +
                           dataset_name + '_medias.csv', medias_tpm(x, y_pred),
                           delimiter=',')

                individuais = np.vstack(
                    (individuais,
                     [algorithm_id, dataset_name, algorithm_variable,
                      np.size(x, 0), len(set(y_pred)), t2 - t1, min_c, max_c,
                      var, dun, silhouette, str(y_pred.tolist()).strip(
                          '[]').replace(',', '')]))

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

    np.savetxt(dissertacao + 'dados/cluster_results.csv', individuais,
               delimiter=',',
               fmt='%s')

    df = loadResults()

    df.groupby('Algoritmo')['Menor cluster', 'Maior cluster', 't (segundos)',
                            'Variância', 'Dunn', 'Silhueta'].mean().to_csv(
                                dissertacao_datadir + 'alg_means.csv')
    df.groupby('Dataset')['Menor cluster', 'Maior cluster', 't (segundos)',
                          'Variância', 'Dunn', 'Silhueta'].mean().to_csv(
                              dissertacao_datadir + 'dataset_means.csv')
    df.groupby('Nº clusters')['Menor cluster', 'Maior cluster', 't (segundos)',
                              'Variância', 'Dunn', 'Silhueta'].mean().to_csv(
                                  dissertacao_datadir + 'nclusters_means.csv')

    ax = df.groupby(
        'Nº clusters')['Variância', 'Dunn', 'Silhueta'].mean().plot(
            title='Índices de validação de clusters / nº clusters',
            legend='best',
            figsize=(8, 6))
    ax.set_ylabel('Índices')
    ax.get_figure().savefig(dissertacao_imgdir + 'validity_by_nclusters.pdf')

    df_enem = df[df['Dataset'] == 'Enem'][df['Algoritmo'] !=
                                          'Aff. Propagation'][df['Algoritmo']
                                                              != 'DBSCAN']
    df_sintetico = df[df['Dataset'] ==
                      'Sintético'][df['Algoritmo'] !=
                                   'Aff. Propagation'][df['Algoritmo'] !=
                                                       'DBSCAN']

    ax = pandas.pivot_table(
        df_enem,
        values='Dunn',
        columns='Algoritmo',
        index='Nº clusters').plot(
            figsize=(8, 6),
            grid=True,
            title='Média Dunn / Algoritmo na base \'Enem\'')
    ax.set_ylabel('Dunn')
    ax.get_figure().savefig(dissertacao_imgdir + 'dunn_by_algorithm_enem.pdf')

    ax = pandas.pivot_table(
        df_enem,
        values='Silhueta',
        columns='Algoritmo',
        index='Nº clusters').plot(
            figsize=(8, 6),
            grid=True,
            title='Média silhueta / Algoritmo na base \'Enem\'')
    ax.set_ylabel('Silhueta')
    ax.get_figure().savefig(
        dissertacao_imgdir + 'silhouette_by_algorithm_enem.pdf')

    ax = pandas.pivot_table(
        df_enem,
        values='Menor cluster',
        columns='Algoritmo',
        index='Nº clusters').plot(
            figsize=(8, 6),
            grid=True,
            title='Itens no menor cluster / Algoritmo na base \'Enem\'')
    ax.set_ylabel('Itens no menor cluster')
    ax.get_figure().savefig(
        dissertacao_imgdir + 'smallestcluster_by_algorithm_enem.pdf')

    ax = pandas.pivot_table(df_sintetico, values='Dunn',
                            columns='Algoritmo', index='Nº clusters').plot(
        figsize=(8, 6),
        grid=True,
        title='Média Dunn /' +
        ' Algoritmo na base \'Sintética\'')
    ax.set_ylabel('Dunn')
    ax.get_figure().savefig(dissertacao_imgdir +
                            'dunn_by_algorithm_sintetico.pdf')

    ax = pandas.pivot_table(df_sintetico, values='Silhueta',
                            columns='Algoritmo', index='Nº clusters').plot(
        figsize=(8, 6), grid=True, title='Média silhueta' +
        '/ Algoritmo na base \'Sintética\'')
    ax.set_ylabel('Silhueta')
    ax.get_figure().savefig(
        dissertacao_imgdir + 'silhouette_by_algorithm_sintetico.pdf')

    ax = pandas.pivot_table(df_sintetico, values='Menor cluster',
                            columns='Algoritmo', index='Nº clusters').plot(
        figsize=(8, 6), grid=True, title='Itens no menor' +
        'cluster / Algoritmo na base \'Sintética\'')
    ax.set_ylabel('Itens no menor cluster')
    ax.get_figure().savefig(
        dissertacao_imgdir + 'smallestcluster_by_algorithm_sintetico.pdf')

    dfdb = df[df['Algoritmo'] == 'DBSCAN']
    dfdb = dfdb.rename(columns={'Variável': '$\epsilon$'})

    ax = pandas.pivot_table(dfdb,
                            values='Dunn',
                            columns='Dataset',
                            index='$\epsilon$').plot(
                                figsize=(8, 6),
                                grid=True,
                                title='Média Dunn / $\epsilon$ para DBSCAN')
    ax.set_ylabel('Dunn')
    ax.get_figure().savefig(dissertacao_imgdir + 'dunn_by_dbscan.pdf')

    ax = pandas.pivot_table(
        dfdb,
        values='Silhueta',
        columns='Dataset',
        index='$\epsilon$').plot(
            figsize=(8, 6),
            grid=True,
            title='Média silhueta / $\epsilon$ para DBSCAN')
    ax.set_ylabel('Silhueta')
    ax.get_figure().savefig(dissertacao_imgdir + 'silhouette_by_dbscan.pdf')

    ax = pandas.pivot_table(
        dfdb,
        values='Menor cluster',
        columns='Dataset',
        index='$\epsilon$').plot(
            figsize=(8, 6),
            grid=True,
            title='Itens no menor cluster / $\epsilon$ para DBSCAN')
    ax.set_ylabel('Itens no menor cluster')
    ax.get_figure().savefig(
        dissertacao_imgdir + 'smallestcluster_by_dbscan.pdf')

    ax = pandas.pivot_table(
        dfdb,
        values='pct. sem classificação',
        columns='Dataset',
        index='$\epsilon$').plot(
            figsize=(8, 6),
            grid=True,
            title='% de itens não classificados / $\epsilon$ para DBSCAN')
    ax.set_ylabel('% Itens')
    ax.get_figure().savefig(dissertacao_imgdir + 'unclassified_by_dbscan.pdf')

    ax = pandas.pivot_table(
        dfdb,
        values='Nº clusters',
        columns='Dataset',
        index='$\epsilon$').plot(
            figsize=(8, 6),
            grid=True,
            title='Nº de clusters / $\epsilon$ para DBSCAN')
    ax.set_ylabel('Nº clusters')
    ax.get_figure().savefig(dissertacao_imgdir + 'nclusters_by_dbscan.pdf')


if __name__ == '__main__':
    dissertacao = '/home/douglas/Desktop/dissertacao/'
    dissertacao_imgdir = dissertacao + 'img/'
    dissertacao_datadir = dissertacao + 'dados/'
    outdir = '/home/douglas/Desktop/out/'

    x = loadDatasets()
    distances.pnorm(x[0][1], 2)
