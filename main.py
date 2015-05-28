# -*- coding: utf-8 -*-

import os
import time
from multiprocessing import Process
from datetime import timedelta

import numpy as np
from numpy import random

from sklearn.metrics import silhouette_score
from sklearn import cluster as skcluster, preprocessing

import catsim.cat.irt
import catsim.misc.plot
import catsim.cat.simulate
import catsim.misc.results
import catsim.cluster.stats
import catsim.cluster.kmeans
import catsim.cluster.kmedoids
import catsim.cluster.distances

from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


def medias_tpm(x, c):
    '''médias dos parâmetros a, b e c para cada cluster'''
    medias = np.zeros((np.max(c) + 1, 3))

    for i in np.arange(0, np.size(c)):
        if c[i] == -1:
            continue
        medias[c[i], 0] += x[i, 0]
        medias[c[i], 1] += x[i, 1]
        medias[c[i], 2] += x[i, 2]

    ocorrencias = np.bincount(np.delete(c, np.where(c == -1)).astype(np.int64))

    for counter, i in enumerate(ocorrencias):
        medias[counter, 0] = medias[counter, 0] / i
        medias[counter, 1] = medias[counter, 1] / i
        medias[counter, 2] = medias[counter, 2] / i

    return medias


def loadDatasets(enem=True, sintetico=True, enem_n=True, sintetico_n=True):
    '''carrega datasets utilizados no experimento'''

    retorno = []

    if enem or enem_n:
        mt = np.genfromtxt(dissertacao_datadir + 'mt_params.csv',
                           dtype='double', delimiter=',', skip_header=True)
        ch = np.genfromtxt(dissertacao_datadir + 'ch_params.csv',
                           dtype='double', delimiter=',', skip_header=True)
        cn = np.genfromtxt(dissertacao_datadir + 'cn_params.csv',
                           dtype='double', delimiter=',', skip_header=True)
        lc = np.genfromtxt(dissertacao_datadir + 'lc_params.csv',
                           dtype='double', delimiter=',', skip_header=True)
        enem_data = np.concatenate((ch, cn, lc, mt))

        if enem:
            retorno.append(['Enem', enem_data])
        if enem_n:
            retorno.append(['Enem normalizado',
                            preprocessing.scale(enem_data)])

    if sintetico or sintetico_n:
        # base sintética gerada usando as distribuições utilizadas pelo barrada
        # nos artigos de 2008 e 2009 dele
        if not os.path.exists(dissertacao_datadir + 'synth_params.csv'):
            np.savetxt(dissertacao_datadir + 'synth_params.csv',
                       np.array([random.normal(1.2, .25, 500),
                                 random.normal(0, 1, 500),
                                 random.normal(.25, .02, 500)],
                                dtype='double').T,
                       delimiter=',')

        sintetico_data = np.genfromtxt(dissertacao_datadir +
                                       'synth_params.csv',
                                       dtype='double',
                                       delimiter=',')

        if sintetico:
            retorno.append(['Sintético', sintetico_data])
        if sintetico_n:
            retorno.append(['Sintético normalizado',
                            preprocessing.scale(sintetico_data)])

    return retorno


def dodoKmeansTest(dataset_name, x, k_init=2, k_end=None):
    n_init = 100

    if k_end is None:
        k_end = int((np.size(x, 0) / 2) - 1)
    else:
        k_end = k_end + 1


    algorithm_name = 'Dodô K-means'
    for k in range(k_init, k_end):
        for init_method in ['naive']:
            t1 = time.time()
            res1 = catsim.cluster.kmeans.kmeans(x, k,
                                                init_method=init_method,
                                                iters=1000, n_init=n_init,
                                                debug=False)
            t2 = time.time()

        # calcula menor e maior clusters
        if len(set(res1)) > 1:
            cluster_bins = np.bincount(
                np.delete(res1, np.where(res1 == -1)).astype(np.int64))
            min_c = min(cluster_bins)
            max_c = max(cluster_bins)

            var = catsim.cluster.stats.mean_variance(x, res1)
            dun = catsim.cluster.stats.dunn(
              res1, catsim.cluster.distances.euclidean(x))
            silhouette = silhouette_score(x, res1)

            cluster_mediasdir = outdir + 'medias/'
            if not os.path.exists(cluster_mediasdir):
                os.makedirs(cluster_mediasdir)

            np.savetxt(cluster_mediasdir + algorithm_name + '_' +
                       dataset_name + '_medias.csv', medias_tpm(x, res1),
                       delimiter=',')

            print(time.time(),
                  algorithm_name,
                  dataset_name,
                  init_method,
                  np.size(x, 0),
                  len(set(res1)),
                  (t2 - t1) / n_init,
                  min_c,
                  max_c,
                  var,
                  dun,
                  silhouette,
                  str(res1.tolist()).strip(
                                '[]').replace(',', ''))

            catsim.misc.results.saveClusterResults(
                          time.time(),
                          algorithm_name,
                          dataset_name,
                          'Euclideana',
                          init_method,
                          np.size(x, 0),
                          len(set(res1)),
                          (t2 - t1) / n_init,
                          min_c,
                          max_c,
                          var,
                          dun,
                          silhouette,
                          str(res1.tolist()).strip(
                                '[]').replace(',', ''),
                          resultados_dir)


def dodoKmedoidsTest(dataset_name, x):
    algorithm_name = 'K-medóides'

    for p in range(1, 8):
        if p >= 1 and p <= 6:
            D = catsim.cluster.distances.pnorm(x, p=p)
            if p == 1:
                d_name = 'Manhattan'
            elif p == 2:
                d_name = 'Euclideana'
            elif p > 2 and p < float('inf'):
                d_name = 'Minkowski (p = ' + format(p) + ')'
        elif p == 7:
            p = float('inf')
            d_name = 'Chebyshev'
            D = catsim.cluster.distances.pnorm(x, p=p)
        elif p == 8:
            D = cdist(x, x, 'mahalanobis')
            d_name = 'Mahalanobis'

        for k in range(2,  int((np.size(x, 0) / 2) - 1)):
            var = 0
            for iteration in range(100):
                t1_temp = time.time()
                _, res1_temp = catsim.cluster.kmedoids.kmedoids(D, k,
                                                                iters=1000)
                t2_temp = time.time()
                var_temp = catsim.cluster.stats.mean_variance(x, res1_temp)
                if var_temp < var:
                    t1 = t1_temp
                    t2 = t2_temp
                    var = var_temp
                    res1 = res1_temp

        # calcula menor e maior clusters
        if len(set(res1)) > 1:
            cluster_bins = np.bincount(
                np.delete(res1, np.where(res1 == -1)))
            min_c = min(cluster_bins)
            max_c = max(cluster_bins)

            var = catsim.cluster.stats.mean_variance(x, res1)
            dun = catsim.cluster.stats.dunn(
              res1, catsim.cluster.distances.euclidean(x))
            silhouette = silhouette_score(x, res1)

            cluster_mediasdir = outdir + 'medias/'
            if not os.path.exists(cluster_mediasdir):
                os.makedirs(cluster_mediasdir)

            np.savetxt(cluster_mediasdir + algorithm_name + '_' +
                       dataset_name + '_medias.csv', medias_tpm(x, res1),
                       delimiter=',')

            catsim.misc.results.saveClusterResults(time.time(),
                                                   algorithm_name,
                                                   dataset_name,
                                                   d_name,
                                                   p,
                                                   np.size(x, 0),
                                                   len(set(res1)),
                                                   t2 - t1,
                                                   min_c,
                                                   max_c,
                                                   var,
                                                   dun,
                                                   silhouette,
                                                   str(res1.tolist()).strip(
                                                       '[]').replace(',', ''),
                                                   resultados_dir)


def sklearnTests(dataset_name, x):
    '''Agrupa os itens em clusters'''

    algorithms = []
    distances = catsim.cluster.distances.euclidean(x)

    # cria estimadores de clustering
    # for n_clusters in np.arange(2, (np.size(x, 0) / 2) - 1):
    #     n_clusters = int(n_clusters)
    #     # k-Means, ligação média, ligação completa, Ward, espectral
    #     algorithms.extend(
    #         [
    #          ('K-Means', 'K-Means (k = ' + format(n_clusters) + ')',
    #           n_clusters, skcluster.KMeans(n_clusters=n_clusters,
    #                                        init='random')),
    #          ('Hier. ligação média',
    #           'Hier. ligação média (k = ' + format(n_clusters) + ')',
    #           n_clusters, skcluster.AgglomerativeClustering(
    #               linkage='average',
    #               affinity='euclidean',
    #               n_clusters=n_clusters)),
    #          ('Hier. ligação completa',
    #           'Hier. ligação completa (k = ' + format(n_clusters) + ')',
    #           n_clusters, skcluster.AgglomerativeClustering(
    #               linkage='complete',
    #               affinity='euclidean',
    #               n_clusters=n_clusters)),
    #          ('Hier. Ward', 'Hier. Ward (k = ' + format(n_clusters) + ')',
    #           n_clusters, skcluster.AgglomerativeClustering(
    #              linkage='ward',
    #              n_clusters=n_clusters)),
    #          ('Espectral', 'Espectral (k = ' + format(n_clusters) + ')',
    #           n_clusters, skcluster.SpectralClustering(
    #              n_clusters=n_clusters,
    #              eigen_solver='arpack',
    #              affinity="nearest_neighbors"))
    #          ])

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
            cluster_bins = np.bincount(
                np.delete(y_pred, np.where(y_pred == -1)))
            min_c = min(cluster_bins)
            max_c = max(cluster_bins)

            var = catsim.cluster.stats.mean_variance(x, np.delete(
                y_pred, np.where(y_pred == -1)))
            dun = catsim.cluster.stats.dunn(y_pred, distances)
            silhouette = silhouette_score(x, y_pred)

            cluster_mediasdir = outdir + 'medias/'
            if not os.path.exists(cluster_mediasdir):
                os.makedirs(cluster_mediasdir)

            np.savetxt(cluster_mediasdir + algorithm_name + '_' +
                       dataset_name + '_medias.csv', medias_tpm(x, y_pred),
                       delimiter=',')

            catsim.misc.results.saveClusterResults(time.time(),
                                                   algorithm_id,
                                                   dataset_name,
                                                   'Euclideana',
                                                   algorithm_variable,
                                                   np.size(x, 0),
                                                   len(set(y_pred)),
                                                   t2 - t1,
                                                   min_c,
                                                   max_c,
                                                   var,
                                                   dun,
                                                   silhouette,
                                                   str(y_pred.tolist()).strip(
                                                       '[]').replace(',', ''),
                                                   resultados_dir)

            # if plots:
            #     if hasattr(algorithm, 'cluster_centers_'):
            #         centers = algorithm.cluster_centers_
            #     else:
            #         centers = None
            #     catsim.misc.plot.plot3D(x, y_pred, dataset_name +
            #                             ' - ' + algorithm_name,
            #                             centers=centers)
            # if videos:
            #     for ii in range(0, 360, 1):
            #         ax.view_init(elev=10., azim=ii)
            #         print('Gerando imagens para renderizar vídeo: ' +
            #               format(round(100 / 360 * ii, 2)) + '%')
            #         plt.savefig(cluster_graphdir + format(ii) + '.jpg',
            #                     dpi=500,
            #                     bbox_inches='tight')

            #     if os.path.isfile(cluster_graphdir + 'video.avi'):
            #         os.remove(cluster_graphdir + 'video.avi')

            #     call('ffmpeg -i \'' + cluster_graphdir +
            #          '%d.jpg\' -vcodec mpeg4 \'' + cluster_graphdir +
            #          'video.mp4\'',
            #          shell=True)
            #     filelist = [f for f in os.listdir(cluster_graphdir)
            #                 if f.endswith(".jpg")]
            #     for f in filelist:
            #         os.remove(cluster_graphdir + f)


def scipyTests(dataset_name, x):
    npoints, nfeatures = x.shape
    for k in range(2, int(npoints / 2)):  # try for many possible k values

        # try for many possible linkage types
        for link_method in ['single', 'complete', 'average', 'weighted',
                            'ward', 'centroid', 'median']:

            # set algorithm name to go into file
            algorithm_name = 'Hierárquico ' + link_method

            # try for many possible linkage metrics
            # these three linkage methods only accept euclidean distance
            if link_method in ['ward', 'centroid', 'median']:
                link_metrics = ['euclidean']
            # otherwise, test for these main four metrics
            else:
                link_metrics = ['cityblock', 'euclidean',
                                'mahalanobis', 'chebyshev']

            # run algorithm and extract statistics and validations
            for link_metric in link_metrics:
                if link_metric == 'cityblock':
                    d_name = 'Manhattan'
                if link_metric == 'euclidean':
                    d_name = 'Euclideana'
                if link_metric == 'mahalanobis':
                    d_name = 'Mahalanobis'
                if link_metric == 'chebyshev':
                    d_name = 'Chebyshev'

                t1 = time.time()
                links = linkage(x, method=link_method, metric=link_metric)
                clusters = fcluster(links, k, criterion='maxclust') - 1
                t2 = time.time()
                if len(set(clusters)) > 1:
                    cluster_bins = np.bincount(
                        np.delete(clusters, np.where(clusters == -1)))
                    min_c = min(cluster_bins)
                    max_c = max(cluster_bins)

                    var = catsim.cluster.stats.mean_variance(x, clusters)
                    dun = catsim.cluster.stats.dunn(clusters,
                                                    cdist(x, x,
                                                          link_metric))
                    silhouette = silhouette_score(x, clusters)

                    cluster_mediasdir = outdir + 'medias/'
                    if not os.path.exists(cluster_mediasdir):
                        os.makedirs(cluster_mediasdir)

                    np.savetxt(cluster_mediasdir + algorithm_name + '_' +
                               dataset_name + '_medias.csv',
                               medias_tpm(x, clusters),
                               delimiter=',')

                    catsim.misc.results.saveClusterResults(
                      time.time(),
                      algorithm_name,
                      dataset_name,
                      d_name,
                      link_metric,
                      np.size(x, 0),
                      len(set(clusters)),
                      t2 - t1,
                      min_c,
                      max_c,
                      var,
                      dun,
                      silhouette,
                      str(clusters.tolist()).strip('[]').replace(',', ''),
                      resultados_dir)


if __name__ == '__main__':
    dissertacao = '/home/douglas/repos/dissertacao/'
    dissertacao_imgdir = dissertacao + 'img/'
    dissertacao_datadir = dissertacao + 'dados/'
    outdir = '/home/douglas/Desktop/out/'
    resultados_dir = outdir + 'results.csv'

    for dataset_name, x in loadDatasets(enem=False, enem_n=False,
                                        sintetico=False):
        Process(target=dodoKmeansTest,
                args=[dataset_name, x, 95, 197]).start()
        Process(target=dodoKmeansTest,
                args=[dataset_name, x, 198, 299]).start()
        Process(target=dodoKmeansTest,
                args=[dataset_name, x, 300, 400]).start()
        Process(target=dodoKmeansTest,
                args=[dataset_name, x, 401, 500]).start()

    datasets = loadDatasets(enem=False, enem_n=False, sintetico_n=False)

    for dataset_name, x in loadDatasets(enem=False, enem_n=False,
                                        sintetico_n=False):
        Process(target=dodoKmeansTest,
                args=[dataset_name, x, 205, 279]).start()
        Process(target=dodoKmeansTest,
                args=[dataset_name, x, 280, 354]).start()
        Process(target=dodoKmeansTest,
                args=[dataset_name, x, 355, 429]).start()
        Process(target=dodoKmeansTest,
                args=[dataset_name, x, 430, 500]).start()
        # p.join()
        # dodoKmeansTest(dataset_name, x, k_init=205)

        # scipyTests(dataset_name, x)
        # sklearnTests(dataset_name, x)
        # dodoKmedoidsTest(dataset_name, x)

        # results = catsim.misc.results.loadClusterResults(resultados_dir)
        # for result in results:
        #     print(results['Classificações'])
