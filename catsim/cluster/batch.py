import os
import time

import numpy as np
from sklearn.metrics import silhouette_score
from sklearn import cluster as skcluster
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster

import catsim.cat.irt
import catsim.misc.plot
import catsim.cat.simulate
import catsim.misc.results
import catsim.cluster.stats
import catsim.cluster.kmeans
import catsim.cluster.helpers
import catsim.cluster.kmedoids
import catsim.cluster.distances


def batch_kmeans(dataset_name, x, k_init=2, k_end=None, metric='euclidean'):
    n_init = 10
    readable_metric = 'Mahalanobis' if metric == 'mahalanobis' else 'Euclideana'

    # if there is no upper limit for k, it is set as half the number of data
    # points
    if k_end is None:
        k_end = k_init + 1
    else:
        k_end = k_end + 1

    algorithm_name = 'K-means'
    for k in range(k_init, k_end):
        for init_method in ['ward']:
            t1 = time.time()
            res1 = catsim.cluster.kmeans.kmeans(x, k,
                                                init_method=init_method,
                                                iters=1000, n_init=n_init,
                                                debug=False, metric=metric)
            t2 = time.time()

        # calcula menor e maior clusters
        if len(set(res1)) > 1:
            cluster_bins = np.bincount(
                np.delete(res1, np.where(res1 == -1)).astype(np.int64))
            min_c = min(cluster_bins)
            max_c = max(cluster_bins)

            var = np.sum(catsim.cluster.stats.variances(x, res1))
            dun = catsim.cluster.stats.dunn(
                res1, catsim.cluster.distances.euclidean(x))
            silhouette = silhouette_score(x, res1)

            print(time.time(),
                  algorithm_name,
                  dataset_name,
                  readable_metric,
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
                readable_metric,
                init_method,
                np.size(x, 0),
                len(set(res1)),
                (t2 - t1) / n_init,
                min_c,
                max_c,
                var,
                dun,
                silhouette,
                res1,
                '/home/douglas/Desktop/dodokmeans-ward.csv')

        print('K-means with k = ' + str(k) + ' finished')


def batch_kmedoids(dataset_name, x, k_init=2, k_end=None):
    algorithm_name = 'K-medóides'

    if k_init < 2:
        k_init = 2

    if k_end is None:
        k_end = k_init + 1
    else:
        k_end = k_end + 1

    for p in range(1, 8):
        if 1 <= p <= 6:
            D = catsim.cluster.distances.pnorm(x, p=p)
            if p == 1:
                d_name = 'Manhattan'
            elif p == 2:
                d_name = 'Euclideana'
            elif 2 < p < float('inf'):
                d_name = 'Minkowski (p = ' + format(p) + ')'
        elif p == 7:
            p = float('inf')
            d_name = 'Chebyshev'
            D = catsim.cluster.distances.pnorm(x, p=p)
        elif p == 8:
            D = cdist(x, x, 'mahalanobis')
            d_name = 'Mahalanobis'

        for k in range(k_init, k_end):
            var = 0
            for iteration in range(100):
                t1_temp = time.time()
                _, res1_temp = catsim.cluster.kmedoids.kmedoids(D, k,
                                                                iters=1000)
                t2_temp = time.time()
                var_temp = np.sum(catsim.cluster.stats.variances(x, res1_temp))
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
                                                   res1,
                                                   resultados_dir)


def batch_sklearn_kmeans(dataset_name, x):
    """Agrupa os itens em clusters"""

    algorithms = []
    distances = catsim.cluster.distances.euclidean(x)

    # cria estimadores de clustering
    for n_clusters in np.arange(248, 250):
        n_clusters = int(n_clusters)
        algorithm_id = 'K-means'
        algorithm_name = 'K-Means (k = ' + format(n_clusters) + ')'
        algorithm_variable = n_clusters
        algorithm = skcluster.KMeans(n_clusters=n_clusters, init='random')

        print(str(n_clusters) + '            \r', end='\r')
        # roda algoritmos

        var = float('inf')
        for i in range(10):
            t1 = time.time()
            algorithm.fit(x)
            if hasattr(algorithm, 'labels_'):
                aux = algorithm.labels_.astype(np.int)
            else:
                aux = algorithm.predict(x)

            curr_var = np.sum(catsim.cluster.stats.variances(x, aux))

            if curr_var < var:
                y_pred = aux
                var = curr_var

            t2 = time.time()

        # calcula menor e maior clusters
        if len(set(y_pred)) > 1:
            cluster_bins = np.bincount(
                np.delete(y_pred, np.where(y_pred == -1)))
            min_c = min(cluster_bins)
            max_c = max(cluster_bins)

            var = np.sum(catsim.cluster.stats.variances(x, np.delete(
                y_pred, np.where(y_pred == -1))))
            dun = catsim.cluster.stats.dunn(y_pred, distances)
            silhouette = silhouette_score(x, y_pred)

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
                                                   y_pred,
                                                   '/home/douglas/Desktop/kmeans.csv')


def batch_hierarchy(dataset_name, x, k_init=2, k_end=None):
    # if there is no upper limit for k, it is set as half the number of data
    # points
    if k_end is None:
        k_end = k_init + 1
    else:
        k_end = k_end + 1

    for k in range(k_init, k_end):
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
                elif link_metric == 'euclidean':
                    d_name = 'Euclideana'
                elif link_metric == 'mahalanobis':
                    d_name = 'Mahalanobis'
                elif link_metric == 'chebyshev':
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

                    var = np.sum(catsim.cluster.stats.variances(x, clusters))
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
                        clusters,
                        resultados_dir)
