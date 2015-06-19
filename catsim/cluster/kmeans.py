import numpy as np
import catsim.cluster.distances
import catsim.misc.stats
import catsim.cluster.stats
from catsim.cluster.helpers import normalize


possible_inits = ['naive', 'varCovar', 'ward']
possible_metrics = ['euclidean', 'mahalanobis']


def naive_init(x, k):
    """Initialize first :math:`k` centroids by randomly choosing
    representatives from the :math:`x` matrix
    """
    return x[np.random.choice(x.shape[0], size=k, replace=False)]


def var_covar_init(x, k):
    """Centroid initialization method proposed by [Eri11]_.

    Only two features of the dataset are used in this method. Feature
    :math:`X_1` is the one that has the largest absolute coefficient of
    variation. :math:`X_2` is the one that has the largest absolute
    correlation coefficient wrt :math:`X_1`.

    Centroid :math:`c_1` is selected as the farthest data point to the mean.
    :math:`c_i, i = 2, \ldots, k` are selected as the farthest data points wrt
    :math:`c_{i-1}`

    .. [Eri11] Erisoglu, M., Calis, N., & Sakallioglu, S. (2011). A new
       algorithm for initial cluster centers in k-means algorithm. Pattern
       Recognition Letters, 32(14), 1701–1705.
       http://doi.org/10.1016/j.patrec.2011.07.011
    """

    # calculate number of data points, number of features, coefficient of
    # variation for each feature and correlation matrix of features
    npoints, nfeatures = x.shape
    coefVar = catsim.misc.stats.coefvariation(x)
    corr = catsim.misc.stats.coefCorrelation(x)

    # chooses the feature that has the largest absolute coefficient of
    # variation as the first axis
    xi = np.argmax(np.absolute(coefVar))
    # chooses the feature that has the largest absolute correlation
    # coefficient wrt. the first feature as the second axis
    xii = np.argmin(np.absolute(corr[xi]))

    # joins both features in a new matrix, calculates the mean of both
    # features and the distances between each data point, using only the two
    # selected features
    x_new = np.hstack([x[:, [xi]], x[:, [xii]]])
    m = np.reshape(np.mean(x_new, axis=0), [1, 2])
    D = catsim.cluster.distances.euclidean(x_new)

    # print(xi, xii, x_new, m, D, m.shape, sep='\n')

    # the first centroid is selected as the farthest data point to the mean
    centroids_indexes = []
    centroids_indexes.append(np.argmax(
        catsim.cluster.distances.euclidean(x_new, m)))

    # additional k - 1 centroids are selected as the farthest data points wrt
    # the last centroid
    for i in range(1, k):
        centroids_indexes.append(np.argmax(D[centroids_indexes[i - 1]]))

    return x[centroids_indexes]


def ward_init(x, k):
    """Ward initialization method for the centroids of k-means algorithm. In this
    method, the data is clustered hierarchically using Ward's function, which
    is a greedy iterative method that joins the two clusters in a way that the
    resulting cluster has the minimum possible variance of all the other
    clusters that could possibly be merged in the previous step

    After the :math:`k` clusters are generated, their centroids are calculated
    and used to initialize k-means.
    """
    from scipy.cluster.hierarchy import ward
    from scipy.cluster.hierarchy import fcluster

    npoints, nfeatures = x.shape
    links = ward(x)
    clusters = fcluster(links, k, criterion='maxclust') - 1
    centroids = np.zeros([k, nfeatures])

    for i in range(len(np.bincount(clusters))):
        clusters_aux = np.where(clusters == i)[0]
        centroids[i] = x[clusters_aux].mean(axis=0)

    return centroids


def kmeans(x, k, init_method='naive', iters=100, n_init=1,
           debug=False, metric='euclidean'):
    """Cluster a set of data points using the k-means algorithm.

    :param x: a matrix in which columns are features and lines are observations
    :type x: numpy.ndarray
    :param k: number of desired clusters
    :type k: integer
    :param init_method: the centroids initialization method. Can be one of the
                   following: ['naive', 'varCovar', 'ward'] in which 'naive'
                   initializes the k centroids in a way that all features are
                   within the minimum and maximum values of each feature of x;
                   'varCovar' [Eri11]_ selects two features that explain most
                   of the dataset and iteratively selects the data points
                   furthest from the center and form each other as the initial
                   centroids; and 'ward' initializes the centroids using the
                   means of k clusters generated via the hierarchical
                   agglomerative clustering procedure that uses Ward function.
    :type init_method: string
    :param iters: number of maximum iterations if convergence is not reached
                  before
    :type iters: integer
    :param n_init: number of re-initializations; the fuction returns the best
                   of n_init results, chosen as the one with minimum sum of
                   intra-cluster variances.
    :type n_init: integer
    :param debug: if using an interactive console, such as iPython, passing
                  True allows for extra output messages separated by user
                  input.
    :type debug: boolean
    """
    if init_method not in possible_inits:
        raise ValueError(
            '"' + init_method + '" is not a valid initialization method')

    if metric not in possible_metrics:
        raise ValueError(
            '"' + metric + '" is not a valid metric')

    npoints, nfeatures = x.shape
    centroidsN = np.zeros([k, nfeatures])
    clusters = np.zeros(npoints)
    final_clusters = np.empty(npoints).fill(-1)

    # initialize centroids according to a given initialization method
    # centroids = naive_centroid_init(x, k)

    if debug:
        print('init = ', init_method)

    # if method is ward, algorithm is deterministic, so no reason to try to
    # minimize the sum of intra-cluster variances
    if init_method == 'ward':
        centroids = ward_init(x, k)
        n_init = 1

    var = float('inf')
    for init in range(n_init):
        if init_method == 'naive':
            centroids = naive_init(x, k)
        elif init_method == 'varCovar':
            centroids = var_covar_init(x, k)
        if debug:
            print('initial centroids', centroids.shape, ':\n', centroids)
            input()
        for t in range(iters):
            # calculates distances from data points to centroids
            # according to a meeting with the statistics teacher, both the
            # Euclidean and Mahalanobis distance use the mean as the best
            # estimator to minimize squared error criterion/sum of
            # intra-cluster variances, so in this implementation of k-means,
            # both distances can be used without significantly changing the
            # code
            if metric == 'euclidean':
                D = catsim.cluster.distances.euclidean(x, centroids)
            elif metric == 'mahalanobis':
                D = catsim.cluster.distances.mahalanobis(x, centroids)
            if debug:
                print('Distances: ', D.shape, '\n', D)

            # assigns data points to the closest centroids
            for i in range(npoints):
                clusters[i] = np.argmin(D[i])

                if debug:
                    print(clusters[i])
                    print(D[i])

            # re-calculate centroids
            for i in range(k):
                clusters_aux = np.where(clusters == i)[0].astype(np.int64)
                centroidsN[i] = x[clusters_aux].mean(axis=0) if len(
                    clusters_aux) > 0 else centroids[i]

                if debug:
                    print('Points in cluster', i, clusters_aux)
                    print('Old centroid for cluster', i, centroids[i])
                    print('New centroid for cluster', i, centroidsN[i])
                    input()

            if np.array_equal(centroids, centroidsN):
                break

            centroids = centroidsN

        clusters = clusters.astype(np.int64)
        this_var = np.sum(catsim.cluster.stats.variances(x, clusters))
        if this_var < var:
            var = this_var
            final_clusters = clusters
    return final_clusters

if __name__ == '__main__':
    from sklearn.cluster import KMeans as sKmeans
    from sklearn.datasets import make_blobs
    n_clusters = 20
    x, clusters = make_blobs(500, 2, n_clusters)

    dodo_res = catsim.cluster.kmeans.kmeans(x, n_clusters, init_method='naive',
                                            iters=1000, n_init=50,
                                            metric='mahalanobis')

    clusters = catsim.cluster.helpers.normalize(clusters)
    dodo_res = catsim.cluster.helpers.normalize(dodo_res)
    sklearn_res = catsim.cluster.helpers.normalize(
        sKmeans(n_clusters=n_clusters, init='random').fit_predict(x))

    dodo_acertos = 0
    sklearn_acertos = 0
    for i in range(len(clusters)):
        if clusters[i] == dodo_res[i]:
            dodo_acertos += 1
        if clusters[i] == sklearn_res[i]:
            sklearn_acertos += 1

    print(len(clusters), '\n',
          dodo_acertos, dodo_acertos /
          len(clusters) *
          100, sum(catsim.cluster.stats.variances(x, dodo_res)), '\n',
          sklearn_acertos, sklearn_acertos / len(clusters) * 100,
          sum(catsim.cluster.stats.variances(x, sklearn_res)))
