import numpy as np
from catsim.cluster import distances


def means(x, clusters):
    # '''
    # Calculates de mean distances for each cluster:

    # .. math:: \\mu_g = \\frac{1}{N_g}\\sum_{i=1; j=i+1}^{N_g}d(i, j)
    # '''

    npoints, nfeatures = x.shape
    centroids = np.zeros([max(clusters) + 1, nfeatures])

    for i in range(max(clusters) + 1):
        clusters_aux = np.where(clusters == i)[0]
        centroids[i] = x[clusters_aux].mean(axis=0)

    return centroids


def variances(x, clusters):
    '''
    Calculates the variance for each cluster

    .. math:: \\sigma^2_g = \\frac{1}{N_g}\\sum_{i=1; j=i+1}^{N_g} (\mu_g - d(i, j))
    '''

    npoints, nfeatures = x.shape
    cluster_means = means(x, clusters)
    variances = np.zeros([max(clusters) + 1, nfeatures])
    # cluster_bins = np.bincount(np.delete(clusters, np.where(clusters == -1)))
    D = distances.euclidean(x, cluster_means)

    for i in range(max(clusters) + 1):
        clusters_aux = np.where(clusters == i)[0]
        variances[i] = np.sum(D[clusters_aux])

    return variances


def mean_variance(x, clusters):
    '''
    Returns the mean variance for all clusters

    .. math:: \\sigma^2 = \\frac{1}{G}\\sum_{g=1}^G \\frac{1}{N} \\sum_{i=1; j=i+1}^N (\mu_g - d(i, j))
    '''
    return np.mean(variances(x, clusters))


def dunn(c, distances):
    '''
    Dunn index for cluster validation (the bigger, the better)

    .. math:: D = \\min_{i = 1 \\ldots n_c; j = i + 1\ldots n_c}
    \\left\\lbrace
    \\frac{d \\left( c_i,c_j \\right)}{\\max_{k = 1 \\ldots n_c} \\left(diam
    \\left(c_k \\right) \\right)} \\right\\rbrace

    where :math:`d(c_i,c_j)` represents the distance between clusters
    :math:`c_i` and :math:`c_j`, given by the distances between its two
        closest data points, and :math:`diam(c_k)` is the diameter of cluster
    :math:`c_k`, given by the distance between its two farthest data points.

    The bigger the value of the resulting Dunn index, the better the
    clustering result is considered, since higher values indicate that
    clusters are compact (small :math:`diam(c_k)`) and far apart.

    .. [Kovacs2005] Kovács, F., Legány, C., & Babos, A. (2005). Cluster
    validity measurement techniques. 6th International Symposium of Hungarian
    Researchers on Computational Intelligence.
    '''
    unique_cluster_distances = np.unique(min_cluster_distances(c, distances))
    max_diameter = max(diameter(c, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter


def min_cluster_distances(c, D):
    '''
    Calculates the distances between the two nearest points of each cluster.
    '''

    # creates empty matrix
    min_distances = np.zeros((max(c) + 1, max(c) + 1))

    # iterates through all clusters
    for i in np.arange(len(c)):

        # ignores outliers (in the case of DBSCAN, for example)
        if c[i] == -1:
            continue
        for ii in np.arange(i + 1, len(c)):
            if c[ii] == -1:
                continue

            # checks if data points belong to different clusters and updates
            # the minimum distance between clusters i and ii
            if c[i] != c[ii] and D[i, ii] > min_distances[c[i], c[ii]]:
                min_distances[c[i],
                              c[ii]] = min_distances[c[ii],
                                                     c[i]] = D[i, ii]
    return min_distances


def diameter(c, D):
    '''
    Calculates cluster diameters (the distance between the two farthest data
    points in a cluster)
    '''

    # creates empty matrix
    diameters = np.zeros(max(c) + 1)

    # iterates through all clusters
    for i in np.arange(len(c)):

        # ignores outliers
        if c[i] == -1:
            continue
        for ii in np.arange(i + 1, len(c)):
            if c[ii] == -1:
                continue
            # if c[i] != -1 or c[ii] != -1 and c[i] == c[ii] and distances[i,
            # ii] > diameters[c[i]]:

            # checks if data points are in the same clusters and updates
            # the diameter accordingly
            if c[i] == c[ii] and D[i, ii] > diameters[c[i]]:
                diameters[c[i]] = D[i, ii]
    return diameters
