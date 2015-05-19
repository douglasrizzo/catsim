import numpy as np


def means(clusters, distances):
    """
    Calculates de mean distances for each cluster:

    .. math:: \\mu_g = \\frac{1}{N_g}\\sum_{i=1; j=i+1}^{N_g}d(i, j)
    """
    means = []
    cluster_bins = np.bincount(np.delete(clusters, np.where(clusters == -1)))

    for i in np.arange(0, len(clusters)):
        means[clusters[i]] = 0
        for ii in np.arange(0, len(clusters)):
            if clusters[i] == clusters[ii]:
                means[clusters[i]] += distances[i, ii]

        means[clusters[i]] /= cluster_bins[i]

    return means


def variances(clusters, distances):
    """
    Calculates the variance for each cluster

    .. math:: \\sigma^2_g = \\frac{1}{N_g}\\sum_{i=1; j=i+1}^{N_g} (\mu_g - d(i, j))
    """
    cluster_means = means(clusters, distances)
    variances = []
    cluster_bins = np.bincount(np.delete(clusters, np.where(clusters == -1)))

    for i in np.arange(0, len(clusters)):
        variances[clusters[i]] = 0
        for ii in np.arange(0, len(clusters)):
            if clusters[i] == clusters[ii]:
                variances[
                    clusters[i]] += cluster_means[clusters[i]] - distances[i, ii]

        variances[clusters[i]] /= cluster_bins[i]

    return variances


def mean_variance(clusters, distances):
    """
    Returns the mean variance for all clusters

    .. math:: \\sigma^2 = \\frac{1}{G}\\sum_{g=1}^G \\frac{1}{N}\\sum_{i=1; j=i+1}^N (\mu_g - d(i, j))
    """
    return np.average(variances(clusters, distances))


def dunn(c, distances):
    """
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
    """
    unique_cluster_distances = np.unique(min_cluster_distances(c, distances))
    max_diameter = max(diameter(c, distances))

    if np.size(unique_cluster_distances) > 1:
        return unique_cluster_distances[1] / max_diameter
    else:
        return unique_cluster_distances[0] / max_diameter


def min_cluster_distances(c, distances):
    """
    Calculates the distances between the two nearest points of each cluster.
    """

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
            if c[i] != c[ii] and distances[i, ii] > min_distances[c[i], c[ii]]:
                min_distances[c[i],
                              c[ii]] = min_distances[c[ii],
                                                     c[i]] = distances[i, ii]
    return min_distances


def diameter(c, distances):
    """
    Calculates cluster diameters (the distance between the two farthest data
    points in a cluster)
    """

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
            if c[i] == c[ii] and distances[i, ii] > diameters[c[i]]:
                diameters[c[i]] = distances[i, ii]
    return diameters
