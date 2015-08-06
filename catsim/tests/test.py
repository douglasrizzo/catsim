import unittest

import numpy as np
from scipy.spatial.distance import cdist

import sklearn.datasets

import catsim.cat.irt
import catsim.cat.simulate
import catsim.misc.plot
import catsim.misc.results
import catsim.cluster.stats
import catsim.cluster.kmeans
import catsim.cluster.kmedoids
import catsim.cluster.distances

n_clusters = 5
blob, blob_clusters = sklearn.datasets.make_blobs(
    2000, 3, n_clusters)
iris = sklearn.datasets.load_iris()['data']


class TestStuff(unittest.TestCase):
    def distances(self):
        for p in range(1, 10):
            self.assertEqual(np.mean(
                catsim.cluster.distances.pnorm(iris, p=p) -
                cdist(iris, iris, 'minkowski', p=p)), 0)

        che1 = cdist(iris, iris, 'chebyshev')
        che2 = catsim.cluster.distances.chebyshev(iris)
        self.assertEqual(np.mean(che1 - che2))

    def testKmeans(self):
        for m in ['naive', 'ward']:
            catsim.cluster.kmeans.kmeans(blob, n_clusters, init_method=m,
                                         n_init=10, debug=False)

    def kmedoids(self):
        catsim.cluster.kmedoids.kmedoids(
            catsim.cluster.distances.euclidean(blob), n_clusters)

    def miscStats(self):
        minha_cov = catsim.cluster.stats.covariance(iris)
        cov_deles = np.cov(iris.T)

        self.assertTrue(np.array_equal(minha_cov, cov_deles))
        catsim.cluster.stats.coefCorrelation(iris)
        catsim.misc.stats.bincount(np.array([-4, 0, 1, 1, 3, 2, 1, 7, 23]))


if __name__ == '__main__':
    unittest.main()
