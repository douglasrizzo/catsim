import numpy as np
import catsim.cat.irt
import catsim.cat.simulate
import catsim.misc.plot
import catsim.misc.results
import catsim.cluster.stats
import catsim.cluster.distances
from scipy.spatial.distance import cdist
import sklearn.datasets


def distances():
    x = sklearn.datasets.load_iris()['data']

    for p in range(1, 10):
        print(np.mean(catsim.cluster.distances.pnorm(x, p=p) -
                      cdist(x, x, 'minkowski', p=p)))

    che1 = cdist(x, x, 'chebyshev')
    che2 = catsim.cluster.distances.chebyshev(x)
    print(np.mean(che1 - che2))


def kmeans():
    x = sklearn.datasets.load_iris()['data']
    for m in ['naive', 'varCovar', 'ward']:
        kmeans(x, 3, init=m, debug=True)


def kmedoids():
    x = sklearn.datasets.load_iris()['data']
    catsim.cluster.kmedoids(catsim.cluster.distances.euclidean(x), 3)


def miscStats():
    x = sklearn.datasets.load_iris()['data']
    minha_cov = catsim.cluster.stats.covariance(x)
    cov_deles = np.cov(x.T)

    print('covariância', 'tá certa!' if np.array_equal(
        minha_cov, cov_deles) else 'tá errada!')
    print(catsim.cluster.stats.coefCorrelation(x))
    print(catsim.misc.stats.bincount(np.array([-4, 0, 1, 1, 3, 2, 1, 7, 23])))
