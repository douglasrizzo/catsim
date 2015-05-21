import numpy as np


def coefvariation(x, axis=0):
    if not isinstance(x, np.matrix):
        x = np.asarray(x)

    mean = np.mean(x, axis=axis)
    stddev = np.std(x, axis=axis)

    print('Means:', mean)
    print('Std. Devs:', stddev)

    result = stddev / \
        mean if axis == 0 else np.transpose(stddev) / np.transpose(mean)

    return result


def coefCorrelation(x):
    cov = covariance(x, False)
    stddev = np.std(x, axis=0)
    n_obs, n_features = x.shape

    corr = np.zeros([n_features, n_features])

    for i in range(n_features):
        for ii in range(i, n_features):
            corr[i, ii] = corr[ii, i] = cov[i, ii] / (stddev[i] * stddev[ii])

    return corr


def covariance(x, minus_one=True):
    x_means = np.mean(x, axis=0)
    n_obs, n_features = x.shape

    covars = np.zeros([n_features, n_features])

    for i in range(n_features):
        for ii in range(i, n_features):
            sum = 0
            for iii in range(n_obs):
                sum += (x[iii, i] - x_means[i]) * (x[iii, ii] - x_means[ii])
            covars[i, ii] = covars[ii, i] = sum / \
                ((n_obs - 1) if minus_one else n_obs)

    return covars


def bincount(x):
    x_max = np.max(x)
    x_min = np.min(x)
    size = abs(x_max) + abs(x_min) + 1

    count = np.zeros(size)

    for i in x:
        count[i + abs(x_min)] += 1

    return count


if __name__ == '__main__':
    from sklearn.datasets import load_iris

    x = load_iris()['data']
    print(x)

    minha_cov = covariance(x)
    cov_deles = np.cov(x.T)

    print('covariância', 'tá certa!' if np.array_equal(
        minha_cov, cov_deles) else 'tá errada!')

    print(coefCorrelation(x))

    print(bincount(np.array([-4, 0, 1, 1, 3, 2, 1, 7, 23])))
