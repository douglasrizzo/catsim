"""Module containing functions relevant to the process of simulating the
application of adaptive tests. Most of this module is based on the work of
[Bar10]_.

.. [Bar10] BARRADA, Juan Ramón et al. A method for the comparison of item
   selection rules in computerized adaptive testing. Applied Psychological
   Measurement, v. 34, n. 6, p. 438-452, 2010."""

import math
import numpy as np
from catsim.cat.irt import bruteMLE, inf, tpm, negativelogLik
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

existent_methods = ['max_info', 'item_info', 'cluster_info', 'weighted_info']
cluster_dependent_methods = ['item_info', 'cluster_info', 'weighted_info']


def simCAT(items, clusters=None, examinees=1, n_itens=20,
           r_max=1, method='item_info'):
    """CAT simulation and validation method proposed by [Bar10]_.

    :param items: an n x 3 matrix containing item parameters
    :type items: numpy.ndarray
    :param clusters: a list containing item cluster memberships
    :type clusters: list
    :param n_itens: the number of items an examinee will answer during the
                    adaptive test
    :type n_itens: int
    :param r_max: maximum exposure rate for items
    :type r_max: float
    :param method: one of the available methods for cluster selection. Given
                   the estimated theta value at each step:

                       ``item_info``: selects the cluster which has the item
                       with maximum information;

                       ``cluster_info``: selects the cluster whose items sum of
                       information is maximum;

                       ``weighted_info``: selects the cluster whose weighted
                       sum of information is maximum. The weighted equals the
                       number of items in the cluster;

    :type method: string
    :return: a list containing two dictionaries:

            **globalResults**: The global results of the simulation process.
                *Qtd. Itens*: number of items in the test;

                *RMSE*: root mean squared error of the estimations;

                *Overlap*: overlap rate;

                *r_max*: maximum exposure rate.

            **localResults**: Individual results for each simulated examinee.
                *Theta*: true theta value of the individual;

                *Est. theta*: estimated theta value of the individual;

                *Id. Itens*: a list containing the id. of the items used
                during the test, in the order they were used;

                *r_max*: maximum exposure rate.

    :rtype: list
    """

    if r_max > 1:
        raise ValueError(
            'r_max must be greater than 0 and lesser or equal to 1')
    if items.shape[1] != 3:
        raise ValueError('item matrix has the incorrect number of parameters')
    if n_itens < 1:
        raise ValueError('Number of items must be positive.')
    if items.shape[0] < n_itens:
        raise ValueError('There are not enough items in the item matrix.')
    if method not in existent_methods:
        raise ValueError(
            'Invalid method, select one from' + str(existent_methods) + '.')
    if method in cluster_dependent_methods and clusters is None:
        raise ValueError(
            'Method {0} cannot be used when clusters is None'.format(method))

    # true thetas extracted from a normal distribution
    true_thetas = np.random.normal(0, 1, examinees)

    # adds a column for each item's exposure rate and their cluster membership
    items = np.append(items, np.zeros([items.shape[0], 1]), axis=1)
    items = np.append(
        items, np.array(clusters).reshape(clusters.shape[0], 1), axis=1).astype(np.float64)

    globalResults = []
    localResults = []
    est_thetas = []

    for true_theta in true_thetas:

        # estimated theta value
        est_theta = np.random.uniform(-5, 5)

        # keeps indexes of items that were already administered for this
        # examinee
        administered_items = []
        response_vector = []

        for q in range(n_itens):
            if method == 'max_info':
                # get the indexes of all items that have not yet been
                # administered, calculate their information value and pick the
                # one with maximum information
                valid_indexes = np.array(
                    list(set(range(items.shape[0])) - set(administered_items)))

                inf_values = [inf(est_theta, i[0], i[1], i[2])
                              for i in items[valid_indexes]]

                valid_indexes = [
                    index for (inf_value, index) in sorted(zip(inf_values, valid_indexes), reverse=True)]

                selected_item = valid_indexes[0]
            else:
                selected_cluster = None
                # this part of the code selects the cluster from which the item at
                # the current point of the test will be chosen
                if method == 'item_info':
                    # finds the item in the matrix which maximizes the
                    # information, given the current estimated theta value
                    max_inf = 0
                    for counter, i in enumerate(items):
                        if inf(est_theta, i[0], i[1], i[2]) > max_inf:
                            # gets the indexes of all the items in the same cluster
                            # as the current selected item that have not been
                            # administered
                            valid_indexes = np.array(list(set(np.nonzero(
                                items[:, 4] == i[4])[0]) - set(administered_items)))

                            # checks if at least one item from this cluster has not
                            # been adminitered to this examinee yet
                            if len(valid_indexes) > 0:
                                selected_cluster = i[4]
                                max_inf = inf(est_theta, i[0], i[1], i[2])

                elif method in ['cluster_info', 'weighted_info']:
                    # calculates the cluster information, depending on the method
                    # selected
                    if method == 'cluster_info':
                        cluster_infos = sum_cluster_infos(
                            est_theta, items, clusters)
                    elif method == 'weighted_info':
                        cluster_infos = weighted_cluster_infos(
                            est_theta, items, clusters)

                    # sorts clusters descending by their information values
                    # this type of sorting was seem on
                    # http://stackoverflow.com/a/6618543
                    sorted_clusters = np.array(
                        [cluster for (inf_value, cluster) in sorted(zip(cluster_infos, set(clusters)), reverse=True)], dtype=float)

                    # walks through the sorted clusters in order
                    for i in range(len(sorted_clusters)):
                        valid_indexes = np.nonzero(
                            items[:, 4] == sorted_clusters[i])[0]

                        # checks if at least one item from this cluster has not
                        # been adminitered to this examinee yet
                        if set(valid_indexes).intersection(administered_items) != set(valid_indexes):
                            selected_cluster = sorted_clusters[i]
                            break
                    # the for loop ends with the cluster that has a) the maximum
                    # information possible and b) at least one item that has not
                    # yet been administered

                assert(selected_cluster is not None)

                # in this part, an item is chosen from the cluster that was
                # selected above
                selected_item = None

                # gets the indexes and information values from the items in the
                # selected cluster that have not been administered
                valid_indexes = np.array(list(set(np.nonzero(
                    items[:, 4] == selected_cluster)[0]) - set(administered_items)))

                # gets the indexes and information values from the items in the
                # selected cluster with r < rmax that have not been
                # administered
                valid_indexes_low_r = np.array(list(set(np.nonzero(
                    (items[:, 4] == selected_cluster) & (items[:, 3] < r_max))[0]) - set(administered_items)))

                if len(valid_indexes_low_r) > 0:
                    # sort both items and their indexes by their information
                    # value
                    inf_values = [inf(est_theta, i[0], i[1], i[2])
                                  for i in items[valid_indexes_low_r]]
                    valid_indexes_low_r = [
                        index for (inf_value, index) in sorted(zip(inf_values, valid_indexes_low_r), reverse=True)]
                    # sorted_items = items[valid_indexes_low_r]

                    selected_item = valid_indexes_low_r[0]

                # if all items in the selected cluster have exceed their r values,
                # select the one with smallest r, regardless of information
                else:
                    inf_values = [inf(est_theta, i[0], i[1], i[2])
                                  for i in items[valid_indexes]]
                    valid_indexes = [
                        index for (inf_value, index) in sorted(zip(inf_values, valid_indexes), reverse=True)]
                    # sorted_items = items[valid_indexes_low_r]

                    selected_item = valid_indexes[0]

            if selected_item is None:
                print('selected_cluster = ' + str(selected_cluster))
                print('inf_values = ' + str(inf_values))
                print('valid_indexes_low_r = ' + str(valid_indexes_low_r))
                print('valid_indexes = ' + str(valid_indexes))
                print('administered_items = ' + str(administered_items))

            assert(selected_item is not None)

            # simulates the examinee's response via the three-parameter
            # logistic function
            response = tpm(
                true_theta,
                items[selected_item][0],
                items[selected_item][1],
                items[selected_item][2]) >= np.random.uniform()

            response_vector.append(response)
            # adds the administered item to the pool of administered items
            administered_items.append(selected_item)

            # update the exposure value for this item
            items[selected_item, 3] = (
                (items[selected_item, 3] * examinees) + 1) / examinees

            # reestimation of the examinee's proficiency: if the response
            # vector contains only success or errors, Dodd's method is used
            # to reestimate the proficiency
            if all(response_vector[0] == response for response in response_vector):
                est_theta = dodd(est_theta, items, response)
            # else, a maximum likelihood approach is used
            else:
                try:
                    est_theta = bruteMLE(
                        response_vector, items[administered_items])
                except:
                    res = differential_evolution(
                        negativelogLik, bounds=[[-6, 6]],
                        args=(response_vector, items[administered_items]))
                    est_theta = res.x[0]

        # save the results for this examinee simulation
        localResults.append({'Theta': true_theta,
                             'Est. Theta': est_theta,
                             'Id. Itens': administered_items,
                             'r_max': r_max})

        est_thetas.append(est_theta)
    # end true_theta loop

    # save the results for this r value
    globalResults.append({
        'Nº de grupos': len(set(clusters)),
        'Qtd. Itens': n_itens,
        'RMSE': rmse(true_thetas, est_thetas),
        'Overlap': overlap_rate(items, n_itens),
        'r_max': r_max})

    return globalResults, localResults


def dodd(theta, items, correct):
    """Method proposed by [Dod90]_ for the reestimation of
    :math:`\\hat{\\theta}` when the response vector is composed entirely of 1s
    or 0s

    .. math::

        \\hat{\\theta}_{t+1} = \\left\\lbrace \\begin{array}{ll}
        \\hat{\\theta}_t+\\frac{b_{max}-\\hat{\\theta_t}}{2} & \\text{if } X_t = 1 \\\\
        \\hat{\\theta}_t-\\frac{\\hat{\\theta}_t-b_{min}}{2} & \\text{if }  X_t = 0
        \\end{array} \\right\\rbrace

    :param theta: the initial profficiency level
    :param items: a numpy array containing the parameters of the items in the
                  database. This is necessary to capture the maximum and minimum
                  difficulty levels necessary for the method.
    :param correct: a boolean value informing whether or not the examinee
                    correctly answered the current item.

    .. [Dod90] Dodd, B. G. (1990). The Effect of Item Selection Procedure and
       Stepsize on Computerized Adaptive Attitude Measurement Using the Rating
       Scale Model. Applied Psychological Measurement, 14(4), 355–366.
       http://doi.org/10.1177/014662169001400403
    """
    b = items[:, 1]
    b_max = max(b)
    b_min = min(b)

    dodd = theta + \
        ((b_max - theta) / 2) if correct else theta - ((theta - b_min) / 2)

    return (dodd)


def rmse(actual, predicted):
    """Root mean squared error:

    .. math:: RMSE = \\sqrt{\\frac{\\sum_{i=1}^{N} (\\hat{\\theta}_i - \\theta_{i})^2}{N}}

    :param actual: a list or 1-D numpy array containing the true profficiency
                   values
    :param predicted: a list or 1-D numpy array containing the estimated
                      profficiency values
    """
    return math.sqrt(mean_squared_error(actual, predicted))


def overlap_rate(items, testSize):
    """Test overlap rate:

    .. math:: T=\\frac{N}{Q}S_{r}^2 + \\frac{Q}{N}

    :param items: a numpy array containing, in the 4th column, the number of
                  times each item was used in the tests.
    :param testSize: an integer informing the number of items in a test.
    """

    bankSize = items.shape[0]
    varR = np.var(items[:, 3])

    T = (bankSize / testSize) * varR + (testSize / bankSize)

    return T


def sum_cluster_infos(theta, items, clusters):
    """Returns the sum of item informations, separated by cluster"""
    cluster_infos = np.zeros((len(set(clusters))))

    for cluster in set(clusters):
        cluster_indexes = np.nonzero(clusters == cluster)[0]

        for item in items[cluster_indexes]:
            cluster_infos[cluster] = cluster_infos[
                cluster] + inf(theta, item[0], item[1], item[2])

    return cluster_infos


def weighted_cluster_infos(theta, items, clusters):
    """Returns the weighted sum of item informations, separated by cluster.
       The weight is the number of items in each cluster."""
    cluster_infos = sum_cluster_infos(theta, items, clusters)
    count = np.bincount(clusters)

    for i in range(len(cluster_infos)):
        cluster_infos[i] = cluster_infos[i] / count[i]

    return cluster_infos
