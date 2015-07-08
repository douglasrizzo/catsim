"""Module containing functions relevant to the process of simulating the
application of adaptive tests. Most of this module is based on the work of
[Bar10]_.

.. [Bar10] BARRADA, Juan Ramón et al. A method for the comparison of item
   selection rules in computerized adaptive testing. Applied Psychological
   Measurement, v. 34, n. 6, p. 438-452, 2010."""

import math
import numpy as np
import catsim.cat.irt
from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution


def simCAT(items, clusters, examinees=1, n_itens=20, r_max=1):
    """CAT simulation and validation method proposed by [Bar10]_.
       :param items: an n x 3 matrix containing item parameters
       :param clusters: a list containing item cluster memberships
       :param n_itens: the number of items an examinee will answer during the
       adaptive test
       :param r_max: maximum exposure rate for items
       :type items: numpy.ndarray
       :type clusters: list
       :type n_itens: int
       :type r_max: float
       :return: a list containing two dictionaries:

                **globalResults**:
                The global results of the simulation process.
                    *Qtd. Itens*: number of items in the test;

                    *RMSE*: root mean squared error of the estimations;

                    *Overlap*: overlap rate;

                    *r_max*: maximum exposure rate.

                **localResults**:
                Individual results for each simulated examinee.
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

    # true thetas extracted from a normal distribution
    true_thetas = np.random.normal(0, 1, examinees)

    # adds a column for each item's exposure rate and their cluster membership
    items = np.append(items, np.zeros([items.shape[0], 1]), axis=1)
    items = np.append(
        items, np.array(clusters).reshape(clusters.shape[0], 1), axis=1).astype(np.float64)

    globalResults = []
    localResults = []
    estimatedThetasForThisR = []
    id_itens = []
    items[3] = 0
    for true_theta in true_thetas:

        # estimated theta value
        est_theta = np.random.uniform(-5, 5)

        # keeps indexes of items that were already administered for this
        # examinee
        administered_items = []
        response_vector = []

        for q in range(n_itens):
            # iterates through all items, looking for the item that has the
            # biggest information value, given the estimated theta
            selected_item = None
            max_inf = 0
            for counter, i in enumerate(items):
                if (counter not in administered_items and
                    catsim.cat.irt.inf(
                        est_theta, i[0], i[1], i[2]) > max_inf):
                    selected_item = counter

            # if the selected item's exposure rate is bigger than the
            # maximum exposure rate allowed, the algorithm picks another
            # item from the same cluster the original item came from, with
            # an exposure rate under the allowed constraints, and applies
            # it
            if items[counter, 3] == 0 or (
                    items[counter, 3] != 0 and (items[counter, 3]) >= r_max):

                selected_item_cluster = np.float64(clusters[selected_item])
                # checks whether there is an item in the same cluster with
                # exposure rate below the maximum threshold
                if any(items[np.where(items[:, 4] == selected_item_cluster)][:, 3] < r_max):
                    random_item = None
                    while random_item is None:
                        # print('.')
                        random_item = np.random.randint(0, items.shape[0])
                        if(
                            selected_item_cluster == items[:, 4][random_item] and
                            random_item not in administered_items
                        ):
                            selected_item = random_item
                        else:
                            random_item = None

                # if not, selects the one with smallest exposure rate
                else:
                    # get indices of items in the same cluster
                    item_indexes = np.where(
                        items[:, 4] == selected_item_cluster)[0]

                    selected_item = item_indexes[
                        np.argmin(items[item_indexes][3], axis=0)]

            id_itens.append(selected_item)

            # simulates the examinee's response via the three-parameter
            # logistic function
            acertou = catsim.cat.irt.tpm(
                true_theta,
                items[selected_item][0],
                items[selected_item][1],
                items[selected_item][2]) >= np.random.uniform()

            response_vector.append(acertou)
            # adds the administered item to the pool of administered items
            administered_items.append(selected_item)

            # update the exposure value for this item
            items[selected_item][3] = (
                (items[selected_item][3] * examinees) + 1) / examinees

            # reestimation of the examinee's proficiency: if the response
            # vector contains only success or errors, Dodd's method is used
            # to reestimate the proficiency
            if all(response_vector[0] == response for response in response_vector):
                est_theta = dodd(est_theta, items, acertou)
            # else, a maximum likelihood approach is used
            else:
                try:
                    est_theta = catsim.cat.irt.bruteMLE(
                        response_vector, items[administered_items])
                except:
                    res = differential_evolution(
                        catsim.cat.irt.negativelogLik, bounds=[[-6, 6]],
                        args=(response_vector, items[administered_items]))
                    est_theta = res.x[0]

        # save the results for this examinee simulation
        localResults.append({'Theta': true_theta,
                             'Est. Theta': est_theta,
                             'Id. Itens': id_itens,
                             'r_max': r_max})

        estimatedThetasForThisR.append(est_theta)
    # end true_theta loop

    # save the results for this r value
    globalResults.append({
        'Nº de grupos': len(set(clusters)),
        'Qtd. Itens': n_itens,
        'RMSE': rmse(true_thetas, estimatedThetasForThisR),
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
    :param items: a numpy array containing the parameters of the items in the database. This is necessary to capture the maximum and minimum difficulty levels necessary for the method.
    :param correct: a boolean value informing whether or not the examinee correctly answered the current item.

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
