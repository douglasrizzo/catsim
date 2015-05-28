import numpy as np
import math
from numpy import random
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import catsim.cat.irt


def simCAT(items, clusters, n_itens=20, tests=1):
    '''
    CAT simulation and validation method proposed by [Barrada2010]

    .. [Barrada2010] BARRADA, Juan Ramón et al. A method for the comparison of
    item selection rules in computerized adaptive testing. Applied
    Psychological Measurement, v. 34, n. 6, p. 438-452, 2010.
    '''
    # true thetas extracted from a normal distribution
    true_thetas = np.random.normal(0, 1, tests, n_itens)

    # adds a column for each item's exposure rate to the item parameter matrix
    items = np.append(items, np.zeros((np.size(items, 0), 1)), 1)

    # 10 maximum exposure rates extracted from a linear interval rangin from
    # .1 to 1
    r_maxes = np.linspace(0.1, 1, 10, dtype=float)

    for r_max in r_maxes:
        for true_theta in true_thetas:

            # estimated theta value
            est_theta = random.uniform(-5, 5)

            # keeps indexes of items that were already administered for this
            # examinee
            administered_items = []

            response_vector = []

            for q in np.arange(n_itens):
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
                if items[counter, 3] >= r_max:
                    selected_item_cluster = clusters[selected_item]
                    random_item = None
                    while random_item is None:
                        random_item = random.randint(0, np.size(items, 0))
                        if(
                            selected_item_cluster == clusters[random_item] and
                            random_item not in administered_items
                        ):
                            selected_item = random_item
                        else:
                            random_item = None

                # simulates the examinee's response via the three-parameter
                # logistic function
                acertou = catsim.cat.irt.tpm(
                    true_theta,
                    items[selected_item][0],
                    items[selected_item][1],
                    items[selected_item][2]) >= random.uniform()

                response_vector.append(acertou)
                # adds the administered item to the pool of administered items
                administered_items.append(selected_item)

                # reestimation of the examinee's proficiency: if the response
                # vector contains only success or errors, Dodd's method is used
                # to reestimate the proficiency
                if all(items[0] == item for item in items):
                    est_theta = dodd(est_theta, items, acertou)
                # else, a maximum likelihood approach is used
                else:
                    res = minimize(
                        catsim.cat.irt.negativelogLik, [est_theta],
                        args=[response_vector, items[administered_items]],
                        options={'disp': True})
                    est_theta = res.x


def dodd(theta, items, acertou):
    '''
    Method proposed by [Dodd1990] for the reestimation of
    :math:`\\hat{\\theta}` when the response vector is composed entirely of 1s
    or 0s

    .. math::
        \\hat{\\theta}_{t+1} = \\left\\lbrace \\begin{array}{ll}
        \\hat{\\theta}_t+\\frac{b_{max}-\\hat{\\theta_t}}{2} & \\text{if } X_t
        \\= 1 \\\\
        \\hat{\\theta}_t-\\frac{\\hat{\\theta}_t-b_{min}}{2} & \\text{if }  X_t
        \\= 0
        \\end{array} \\right\\rbrace

    .. [Dood1990] Dodd, B. G. (1990). The Effect of Item Selection Procedure
    and Stepsize on Computerized Adaptive Attitude Measurement Using the Rating
    Scale Model. Applied Psychological Measurement, 14(4), 355–366.
    http://doi.org/10.1177/014662169001400403
    '''
    b = items[:, 1]
    b_max = max(b)
    b_min = min(b)

    return (theta +
            ((b_max - theta) / 2)) if acertou else (theta -
                                                    ((theta - b_min) / 2))


def rmse(actual, predicted):
    '''
    Root mean squared error

    .. math:: RMSE =
    \\sqrt{\\frac{\\sum_{i=1}^N(\\hat{\\theta}_i-\\theta_i)^2}{N}}
    '''
    return math.pow(math.sqrt(mean_squared_error(actual, predicted)), .5)


def overlap_rate():
    '''
    Test overlap rate

    .. math:: T=\\frac{N}{Q}S_{r}^2 + \\frac{Q}{N}
    '''

    return 0
