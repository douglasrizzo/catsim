"""Module containing functions relevant to the process of simulating the
application of adaptive tests. Most of this module is based on the work of
[Bar10]_.

.. [Bar10] BARRADA, Juan Ramón et al. A method for the comparison of item
   selection rules in computerized adaptive testing. Applied Psychological
   Measurement, v. 34, n. 6, p. 438-452, 2010."""

from catsim import irt
import numpy as np
from scipy.optimize import differential_evolution, fmin


class Simulator:

    def __init__(self, initialization, selection, reestimation, stop):
        self.existent_methods = ['max_info', 'item_info', 'cluster_info', 'weighted_info']
        self.cluster_dependent_methods = ['item_info', 'cluster_info', 'weighted_info']
        self.initialization = initialization
        self.selection = selection
        self.reestimation = reestimation
        self.stop = stop

    def simulate(self, initialization, selection, reestimation, stop):
        if procedure is not None:
            self.p = procedure

        while not self.p.stop():
            self.p.next()

        return self.p.results

    def simCAT(self, items, clusters=None, examinees=1, n_itens=20, r_max=1,
               method='item_info', optimization='fmin', r_control='passive'):
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

                           ``max_info``: ignores cluster selection altogether and
                           selects the item with maximum information to be applied
                           at each step. This is the traditional item selection
                           method used by CATs, prioritizing precision over item
                           exposure;

                           ``item_info``: selects the cluster which has the item
                           with maximum information;

                           ``cluster_info``: selects the cluster whose items sum of
                           information is maximum;

                           ``weighted_info``: selects the cluster whose weighted
                           sum of information is maximum. The weighted equals the
                           number of items in the cluster;

        :type method: string
        :param optimization: the optimization to be used in order to estimate the
                             :math:`\\hat{\\theta}` values. `hill` for a hill-climbing
                             algorithm; `binary` for a binary search algorithm;
                             `fmin` for scipy's function minimization method;
                             `DE` for scipy's differential evolution. With their default
                             parameters, the first method takes roughly 35 function
                             evaluations to converge; the second takes 40 function
                             evaluations; and the last, between 80 and 100 function
                             evaluations. The default method is `fmin`, due to its speed.
        :type optimization: string
        :param r_control: if `passive` and all items :math:`i` in the selected
                          cluster have :math:`r_i > r^{max}`, applies the item with
                          maximum information; if `aggressive`, applies the item
                          with smallest :math:`r` value.
        :type r_control: string
        :return: a list containing two dictionaries. The first contains the global
                 results of the simulation process.
                    *Qtd. Itens*: number of items in the test;

                    *RMSE*: root mean squared error of the estimations;

                    *Overlap*: overlap rate;

                    *r_max*: maximum exposure rate.

                **localResults**: Individual results for each simulated examinee.
                    *Theta*: true theta value of the individual;

                    *Est. theta*: estimated theta value of the individual;

                    *Id. Itens*: a list containing the id. of the items used
                    during the test, in the order they were used;

                    *r*: exposure rate of the items in the bank, after the
                    simulations

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
        if method not in self.existent_methods:
            raise ValueError(
                'Invalid method, select one from' + str(self.existent_methods) + '.')
        if method in self.cluster_dependent_methods and clusters is None:
            raise ValueError(
                'Method {0} cannot be used when clusters is None'.format(method))
        if optimization not in ['hill', 'binary', 'fmin', 'DE']:
            raise ValueError('Optimization method not supported')
        if r_control not in ['passive', 'aggressive']:
            raise ValueError('Exposure control method not supported')

        # true thetas extracted from a normal distribution
        true_thetas = np.random.normal(0, 1, examinees)
        min_difficulty = np.min(items[:, 1])
        max_difficulty = np.max(items[:, 1])

        # adds a column for each item's exposure rate and their cluster membership
        items = np.append(items, np.zeros([items.shape[0], 1]), axis=1)

        if clusters is None:
            clusters = np.zeros(items.shape[0])

        items = np.append(
            items, np.array(clusters).reshape(clusters.shape[0], 1), axis=1).astype(np.float64)

        localResults = []
        est_thetas = []

        current_examinee = 0
        total_tries = 0
        for true_theta in true_thetas:
            current_examinee += 1
            est_theta = float('inf')

            while abs(est_theta - true_theta) > .5:
                total_tries += 1
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

                        inf_values = [irt.inf(est_theta, i[0], i[1], i[2])
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
                                if irt.inf(est_theta, i[0], i[1], i[2]) > max_inf:
                                    # gets the indexes of all the items in the same cluster
                                    # as the current selected item that have not been
                                    # administered
                                    valid_indexes = np.array(list(set(np.nonzero(
                                        items[:, 4] == i[4])[0]) - set(administered_items)))

                                    # checks if at least one item from this cluster has not
                                    # been adminitered to this examinee yet
                                    if len(valid_indexes) > 0:
                                        selected_cluster = i[4]
                                        max_inf = irt.inf(est_theta, i[0], i[1], i[2])

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
                                [cluster for (inf_value, cluster) in
                                 sorted(zip(cluster_infos, set(clusters)), reverse=True)], dtype=float)

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

                        assert (selected_cluster is not None)

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
                            inf_values = [irt.inf(est_theta, i[0], i[1], i[2])
                                          for i in items[valid_indexes_low_r]]
                            valid_indexes_low_r = [
                                index for (inf_value, index) in sorted(zip(inf_values, valid_indexes_low_r), reverse=True)]

                            selected_item = valid_indexes_low_r[0]

                        # if all items in the selected cluster have exceed their r values,
                        # select the one with smallest r, regardless of information
                        else:
                            if r_control == 'passive':
                                inf_values = [irt.inf(est_theta, i[0], i[1], i[2])
                                              for i in items[valid_indexes]]
                                valid_indexes = [
                                    index for (inf_value, index) in sorted(zip(inf_values, valid_indexes), reverse=True)]
                            elif r_control == 'aggressive':
                                valid_indexes = [
                                    index for (r, index) in sorted(zip(items[valid_indexes,
                                                                             3], valid_indexes))]

                            selected_item = valid_indexes[0]

                    if selected_item is None:
                        print('selected_cluster = ' + str(selected_cluster))
                        print('inf_values = ' + str(inf_values))
                        print('valid_indexes_low_r = ' + str(valid_indexes_low_r))
                        print('valid_indexes = ' + str(valid_indexes))
                        print('administered_items = ' + str(administered_items))

                    assert (selected_item is not None)

                    # simulates the examinee's response via the three-parameter
                    # logistic function
                    response = irt.tpm(
                        true_theta,
                        items[selected_item][0],
                        items[selected_item][1],
                        items[selected_item][2]) >= np.random.uniform()

                    response_vector.append(response)
                    # adds the administered item to the pool of administered items
                    administered_items.append(selected_item)

                    # update the exposure value for this item
                    # items[selected_item, 3] = (
                    #     (items[selected_item, 3] * examinees) + 1) / examinees

                    # reestimation of the examinee's proficiency: if the response
                    # vector contains only success or errors, Dodd's method is used
                    # to reestimate the proficiency
                    if all(response_vector[0] == response for response in response_vector):
                        est_theta = irt.dodd(est_theta, items, response)
                    # else, a maximum likelihood approach is used
                    else:
                        if optimization == 'hill':
                            est_theta = irt.hill_climbing_ml(
                                response_vector, items[administered_items])
                        elif optimization == 'binary':
                            est_theta = irt.binary_search_ml(
                                response_vector, items[administered_items])
                        elif optimization == 'fmin':
                            est_theta = fmin(irt.negativelogLik, est_theta, (response_vector, items[administered_items]))
                        elif optimization == 'DE':
                            est_theta = differential_evolution(
                                irt.negativelogLik, bounds=[
                                    [min_difficulty * 2, max_difficulty * 2]],
                                args=(response_vector, items[administered_items])).x[0]

                            # if abs(est_theta - true_theta) > 1:
                            #     print('....', true_theta, est_theta)

            # items[:, 3] /= examinees

            # print(true_theta, est_theta)

            # update the exposure value for this item
            items[administered_items, 3] = (
                (items[administered_items, 3] * examinees) + 1) / examinees
            est_thetas.append(est_theta)

        # save the results for this examinee simulation
        localResults.append({'Theta': true_theta,
                             'Est. Theta': est_theta,
                             'Id. Itens': administered_items,
                             'r': items[:, 3]})

        # end true_theta loop

        print(examinees, total_tries)

        return {'Nº de grupos': len(set(clusters)),
                'Qtd. Itens': n_itens,
                'RMSE': irt.rmse(true_thetas, est_thetas),
                'Overlap': irt.overlap_rate(items, n_itens),
                'r_max': r_max}, localResults
