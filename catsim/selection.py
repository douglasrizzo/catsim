from catsim import irt
import numpy
from abc import ABCMeta, abstractmethod


class Selector:
    """Base class representing a CAT item selector."""
    __metaclass__ = ABCMeta

    def __init__(self):
        super(Selector, self).__init__()

    @abstractmethod
    def select(self, items: numpy.ndarray, administered_items: list, est_theta: float) -> int:
        """Get the indexes of all items that have not yet been administered, calculate
        their information value and pick the one with maximum information

        :param items: an item matrix in which the first 3 columns represent item discrimination,
                      difficulty and pseudo-guessing parameters, respectively.
        :param administered_items: a list with the indexes of all administered from the item matrix
        :param est_theta: estimated proficiency value
        :returns: index of the first non-administered item with maximum information
        """
        pass


class MaxInfoSelector(Selector):
    """Selector that returns the first non-administered item with maximum information, given an estimated theta"""

    def __init__(self):
        super().__init__()

    def select(self, items: numpy.ndarray, administered_items: list, est_theta: float) -> int:
        """Get the indexes of all items that have not yet been administered, calculate
        their information value and pick the one with maximum information

        :param items: an item matrix in which the first 3 columns represent item discrimination,
                      difficulty and pseudo-guessing parameters, respectively.
        :param administered_items: a list with the indexes of all administered from the item matrix
        :param est_theta: estimated proficiency value
        :returns: index of the first non-administered item with maximum information
        """
        valid_indexes = numpy.array(list(set(range(items.shape[0])) - set(administered_items)))
        inf_values = [irt.inf(est_theta, i[0], i[1], i[2]) for i in items[valid_indexes]]
        valid_indexes = [
            index for (inf_value, index) in sorted(
                zip(inf_values, valid_indexes),
                reverse=True
            )
        ]

        return valid_indexes[0]


class ClusterSelector(Selector):
    """Cluster-based Item Selection Method.

        .. [Men15] Meneghetti, D. R. (2015). Metolodogia de seleção de itens em testes adaptativos informatizados baseada em agrupamento por similaridade (Mestrado). Centro Universitário da FEI. Retrieved from https://www.researchgate.net/publication/283944553_Metodologia_de_selecao_de_itens_em_Testes_Adaptativos_Informatizados_baseada_em_Agrupamento_por_Similaridade

    :param clusters: a list containing item cluster memberships
    :param r_max: maximum exposure rate for items
    :param method: one of the available methods for cluster selection. Given
                   the estimated theta value at each step:

                       ``item_info``: selects the cluster which has the item
                       with maximum information;

                       ``cluster_info``: selects the cluster whose items sum of
                       information is maximum;

                       ``weighted_info``: selects the cluster whose weighted
                       sum of information is maximum. The weighted equals the
                       number of items in the cluster;

    :param r_control: if `passive` and all items :math:`i` in the selected
                      cluster have :math:`r_i > r^{max}`, applies the item with
                      maximum information; if `aggressive`, applies the item
                      with smallest :math:`r` value.
    """

    def __init__(
        self,
        clusters: list,
        method: str='item_info',
        r_max: float=1,
        r_control: str='passive'
    ):
        available_methods = ['item_info', 'cluster_info', 'weighted_info']
        if method not in available_methods:
            raise ValueError(
                '{0} is not a valid cluster selection method; choose one from {1}'.format(
                    method, available_methods
                )
            )
        available_rcontrol = ['passive', 'aggressive']
        if r_control not in available_rcontrol:
            raise ValueError(
                '{0} is not a valid item exposure control method; choose one from {1}'.format(
                    r_control, available_rcontrol
                )
            )

        self._clusters = clusters
        self._method = method
        self._r_max = r_max
        self._r_control = r_control

    def select(self, items: numpy.ndarray, administered_items: list, est_theta: float) -> int:
        """CAT simulation and validation method proposed by [Bar10]_.

        :param items: an item matrix in which the first 3 columns represent item discrimination,
                      difficulty and pseudo-guessing parameters, respectively.
        :param administered_items: a list with the indexes of all administered from the item matrix
        :param est_theta: estimated proficiency value
        :returns: index of the first non-administered item with maximum information
        """
        selected_cluster = None
        # this part of the code selects the cluster from which the item at
        # the current point of the test will be chosen
        if self._method == 'item_info':
            # finds the item in the matrix which maximizes the
            # information, given the current estimated theta value
            max_inf = 0
            for counter, i in enumerate(items):
                if irt.inf(est_theta, i[0], i[1], i[2]) > max_inf:
                    # gets the indexes of all the items in the same cluster
                    # as the current selected item that have not been
                    # administered
                    valid_indexes = numpy.array(
                        list(
                            set(
                                numpy.nonzero(self._clusters == self._clusters[counter])[
                                    0
                                ]
                            ) - set(
                                administered_items
                            )
                        )
                    )

                    # checks if at least one item from this cluster has not
                    # been adminitered to this examinee yet
                    if len(valid_indexes) > 0:
                        selected_cluster = self._clusters[counter]
                        max_inf = irt.inf(est_theta, i[0], i[1], i[2])

        elif self._method in ['cluster_info', 'weighted_info']:
            # calculates the cluster information, depending on the method
            # selected
            if self._method == 'cluster_info':
                cluster_infos = ClusterSelector.sum_cluster_infos(est_theta, items, self._clusters)
            elif self._method == 'weighted_info':
                cluster_infos = ClusterSelector.weighted_cluster_infos(
                    est_theta, items, self._clusters
                )

            # sorts clusters descending by their information values
            # this type of sorting was seem on
            # http://stackoverflow.com/a/6618543
            sorted_clusters = numpy.array(
                [
                    cluster
                    for (inf_value, cluster) in sorted(
                        zip(cluster_infos, set(self._clusters)),
                        reverse=True
                    )
                ],
                dtype=float
            )

            # walks through the sorted clusters in order
            for i in range(len(sorted_clusters)):
                valid_indexes = numpy.nonzero(items[:, 4] == sorted_clusters[i])[0]

                # checks if at least one item from this cluster has not
                # been adminitered to this examinee yet
                if set(valid_indexes).intersection(administered_items) != set(valid_indexes):
                    selected_cluster = sorted_clusters[i]
                    break
                    # the for loop ends with the cluster that has a) the maximum
                    # information possible and b) at least one item that has not
                    # yet been administered

        assert selected_cluster is not None

        # in this part, an item is chosen from the cluster that was
        # selected above
        selected_item = None

        # gets the indexes and information values from the items in the
        # selected cluster that have not been administered
        valid_indexes = numpy.array(
            list(
                set(numpy.nonzero(self._clusters == selected_cluster)[0]) - set(
                    administered_items
                )
            )
        )

        # gets the indexes and information values from the items in the
        # selected cluster with r < rmax that have not been
        # administered
        valid_indexes_low_r = numpy.array(
            list(
                set(
                    numpy.nonzero(
                        (self._clusters == selected_cluster) & (
                            items[:, 3] < self._r_max
                        )
                    )[0]
                ) - set(administered_items)
            )
        )

        if len(valid_indexes_low_r) > 0:
            # sort both items and their indexes by their information
            # value
            inf_values = [irt.inf(est_theta, i[0], i[1], i[2]) for i in items[valid_indexes_low_r]]
            valid_indexes_low_r = [
                index
                for (inf_value, index) in sorted(
                    zip(inf_values, valid_indexes_low_r),
                    reverse=True
                )
            ]

            selected_item = valid_indexes_low_r[0]

        # if all items in the selected cluster have exceed their r values,
        # select the one with smallest r, regardless of information
        else:
            if self._r_control == 'passive':
                inf_values = [irt.inf(est_theta, i[0], i[1], i[2]) for i in items[valid_indexes]]
                valid_indexes = [
                    index
                    for (inf_value, index) in sorted(
                        zip(inf_values, valid_indexes),
                        reverse=True
                    )
                ]
            elif self._r_control == 'aggressive':
                valid_indexes = [
                    index for (r, index) in sorted(zip(items[valid_indexes, 3], valid_indexes))
                ]

            selected_item = valid_indexes[0]

        return selected_item

    @staticmethod
    def sum_cluster_infos(theta: float, items: numpy.ndarray, clusters: list) -> list:
        """Returns the sum of item informations, separated by cluster

        :param theta: an examinee's :math:`\\theta` value
        :param items: a matrix containing item parameters
        :param clusters: a list containing item cluster memberships, represented by integers
        :returns: list containing the sum of item information values for each cluster"""
        cluster_infos = numpy.zeros((len(set(clusters))))

        for cluster in set(clusters):
            cluster_indexes = numpy.nonzero(clusters == cluster)[0]

            for item in items[cluster_indexes]:
                cluster_infos[cluster] = cluster_infos[cluster] + irt.inf(
                    theta, item[0], item[1], item[2]
                )

        return cluster_infos

    @staticmethod
    def weighted_cluster_infos(theta: float, items: numpy.ndarray, clusters: list):
        """Returns the weighted sum of item informations, separated by cluster.
        The weight is the number of items in each cluster.

        :param theta: an examinee's :math:`\\theta` value
        :param items: a matrix containing item parameters
        :param clusters: a list containing item cluster memberships, represented by integers
        :returns: list containing the sum of item information values for each cluster,
                 divided by the number of items in each cluster"""
        cluster_infos = ClusterSelector.sum_cluster_infos(theta, items, clusters)
        count = numpy.bincount(clusters)

        for i in range(len(cluster_infos)):
            cluster_infos[i] = cluster_infos[i] / count[i]

        return cluster_infos

    @staticmethod
    def sum_cluster_params(items: numpy.ndarray, c: list):
        """Returns the sum of item parameter values for each cluster cluster

        :param items: a matrix containing item parameters.
        :param c: a list containing clustering memeberships.
        :returns: a matrix containing the sum of each parameter by cluster. Lines are clusters, columns are parameters."""
        averages = numpy.zeros((numpy.max(c) + 1, 3))

        for i in numpy.arange(0, numpy.size(c)):
            if c[i] == -1:
                continue
            averages[c[i], 0] += items[i, 0]
            averages[c[i], 1] += items[i, 1]
            averages[c[i], 2] += items[i, 2]

        return averages

    @staticmethod
    def avg_cluster_params(items: numpy.ndarray, c: list):
        """Returns the average values of item parameters by cluster

        :param items: a matrix containing item parameters.
        :param c: a list containing clustering memeberships.
        :returns: a matrix containing the average values of each parameter by cluster. Lines are clusters, columns are parameters."""
        averages = ClusterSelector.sum_cluster_params(items, c)

        occurrences = numpy.bincount(numpy.delete(c, numpy.where(c == -1)).astype(numpy.int64))

        for counter, i in enumerate(occurrences):
            averages[counter, 0] = averages[counter, 0] / i
            averages[counter, 1] = averages[counter, 1] / i
            averages[counter, 2] = averages[counter, 2] / i

        return averages
