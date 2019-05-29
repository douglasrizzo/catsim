from abc import abstractmethod
from warnings import warn

import numpy
from scipy.integrate import quad

from . import irt
from .simulation import Selector, FiniteSelector


def _nearest(array: list, value) -> numpy.ndarray:
    """Returns the indexes of values in `array` that are closest to `value`
    :param array: an array of numeric values
    :param value: a numerical value
    :return: an array containing the indexes of numbers in `array`,
             according to how close their are to `value`
    """
    array = numpy.asarray(array)
    return numpy.abs(array - value).argsort()


class MaxInfoSelector(Selector):
    """Selector that returns the first non-administered item with maximum information, given an estimated theta
       
    :param r_max: maximum exposure rate for items
    """

    def __init__(self, r_max: float = 1):
        super().__init__()
        self._r_max = r_max

    def __str__(self):
        return 'Maximum Information Selector'

    @property
    def r_max(self):
        return self._r_max

    def select(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        est_theta: float = None,
        **kwargs
    ) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :returns: index of the next item to be applied or `None` if there are no more items in the item bank.
        """
        if (index is None or self.simulator is None
            ) and (items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        # first, we'll order items by their information value
        if irt.detect_model(items) <= 2:
            # when the logistic model has the number of parameters <= 2,
            # all items have highest information where theta = b
            ordered_items = _nearest(items[:, 1], est_theta)
        else:
            # else, we'll have to calculate the theta value where information is maximum
            inf_values = irt.max_info_hpc(items)
            ordered_items = _nearest(inf_values, est_theta)

        valid_indexes = [x for x in ordered_items if x not in administered_items]
        if len(valid_indexes) == 0:
            warn('There are no more items to be applied.')
            return None

        # gets the indexes and information values from the items with r < rmax
        valid_indexes_low_r = [index for index in valid_indexes if items[index, 3] < self._r_max]

        # return the item with maximum information from the ones available
        if len(valid_indexes_low_r) > 0:
            selected_item = valid_indexes_low_r[0]
        else:
            selected_item = valid_indexes[0]

        return selected_item


class UrrySelector(Selector):
    """Selector that returns the item whose difficulty parameter is closest to the examinee's proficiency"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Urry Selector'

    def select(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        est_theta: float = None,
        **kwargs
    ) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :returns: index of the next item to be applied or `None` if there are no more items in the item bank.
        """
        if (index is None or self.simulator is None
            ) and (items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        ordered_items = _nearest(items[:, 1], est_theta)
        valid_indexes = [x for x in ordered_items if x not in administered_items]

        if len(valid_indexes) == 0:
            warn('There are no more items to be applied.')
            return None

        return valid_indexes[0]


class LinearSelector(FiniteSelector):
    """Selector that returns item indexes in a linear order, simulating a standard
    (non-adaptive) test.

    :param indexes: the indexes of the items that will be returned in order"""

    def __str__(self):
        return 'Linear Selector'

    def __init__(self, indexes: list):
        super().__init__(len(indexes))
        self._indexes = indexes
        self._current = 0

    @property
    def indexes(self):
        return self._indexes

    @property
    def current(self):
        return self._current

    def select(self, index: int = None, administered_items: list = None, **kwargs) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param administered_items: a list containing the indexes of items that were already administered
        :returns: index of the next item to be applied or `None` if there are no more items in the item bank.
        """
        if (index is None or self.simulator is None) and (administered_items is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if administered_items is None:
            administered_items = self.simulator.administered_items[index]

        if set(self._indexes) <= set(administered_items):
            warn(
                'A new index was asked for, but there are no more item indexes to present.\nCurrent item:\t\t\t{0}\nItems to be administered:\t{1} (size: {2})\nAdministered items:\t\t{3} (size: {4})'
                .format(
                    self._current, sorted(self._indexes), len(self._indexes),
                    sorted(administered_items), len(administered_items)
                )
            )
            return None

        selected_item = [x for x in self._indexes if x not in administered_items][0]

        return selected_item


class RandomSelector(Selector):
    """Selector that randomly selects items for application.

    :param replace: whether to select an item that has already been selected before for this examinee."""

    def __str__(self):
        return 'Random Selector'

    def __init__(self, replace: bool = False):
        super().__init__()
        self._replace = replace

    def select(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        **kwargs
    ) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :returns: index of the next item to be applied or `None` if there are no more items in the item bank.
        """
        if (index is None or
            self.simulator is None) and (items is None or administered_items is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]

        if len(administered_items) >= items.shape[0] and not self._replace:
            warn(
                'A new item was asked for, but there are no more items to present.\nAdministered items:\t{0}\nItem bank size:\t{1}'
                .format(len(administered_items), items.shape[0])
            )
            return None

        if self._replace:
            return numpy.random.choice(items.shape[0])
        else:
            return numpy.random.choice(list(set(range(items.shape[0])) - set(administered_items)))


class ClusterSelector(Selector):
    """Cluster-based Item Selection Method.

        .. [Men15] Meneghetti, D. R. (2015). Metolodogia de seleção de itens em testes
           adaptativos informatizados baseada em agrupamento por similaridade (Mestrado).
           Centro Universitário da FEI. Retrieved from
           https://www.researchgate.net/publication/283944553_Metodologia_de_selecao_de_itens_em_Testes_Adaptativos_Informatizados_baseada_em_Agrupamento_por_Similaridade

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

    def __str__(self):
        return 'Cluster Selector'

    @property
    def r_max(self):
        return self._r_max

    @property
    def clusters(self):
        return self._clusters

    @property
    def method(self):
        return self._method

    @property
    def r_control(self):
        return self._r_control

    def __init__(
        self,
        clusters: list,
        method: str = 'item_info',
        r_max: float = 1,
        r_control: str = 'passive'
    ):
        super().__init__()
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

    def select(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        est_theta: float = None,
        **kwargs
    ) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None
            ) and (items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        selected_cluster = None
        existent_clusters = set(self._clusters)

        # this part of the code selects the cluster from which the item at
        # the current point of the test will be chosen
        if self._method == 'item_info':
            # get the item indexes sorted by their information value
            infos = _nearest(irt.max_info_hpc(items), est_theta)

            evaluated_clusters = set()

            # iterate over every item in order of information value
            for i in range(items.shape[0]):
                # get the current non-examined item
                max_info_item = infos[i]

                # if the cluster of the current item has already been fully examined, go to the next item
                if self._clusters[max_info_item] in evaluated_clusters:
                    continue

                # get the indexes of all items in the same cluster as the current item
                items_in_cluster = numpy.nonzero(
                    [x == self._clusters[max_info_item] for x in self._clusters]
                )[0]

                # if all the items in the current cluster have already been administered (it happens, theoretically),
                # add this cluster to the list of fully evaluated clusters
                if set(items_in_cluster) <= set(administered_items):
                    evaluated_clusters.add(self._clusters[max_info_item])

                    # if all clusters have been evaluated, the loop ends and the method returns None somewhere below
                    if evaluated_clusters == existent_clusters:
                        break

                    # else, if there are still items and clusters to be explored, keep going
                    continue

                # if the algorithm gets here, it means this cluster can be used
                selected_cluster = self._clusters[max_info_item]
                break

        elif self._method in ['cluster_info', 'weighted_info']:
            # calculates the cluster information, depending on the method
            # selected
            if self._method == 'cluster_info':
                cluster_infos = ClusterSelector.sum_cluster_infos(est_theta, items, self._clusters)
            else:
                cluster_infos = ClusterSelector.weighted_cluster_infos(
                    est_theta, items, self._clusters
                )

            # sorts clusters descending by their information values
            # this type of sorting was seem on
            # http://stackoverflow.com/a/6618543
            sorted_clusters = numpy.array(
                [
                    cluster for (inf_value, cluster) in sorted(
                        zip(cluster_infos, set(self._clusters)),
                        key=lambda pair: pair[0],
                        reverse=True
                    )
                ],
                dtype=float
            )

            # walks through the sorted clusters in order
            for i in range(len(sorted_clusters)):
                valid_indexes = numpy.nonzero([r == sorted_clusters[i] for r in items[:, 4]])[0]

                # checks if at least one item from this cluster has not
                # been administered to this examinee yet
                if set(valid_indexes).intersection(administered_items) != set(valid_indexes):
                    selected_cluster = sorted_clusters[i]
                    break
                    # the for loop ends with the cluster that has a) the maximum
                    # information possible and b) at least one item that has not
                    # yet been administered

        # if the test size gets larger than the item bank size, end the test
        if selected_cluster is None:
            warn("There are no more items to be applied.")
            return None

        # in this part, an item is chosen from the cluster that was
        # selected above

        # gets the indexes and information values from the items in the
        # selected cluster that have not been administered
        valid_indexes = [
            index
            for index in numpy.nonzero([cluster == selected_cluster
                                        for cluster in self._clusters])[0]
            if index not in administered_items
        ]

        # gets the indexes and information values from the items in the
        # selected cluster with r < rmax that have not been
        # administered
        valid_indexes_low_r = [
            index for index in valid_indexes
            if items[index, 4] < self._r_max and index not in administered_items
        ]

        if len(valid_indexes_low_r) > 0:
            # return the item with maximum information from the ones available
            inf_values = irt.inf_hpc(est_theta, items[valid_indexes_low_r])
            selected_item = valid_indexes_low_r[numpy.nonzero(inf_values == max(inf_values))[0][0]]

        # if all items in the selected cluster have exceed their r values,
        # select the one with smallest r, regardless of information
        else:
            if self._r_control == 'passive':
                inf_values = irt.inf_hpc(est_theta, items[valid_indexes])
                selected_item = valid_indexes[numpy.nonzero(inf_values == max(inf_values))[0][0]]
            else:
                selected_item = valid_indexes[items[:, 4].index(min(items[:, 4]))]

        return selected_item

    @staticmethod
    def sum_cluster_infos(theta: float, items: numpy.ndarray, clusters: list) -> numpy.ndarray:
        """Returns the sum of item information values, separated by cluster

        :param theta: an examinee's :math:`\\theta` value
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param clusters: a list containing item cluster memberships, represented by integers
        :returns: array containing the sum of item information values for each cluster"""
        cluster_infos = numpy.zeros((len(set(clusters))))

        for cluster in set(clusters):
            cluster_indexes = numpy.nonzero([c == cluster for c in clusters])[0]

            for item in items[cluster_indexes]:
                cluster_infos[cluster] += irt.inf(theta, item[0], item[1], item[2], item[3])

        return cluster_infos

    @staticmethod
    def weighted_cluster_infos(theta: float, items: numpy.ndarray, clusters: list) -> numpy.ndarray:
        """Returns the weighted sum of item information values, separated by cluster.
        The weight is the number of items in each cluster.

        :param theta: an examinee's :math:`\\theta` value
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param clusters: a list containing item cluster memberships, represented by integers
        :returns: array containing the sum of item information values for each cluster,
                  divided by the number of items in each cluster"""
        cluster_infos = ClusterSelector.sum_cluster_infos(theta, items, clusters)
        count = numpy.bincount(clusters)

        for i in range(len(cluster_infos)):
            cluster_infos[i] /= count[i]

        return cluster_infos

    @staticmethod
    def sum_cluster_params(items: numpy.ndarray, c: list):
        """Returns the sum of item parameter values for each cluster

        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param c: a list containing clustering memeberships.
        :returns: a matrix containing the sum of each parameter by cluster. Lines are clusters, columns are parameters.
        """
        averages = numpy.zeros((numpy.max(c) + 1, 4))

        for i in numpy.arange(0, numpy.size(c)):
            if c[i] == -1:
                continue
            averages[c[i], 0] += items[i, 0]
            averages[c[i], 1] += items[i, 1]
            averages[c[i], 2] += items[i, 2]
            averages[c[i], 3] += items[i, 3]

        return averages

    @staticmethod
    def avg_cluster_params(items: numpy.ndarray, c: list):
        """Returns the average values of item parameters by cluster

        :param items:
        :param c: a list containing clustering memeberships.
        :returns: a matrix containing the average values of each parameter by cluster.
                  Lines are clusters, columns are parameters."""
        averages = ClusterSelector.sum_cluster_params(items, c)

        occurrences = numpy.bincount(numpy.delete(c, numpy.where(c == -1)).astype(numpy.int64))

        for counter, i in enumerate(occurrences):
            averages[counter, 0] /= i
            averages[counter, 1] /= i
            averages[counter, 2] /= i
            averages[counter, 3] /= i

        return averages


class StratifiedSelector(FiniteSelector):

    def __str__(self):
        return 'General Stratified Selector'

    def __init__(self, test_size):
        super().__init__(test_size)
        self._organized_items = None

    @staticmethod
    @abstractmethod
    def sort_items(items: numpy.ndarray) -> numpy.ndarray:
        pass

    def preprocess(self):
        # sort item indexes by their discrimination value
        self._organized_items = __class__.sort_items(self.simulator.items)

    def select(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        **kwargs
    ) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param items: a matrix containing item parameters
        :param administered_items: a list containing the indexes of items that were already administered
        :returns: index of the next item to be applied or `None` if there are no more strata to get items from.
        """
        if (index is None or
            self.simulator is None) and (items is None or administered_items is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]

        # select the item in the correct layer, according to the point in the test the examinee is
        slices = numpy.linspace(0, items.shape[0], self._test_size, endpoint=False, dtype='i')

        try:
            pointer = slices[len(administered_items)]
            max_pointer = items.shape[0] if len(
                administered_items
            ) == self._test_size - 1 else slices[len(administered_items) + 1]
        except IndexError:
            warn(
                "{0}: test size is larger than was informed to the selector\nLength of administered items:\t{0}\nTotal length of the test:\t{1}\nNumber of slices:\t{2}"
                .format(self, len(administered_items), self._test_size, len(slices))
            )
            return None

        organized_items = self._organized_items if self._organized_items is not None else self.sort_items(
            items
        )

        # if the selected item has already been administered, select the next one
        while organized_items[pointer] in administered_items:
            pointer += 1
            if pointer == max_pointer:
                raise ValueError(
                    'There are no more items to be selected from stratum {0}'.format(
                        slices[len(administered_items)]
                    )
                )

        return organized_items[pointer]


class AStratSelector(StratifiedSelector):
    """Implementation of the :math:`\\alpha`-stratified selector proposed by
    [Chang99]_, in which the item bank is sorted in ascending order according to the
    items discrimination parameter and then separated into :math:`K` strata
    (:math:`K` being the test size), each stratum containing gradually higher
    average discrimination. The :math:`\\alpha`-stratified selector then selects the
    first non-administered item from stratum :math:`k`, in which :math:`k`
    represents the position in the test of the current item the examinee is being
    presented.

    .. image:: ../sphinx/alpha-strat.*

    :param test_size: the number of items the test contains. The selector uses this parameter
                      to create the correct number of strata.
    """

    def __str__(self):
        return 'a-Stratified Selector'

    def __init__(self, test_size):
        super().__init__(test_size)

    @staticmethod
    def sort_items(items: numpy.ndarray) -> numpy.ndarray:
        return items[:, 0].argsort()


class AStratBBlockSelector(StratifiedSelector):
    """Implementation of the :math:`\\alpha`-stratified selector with :math:`b`
    blocking proposed by [Chang2001]_, in which the item bank is sorted in ascending
    order according to the items difficulty parameter and then separated into
    :math:`M` strata, each stratum containing gradually higher average difficulty.

    Each of the :math:`M` strata is then again separated into :math:`K`
    sub-strata (:math:`k` being the test size), according to their
    discrimination. The final item bank is then ordered such that the first
    sub-strata of each strata forms the first strata of the new ordered item
    bank, and so on. This method tries to balance the distribution of both
    parameters between all strata, after perceiving that they are correlated.

    .. image:: ../sphinx/b-blocking.*

    :param test_size: the number of items the test contains. The selector uses this parameter to
                      create the correct number of strata.
    """

    def __str__(self):
        return 'a-Stratified b-Blocking Selector'

    def __init__(self, test_size):
        super().__init__(test_size)

    @staticmethod
    def sort_items(items: numpy.ndarray) -> numpy.ndarray:
        return numpy.lexsort((items[:, 0], items[:, 1]))


class MaxInfoStratSelector(StratifiedSelector):
    """Implementation of the maximum information stratification (MIS) selector
    proposed by [Bar06]_, in which the item bank is sorted in ascending order
    according to the items maximum information and then separated into :math:`K`
    strata (:math:`K` being the test size), each stratum containing items with
    gradually higher maximum information. The MIS selector then selects the first
    non-administered item from stratum :math:`k`, in which :math:`k` represents the
    position in the test of the current item the examinee is being presented.

    .. image:: ../sphinx/mis.*

    This method claims to work better than the :math:`a`-stratified method by
    [Chang99]_ for the three-parameter logistic model of IRT, since item difficulty
    and maximum information are not positioned in the same place in the proficiency
    scale in 3PL.

    :param test_size: the number of items the test contains. The selector uses this parameter to
                      create the correct number of strata.
    """

    def __str__(self):
        return 'Maximum Information Stratification Selector'

    def __init__(self, test_size):
        super().__init__(test_size)

    @staticmethod
    def sort_items(items: numpy.ndarray) -> numpy.ndarray:
        maxinfo = irt.max_info_hpc(items)
        return irt.inf_hpc(maxinfo, items).argsort()


class MaxInfoBBlockSelector(StratifiedSelector):
    """Implementation of the maximum information stratification with :math:`b`
    blocking (MIS-B) selector proposed by [Bar06]_, in which the item bank is sorted
    in ascending order according to the items difficulty parameter and then
    separated into :math:`M` strata, each stratum containing gradually higher
    average difficulty.

    Each of the :math:`M` strata is then again separated into :math:`K`
    sub-strata (:math:`k` being the test size), according to the items maximum
    information. The final item bank is then ordered such that the first
    sub-strata of each strata forms the first strata of the new ordered item
    bank, and so on. This method tries to balance the distribution of both
    parameters between all strata and works better than the :math:`a`-stratified
    with :math:`b` blocking method by [Chang2001]_ for the three-parameter
    logistic model of IRT, since item difficulty and maximum information are not
    positioned in the same place in the proficiency scale in 3PL. This may also
    apply, although not mentioned by the authors, for the 4PL.

    .. image:: ../sphinx/mis-b.*

    :param test_size: the number of items the test contains. The selector uses this parameter to
                      create the correct number of strata.
    """

    def __str__(self):
        return 'Maximum Information Stratification with b-Blocking Selector'

    def __init__(self, test_size):
        super().__init__(test_size)

    @staticmethod
    def sort_items(items: numpy.ndarray) -> numpy.ndarray:
        maxinfo = irt.max_info_hpc(items)
        return numpy.lexsort((irt.inf_hpc(maxinfo, items), maxinfo))


class The54321Selector(FiniteSelector):
    """Implementation of the 5-4-3-2-1 selector proposed by [McBride83]_, in which,
    at each step :math:`k` of a test of size :math:`K`, an item is chosen from a bin
    containing the :math:`K-k` most informative items in the bank, given the current
    :math:`\\hat\\theta`. As the test progresses, the bin gets smaller and more
    informative items have a higher probability of being chosen by the end of the
    test, when the estimation of ':math:`\\hat\\theta` is more precise. The
    5-4-3-2-1 selector can be viewed as a specialization of the
    :py:class:`catsim.selection.RandomesqueSelector`, in which the bin size of most
    informative items gets smaller as the test progresses.

    :param test_size: the number of items the test contains. The selector uses
                      this parameter to set the bin size"""

    def __str__(self):
        return '5-4-3-2-1 Selector'

    def __init__(self, test_size):
        super().__init__(test_size)

    def select(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        est_theta: float = None,
        **kwargs
    ) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :returns: index of the next item to be applied or `None` if there are no more items in the item bank.
        """
        if (index is None or self.simulator is None
            ) and (items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        # sort item indexes by their information value descending and remove indexes of administered items
        organized_items = [
            x for x in (-irt.inf_hpc(est_theta, items)).argsort() if x not in administered_items
        ]

        bin_size = self._test_size - len(administered_items)

        if len(organized_items) == 0:
            warn('There are no more items to apply.')
            return None

        return numpy.random.choice(organized_items[0:bin_size])


class RandomesqueSelector(Selector):
    """Implementation of the randomesque selector proposed by [Kingsbury89]_, in which,
    at every step of the test, an item is randomly chosen from the :math:`n` most informative
    items in the item bank, :math:`n` being a predefined value (originally 5, but user-defined
    in this implementation)

    :param bin_size: the number of most informative items to be taken into consideration when
                     randomly selecting one of them.
    """

    def __str__(self):
        return 'Randomesque Selector'

    def __init__(self, bin_size):
        super().__init__()
        self._bin_size = bin_size

    @property
    def bin_size(self):
        return self._bin_size

    def select(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        est_theta: float = None,
        **kwargs
    ) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :returns: index of the next item to be applied or `None` if there are no more items in the item bank.
        """
        if (index is None or self.simulator is None
            ) and (items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        # sort item indexes by their information value descending and remove indexes of administered items
        organized_items = [
            x for x in (-irt.inf_hpc(est_theta, items)).argsort() if x not in administered_items
        ]

        if len(organized_items) == 0:
            warn('There are no more items to apply.')
            return None

        return numpy.random.choice(list(organized_items)[:self._bin_size])


class IntervalInfoSelector(Selector):
    """A selector in which, at every step of the test, the item that maximizes
    the integral of the information function at a predetermined ``interval``
    :math:`\\delta` above and below the current :math:`\\hat\\theta` is chosen.

    .. math:: argmax_{i \\in I} \\int_{\\hat\\theta - \\delta}^{\\hat\\theta - \\delta}I_i(\\hat\\theta)

    :param interval: the interval of the integral. If no interval is passed, the
                     integral is calculated from :math:`[-\\infty, \\infty]`.
    """

    def __str__(self):
        return 'Interval Information Selector'

    def __init__(self, interval: float = None):
        super().__init__()
        self._interval = interval if interval is not None else numpy.inf

    @property
    def interval(self):
        return self._interval

    def select(
        self,
        index: int = None,
        items: numpy.ndarray = None,
        administered_items: list = None,
        est_theta: float = None,
        **kwargs
    ) -> int:
        """Returns the index of the next item to be administered.

        :param index: the index of the current examinee in the simulator.
        :param items: a matrix containing item parameters in the format that `catsim` understands
                      (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :returns: index of the next item to be applied or `None` if there are no more items in the item bank.
        """
        if (index is None or self.simulator is None
            ) and (items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator or all of the other optional parameters to use this component independently.'
            )

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        # sort item indexes by the integral of the information function descending and remove indexes of administered items
        organized_items = [
            x for x in (-numpy.array([
                    quad(
                        irt.inf,
                        est_theta - self._interval,
                        est_theta + self._interval,
                        args=(item[0], item[1], item[2], item[3])
                    )[0] for item in items]
            )).argsort() if x not in administered_items
        ]

        if len(organized_items) == 0:
            warn('There are no more items to apply.')
            return None

        return list(organized_items)[0]
