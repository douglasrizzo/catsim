import numpy
from scipy.integrate import quad

from catsim import irt
from catsim.simulation import Selector


class MaxInfoSelector(Selector):
    """Selector that returns the first non-administered item with maximum information, given an estimated theta"""

    def __init__(self):
        super().__init__()

    def __str__(self):
        return 'Maximum Information Selector'

    def select(self, index: int = None, items: numpy.ndarray = None, administered_items: list = None,
               est_theta: float = None) -> int:
        """Returns the index of the next item to be administered.

        :param items: a matrix containing item parameters in the format that `catsim` understands (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (
                            items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        valid_indexes = [x for x in range(items.shape[0]) if x not in administered_items]
        inf_values = [irt.inf(est_theta, i[0], i[1], i[2], i[3]) for i in items[valid_indexes]]
        valid_indexes = [item_index for (inf_value, item_index) in
                         sorted(zip(inf_values, valid_indexes), key=lambda pair: pair[0], reverse=True)]

        return valid_indexes[0]


class LinearSelector(Selector):
    """Selector that returns item indexes in a linear order, simulating a standard
    (non-adaptive) test.

    :param indexes: the indexes of the items that will be returned in order"""

    def __str__(self):
        return 'Linear Selector'

    def __init__(self, indexes: list):
        super().__init__()
        self._indexes = indexes
        self._current = 0

    @property
    def indexes(self):
        return self._indexes

    @property
    def current(self):
        return self._current

    def select(self, index: int = None, administered_items: list = None) -> int:
        """Returns the index of the next item to be administered.

        :param administered_items: a list containing the indexes of items that were already administered
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (administered_items is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if administered_items is None:
            administered_items = self.simulator.administered_items[index]

        if set(self._indexes).issubset(set(administered_items)):
            raise ValueError(
                'A new index was asked for, but there are no more item indexes to present.\nCurrent item:\t\t\t{0}\nItems to be administered:\t{1} (size: {2})\nAdministered items:\t\t{3} (size: {4})'.format(
                    self._current, sorted(self._indexes), len(self._indexes), sorted(administered_items),
                    len(administered_items)))

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

    def select(self, index: int = None, items: numpy.ndarray = None, administered_items: list = None) -> int:
        """Returns the index of the next item to be administered.

        :param items: a matrix containing item parameters in the format that `catsim` understands (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (items is None or administered_items is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if items is None and administered_items is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]

        if len(administered_items) >= items.shape[0] and not self._replace:
            raise ValueError('A new item was asked for, but there are no more items to present.')

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

    def __init__(self, clusters: list, method: str = 'item_info', r_max: float = 1, r_control: str = 'passive'):
        super().__init__()
        available_methods = ['item_info', 'cluster_info', 'weighted_info']
        if method not in available_methods:
            raise ValueError(
                '{0} is not a valid cluster selection method; choose one from {1}'.format(method, available_methods))
        available_rcontrol = ['passive', 'aggressive']
        if r_control not in available_rcontrol:
            raise ValueError('{0} is not a valid item exposure control method; choose one from {1}'.format(r_control,
                                                                                                           available_rcontrol))

        self._clusters = clusters
        self._method = method
        self._r_max = r_max
        self._r_control = r_control

    def select(self, index: int = None, items: numpy.ndarray = None, administered_items: list = None,
               est_theta: float = None) -> int:
        """Returns the index of the next item to be administered.

        :param items: a matrix containing item parameters in the format that `catsim` understands (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (
                            items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        selected_cluster = None
        # this part of the code selects the cluster from which the item at
        # the current point of the test will be chosen
        if self._method == 'item_info':
            infos = [irt.inf(est_theta, i[0], i[1], i[2], i[3]) for i in items]

            while selected_cluster is None:
                # find item with maximum information
                max_info_item = infos.index(max(infos))

                # gets the indexes of all the items in the same cluster
                # as the current selected item that have not been
                # administered
                valid_indexes = [i for i, x in enumerate(self._clusters) if x == self._clusters[max_info_item]]

                # if all items in the same cluster as the selected have been used,
                # get the next item with maximum information
                if set(valid_indexes).issubset(set(administered_items)):
                    infos[max_info_item] = float('-inf')
                else:
                    selected_cluster = self._clusters[max_info_item]

        elif self._method in ['cluster_info', 'weighted_info']:
            # calculates the cluster information, depending on the method
            # selected
            if self._method == 'cluster_info':
                cluster_infos = ClusterSelector.sum_cluster_infos(est_theta, items, self._clusters)
            elif self._method == 'weighted_info':
                cluster_infos = ClusterSelector.weighted_cluster_infos(est_theta, items, self._clusters)

            # sorts clusters descending by their information values
            # this type of sorting was seem on
            # http://stackoverflow.com/a/6618543
            sorted_clusters = numpy.array([cluster for (inf_value, cluster) in
                                           sorted(zip(cluster_infos, set(self._clusters)), key=lambda pair: pair[0],
                                                  reverse=True)], dtype=float)

            # walks through the sorted clusters in order
            for i in range(len(sorted_clusters)):
                valid_indexes = numpy.nonzero(items[:, 4] == sorted_clusters[i])[0]

                # checks if at least one item from this cluster has not
                # been administered to this examinee yet
                if set(valid_indexes).intersection(administered_items) != set(valid_indexes):
                    selected_cluster = sorted_clusters[i]
                    break
                    # the for loop ends with the cluster that has a) the maximum
                    # information possible and b) at least one item that has not
                    # yet been administered

        # in this part, an item is chosen from the cluster that was
        # selected above

        # gets the indexes and information values from the items in the
        # selected cluster that have not been administered
        valid_indexes = numpy.array(
            list(set(numpy.nonzero(self._clusters == selected_cluster)[0]) - set(administered_items)))

        # gets the indexes and information values from the items in the
        # selected cluster with r < rmax that have not been
        # administered
        valid_indexes_low_r = numpy.array(list(
            set(numpy.nonzero((self._clusters == selected_cluster) & (items[:, 4] < self._r_max))[0]) - set(
                administered_items)))

        if len(valid_indexes_low_r) > 0:
            # sort both items and their indexes by their information
            # value
            inf_values = [irt.inf(est_theta, i[0], i[1], i[2], i[3]) for i in items[valid_indexes_low_r]]
            valid_indexes_low_r = [item_index for (inf_value, item_index) in
                                   sorted(zip(inf_values, valid_indexes_low_r), key=lambda pair: pair[0], reverse=True)]

            selected_item = valid_indexes_low_r[0]

        # if all items in the selected cluster have exceed their r values,
        # select the one with smallest r, regardless of information
        else:
            if self._r_control == 'passive':
                inf_values = [irt.inf(est_theta, i[0], i[1], i[2], i[3]) for i in items[valid_indexes]]
                valid_indexes = [item_index for (inf_value, item_index) in
                                 sorted(zip(inf_values, valid_indexes), key=lambda pair: pair[0], reverse=True)]
            elif self._r_control == 'aggressive':
                valid_indexes = [item_index for (r, item_index) in
                                 sorted(zip(items[valid_indexes, 4], valid_indexes), key=lambda pair: pair[0])]

            selected_item = valid_indexes[0]

        return selected_item

    @staticmethod
    def sum_cluster_infos(theta: float, items: numpy.ndarray, clusters: list) -> list:
        """Returns the sum of item informations, separated by cluster

        :param theta: an examinee's :math:`\\theta` value
        :param items: a matrix containing item parameters in the format that `catsim` understands (see: :py:func:`catsim.cat.generate_item_bank`)
        :param clusters: a list containing item cluster memberships, represented by integers
        :returns: list containing the sum of item information values for each cluster"""
        cluster_infos = numpy.zeros((len(set(clusters))))

        for cluster in set(clusters):
            cluster_indexes = numpy.nonzero(clusters == cluster)[0]

            for item in items[cluster_indexes]:
                cluster_infos[cluster] += irt.inf(theta, item[0], item[1], item[2], item[3])

        return cluster_infos

    @staticmethod
    def weighted_cluster_infos(theta: float, items: numpy.ndarray, clusters: list):
        """Returns the weighted sum of item informations, separated by cluster.
        The weight is the number of items in each cluster.

        :param theta: an examinee's :math:`\\theta` value
        :param items: a matrix containing item parameters in the format that `catsim` understands (see: :py:func:`catsim.cat.generate_item_bank`)
        :param clusters: a list containing item cluster memberships, represented by integers
        :returns: list containing the sum of item information values for each cluster,
                  divided by the number of items in each cluster"""
        cluster_infos = ClusterSelector.sum_cluster_infos(theta, items, clusters)
        count = numpy.bincount(clusters)

        for i in range(len(cluster_infos)):
            cluster_infos[i] /= count[i]

        return cluster_infos

    @staticmethod
    def sum_cluster_params(items: numpy.ndarray, c: list):
        """Returns the sum of item parameter values for each cluster cluster

        :param items: a matrix containing item parameters in the format that `catsim` understands (see: :py:func:`catsim.cat.generate_item_bank`)
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


class AStratifiedSelector(Selector):
    """Implementation of the :math:`\\alpha`-stratified selector proposed by
    [Chang99]_, in which the item bank is sorted in ascending order according to the
    items discrimination parameter and then separated into :math:`K` strata
    (:math:`K` being the test size), each stratum containing gradually higher
    average discrimination. The :math:`\\alpha`-stratified selector then selects the
    first non-administered item from stratum :math:`k`, in which :math:`k`
    represents the position in the test of the current item the examinee is being
    presented.

    .. image:: ../docs/alpha-strat.*

    :param test_size: the number of items the test contains. The selector uses this parameter
    to create the correct number of strata.
    """

    def __str__(self):
        return 'a-Stratified Selector'

    def __init__(self, test_size):
        super().__init__()
        self._organized_items = None
        self._test_size = test_size

    @property
    def test_size(self):
        return self._test_size

    def preprocess(self):
        # sort item indexes by their discrimination value
        self._organized_items = self.simulator.items[:, 0].argsort()

    def select(self, index: int = None, administered_items: list = None) -> int:
        """Returns the index of the next item to be administered.

        :param administered_items: a list containing the indexes of items that were already administered
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (administered_items is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if administered_items is None:
            administered_items = self.simulator.administered_items[index]

        # select the item in the correct layer, according to the point in the test the examinee is
        pointer = len(administered_items) * self._test_size

        # if the selected item has already been administered, select the next one
        while self._organized_items[pointer] in administered_items:
            pointer += 1

        return self._organized_items[pointer]


class AStratifiedBBlockingSelector(Selector):
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

    .. image:: ../docs/b-blocking.*

    :param test_size: the number of items the test contains. The selector uses this parameter to
    create the correct number of strata.
    """

    def __str__(self):
        return 'a-Stratified b-Blocking Selector'

    def __init__(self, test_size):
        super().__init__()
        self._organized_items = None
        self._test_size = test_size

    @property
    def test_size(self):
        return self._test_size

    def preprocess(self):
        # sort item indexes by their difficulty, then their discrimination value
        self._organized_items = numpy.lexsort((self.simulator.items[:, 0], self.simulator.items[:, 1]))

    def select(self, index: int = None, administered_items: list = None) -> int:
        """Returns the index of the next item to be administered.

        :param administered_items: a list containing the indexes of items that were already administered
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (administered_items is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if administered_items is None:
            administered_items = self.simulator.administered_items[index]

        # select the item in the correct layer, according to the point in the test the examinee is
        selected_item = len(administered_items)

        # if the selected item has already been administered, select the next one
        while self._organized_items[selected_item] in administered_items:
            selected_item += self._test_size

        return self._organized_items[selected_item]


class MaxInfoStratificationSelector(Selector):
    """Implementation of the maximum information stratification (MIS) selector
    proposed by [Bar06]_, in which the item bank is sorted in ascending order
    according to the items maximum information and then separated into :math:`K`
    strata (:math:`K` being the test size), each stratum containing items with
    gradually higher maximum information. The MIS selector then selects the first
    non-administered item from stratum :math:`k`, in which :math:`k` represents the
    position in the test of the current item the examinee is being presented.

    .. image:: ../docs/mis.*

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
        super().__init__()
        self._organized_items = None
        self._test_size = test_size

    @property
    def test_size(self):
        return self._test_size

    def preprocess(self):
        # sort item indexes by their maximum information value
        self._organized_items = numpy.array(
            [irt.inf(irt.max_info(item[0], item[1], item[2], item[3]), item[0], item[1], item[2], item[3]) for item in
             self.simulator.items]).argsort()

    def select(self, index: int = None, administered_items: list = None) -> int:
        """Returns the index of the next item to be administered.

        :param administered_items: a list containing the indexes of items that were already administered
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (administered_items is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if administered_items is None:
            administered_items = self.simulator.administered_items[index]

        # select the item in the correct layer, according to the point in the test the examinee is
        pointer = len(administered_items) * self._test_size

        # if the selected item has already been administered, select the next one
        while self._organized_items[pointer] in administered_items:
            pointer += 1

        return self._organized_items[pointer]


class MaxInfoBBlockingSelector(Selector):
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

    .. image:: ../docs/mis-b.*

    :param test_size: the number of items the test contains. The selector uses this parameter to
    create the correct number of strata.
    """

    def __str__(self):
        return 'Maximum Information Stratification with b-Blocking Selector'

    def __init__(self, test_size):
        super().__init__()
        self._organized_items = None
        self._test_size = test_size

    @property
    def test_size(self):
        return self._test_size

    def preprocess(self):
        # sort item indexes by their theta_max values, then by their maximum information
        self._organized_items = numpy.lexsort(([irt.inf(irt.max_info(item[0], item[1], item[2], item[3]), item[0],
                                                        item[1], item[2], item[3]) for item in self.simulator.items],
                                               [irt.max_info(item[0], item[1], item[2], item[3]) for item in
                                                self.simulator.items]))

    def select(self, index: int = None, administered_items: list = None) -> int:
        """Returns the index of the next item to be administered.

        :param administered_items: a list containing the indexes of items that were already administered
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (administered_items is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if administered_items is None:
            administered_items = self.simulator.administered_items[index]

        # select the item in the correct layer, according to the point in the test the examinee is
        pointer = len(administered_items)

        # if the selected item has already been administered, select the next one
        while self._organized_items[pointer] in administered_items:
            pointer += self._test_size

        return self._organized_items[pointer]


class The54321Selector(Selector):
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
        super().__init__()
        self._test_size = test_size

    @property
    def test_size(self):
        return self._test_size

    def select(self, index: int = None, items: numpy.ndarray = None, administered_items: list = None,
               est_theta: float = None) -> int:
        """Returns the index of the next item to be administered.

        :param items: a matrix containing item parameters in the format that `catsim` understands (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (
                            items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        # sort item indexes by their information value and remove indexes of administered items
        organized_items = [x for x in numpy.array(
            [irt.inf(est_theta, item[0], item[1], item[2], item[3]) for item in items]).argsort() if
                           x not in administered_items]

        bin_size = self._test_size - len(administered_items)

        if len(organized_items) == 0:
            raise ValueError('There are no more items to apply.')

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

    def select(self, index: int = None, items: numpy.ndarray = None, administered_items: list = None,
               est_theta: float = None) -> int:
        """Returns the index of the next item to be administered.

        :param items: a matrix containing item parameters in the format that `catsim` understands (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (
                            items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        # sort item indexes by their information value and remove indexes of administered items
        organized_items = [x for x in numpy.array(
            [irt.inf(est_theta, item[0], item[1], item[2], item[3]) for item in items]).argsort() if
                           x not in administered_items]

        if len(organized_items) == 0:
            raise ValueError('There are no more items to apply.')

        return numpy.random.choice(list(organized_items)[:self._bin_size])


class IntervalIntegrationSelector(Selector):
    """Implementation of an interval integration selector in which, at every step of
    the test, the item that maximizes the information function integral at a
    predetermined ``interval`` :math:`\\delta` above and below the current
    :math:`\\hat\\theta` is chosen.

    .. math:: argmax_{i \\in I} \\int_{\\hat\\theta - \\delta}^{\\hat\\theta - \\delta}I_i(\\hat\\theta)

    :param interval: the interval of the integral. If no interval is passed, the
                     integral is calculated from :math:`[-\\infty, \\infty]`.
    """

    def __str__(self):
        return 'Interval Integration Selector'

    def __init__(self, interval: float = None):
        super().__init__()
        self._interval = interval if interval is not None else numpy.inf

    @property
    def interval(self):
        return self._interval

    def select(self, index: int = None, items: numpy.ndarray = None, administered_items: list = None,
               est_theta: float = None) -> int:
        """Returns the index of the next item to be administered.

        :param items: a matrix containing item parameters in the format that `catsim` understands (see: :py:func:`catsim.cat.generate_item_bank`)
        :param administered_items: a list containing the indexes of items that were already administered
        :param est_theta: a float containing the current estimated proficiency
        :param index: the index of the current examinee in the simulator.
        :returns: index of the next item to be applied.
        """
        if (index is None or self.simulator is None) and (
                            items is None or administered_items is None or est_theta is None):
            raise ValueError(
                'Either pass an index for the simulator, or the item bank, administered_items and est_theta to select the next item independently.')

        if items is None and administered_items is None and est_theta is None:
            items = self.simulator.items
            administered_items = self.simulator.administered_items[index]
            est_theta = self.simulator.latest_estimations[index]

        # sort item indexes by the integral of the information function and remove indexes of administered items
        organized_items = [x for x in numpy.array([quad(irt.inf, est_theta - self._interval, est_theta + self._interval,
                                                        args=(item[0], item[1], item[2], item[3]))[0] for item in
                                                   items]).argsort() if x not in administered_items]

        if len(organized_items) == 0:
            raise ValueError('There are no more items to apply.')

        return list(organized_items)[0]
