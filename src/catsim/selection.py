import operator
from abc import abstractmethod
from typing import Any

import numpy
from numpy.typing import NDArray
from scipy.integrate import quad

from . import irt
from .item_bank import ItemBank
from .simulation import FiniteSelector, Selector

# Constants
MIN_ITEM_BANK_COLUMNS = 5  # Minimum number of columns required in item bank (includes exposure rate column)


class MaxInfoSelector(Selector):
  """Selector that returns the first non-administered item with maximum information for the current theta estimate.

  This is one of the most common item selection methods in CAT, choosing the item
  that provides the most information at the examinee's current estimated ability level.

  Parameters
  ----------
  r_max : float, optional
      Maximum exposure rate for items. Items with exposure rates >= r_max will not
      be selected unless no other items are available. Default is 1 (no restriction).
  """

  def __init__(self, r_max: float = 1) -> None:
    """Initialize a MaxInfoSelector object.

    Parameters
    ----------
    r_max : float, optional
        Maximum exposure rate for items. Default is 1.

    Raises
    ------
    ValueError
        If r_max is not between 0 and 1.
    """
    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)

    super().__init__()
    self._r_max = float(r_max)

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Maximum Information Selector"

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate for items the selector accepts.

    Returns
    -------
    float
        Maximum exposure rate for items the selector accepts.
    """
    return self._r_max

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    est_theta: float | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    items : numpy.ndarray or None, optional
        A matrix containing item parameters in the format that `catsim` understands
        (see: :py:meth:`ItemBank.generate_item_bank`). Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    item_bank, administered_items, est_theta = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      **kwargs,
    )

    if item_bank is None:
      msg = "item_bank parameter cannot be None"
      raise ValueError(msg)
    if administered_items is None:
      msg = "administered_items parameter cannot be None"
      raise ValueError(msg)
    if est_theta is None:
      msg = "est_theta parameter cannot be None"
      raise ValueError(msg)

    # sort items by their information value
    ordered_items = self._sort_by_info(item_bank, est_theta)
    # remove administered ones
    valid_indexes = self._get_non_administered(ordered_items, administered_items)

    if len(valid_indexes) == 0:
      msg = "There are no more items to apply."
      raise RuntimeError(msg)

    # gets the indexes and information values from the items with r < rmax
    valid_indexes_low_r = valid_indexes

    # ItemBank always validates and has 5 columns, so no need to check

    valid_indexes_low_r = [index for index in valid_indexes if item_bank.exposure_rates[index] < self._r_max]
    # return the item with maximum information from the ones available
    return valid_indexes_low_r[0] if len(valid_indexes_low_r) > 0 else valid_indexes[0]


class UrrySelector(Selector):
  """Selector that returns the item whose difficulty parameter is closest to the examinee's ability.

  This method, known as Urry's method, selects items based on the proximity of their
  difficulty (b parameter) to the current ability estimate, which is particularly
  effective for 1PL and 2PL models where information is maximized when b = theta.
  """

  def __init__(self) -> None:
    """Initialize a UrrySelector object."""
    super().__init__()

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Urry Selector"

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    est_theta: float | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    items : numpy.ndarray or None, optional
        A matrix containing item parameters in the format that `catsim` understands
        (see: :py:meth:`ItemBank.generate_item_bank`). Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    item_bank, administered_items, est_theta = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      **kwargs,
    )

    if est_theta is None:
      msg = "est_theta parameter cannot be None"
      raise ValueError(msg)
    if administered_items is None:
      msg = "administered_items parameter cannot be None"
      raise ValueError(msg)
    if item_bank is None:
      msg = "item_bank parameter cannot be None"
      raise ValueError(msg)

    ordered_items = self._sort_by_b(item_bank, est_theta)
    valid_indexes = self._get_non_administered(ordered_items, administered_items)

    if len(valid_indexes) == 0:
      msg = "There are no more items to apply."
      raise RuntimeError(msg)

    return valid_indexes[0]


class LinearSelector(FiniteSelector):
  """Selector that returns item indexes in a linear order, simulating a standard (non-adaptive) test.

  This selector is useful for baseline comparisons or for administering a fixed set
  of items in a predetermined order.

  Parameters
  ----------
  indexes : list[int]
      The indexes of the items that will be returned in order.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Linear Selector"

  def __init__(self, indexes: list[int]) -> None:
    """Initialize a LinearSelector object.

    Parameters
    ----------
    indexes : list[int]
        List of item indices to be administered in order.

    Raises
    ------
    ValueError
        If indexes list is empty or contains invalid (negative) values.
    """
    if len(indexes) == 0:
      msg = "indexes list cannot be empty"
      raise ValueError(msg)
    if not all(i >= 0 for i in indexes):
      msg = "All indexes must be non-negative integers"
      raise ValueError(msg)

    super().__init__(len(indexes))
    self._indexes = indexes
    self._current = 0

  @property
  def indexes(self) -> list[int]:
    """The indexes of the items that will be returned in order."""
    return self._indexes

  @property
  def current(self) -> int:
    """The index of the current item."""
    return self._current

  def select(self, index: int | None = None, administered_items: list[int] | None = None, **kwargs: Any) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    (administered_items,) = self._prepare_args(
      return_administered_items=True, index=index, administered_items=administered_items, **kwargs
    )
    valid_indexes = self._get_non_administered(self._indexes, administered_items)
    if len(valid_indexes) == 0:
      msg = (
        f"A new index was asked for, but there are no more item indexes to present.\n"
        f"Current item:\t\t\t{self._current}\n"
        f"Items to be administered:\t{sorted(self._indexes)} (size: {len(self._indexes)})\n"
        f"Administered items:\t\t{sorted(administered_items)} (size: {len(administered_items)})"
      )
      raise RuntimeError(msg)
    return valid_indexes[0]


class RandomSelector(Selector):
  """Selector that randomly selects items for application.

  This selector is useful for baseline comparisons or for studying the impact of
  item selection strategies.

  Parameters
  ----------
  replace : bool, optional
      Whether to select an item that has already been selected before for this examinee.
      Default is False.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Random Selector"

  def __init__(self, replace: bool = False) -> None:
    """Initialize a RandomSelector object.

    Parameters
    ----------
    replace : bool, optional
        Whether to select an item that has already been selected before for this examinee.
        Default is False.
    """
    super().__init__()
    self._replace = replace

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    items : numpy.ndarray or None, optional
        A matrix containing item parameters in the format that `catsim` understands
        (see: :py:meth:`ItemBank.generate_item_bank`). Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    **kwargs
        Additional keyword arguments. Notably:

        * **rng** (:py:class:`numpy.random.Generator`) -- Random number generator used by the object,
          guarantees reproducibility of outputs.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    item_bank, administered_items, rng = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_rng=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      **kwargs,
    )

    assert item_bank is not None
    assert administered_items is not None

    if len(administered_items) >= item_bank.n_items and not self._replace:
      msg = "There are no more items to apply."
      raise RuntimeError(msg)

    if self._replace:
      return rng.choice(item_bank.n_items)
    valid_indexes = self._get_non_administered(list(range(item_bank.n_items)), administered_items)
    return rng.choice(valid_indexes)


class ClusterSelector(Selector):
  """Cluster-based Item Selection Method.

  This method groups items into clusters and selects items from clusters based on
  their information characteristics, helping to balance item exposure across the
  item bank [Men15]_.

  .. [Men15] Meneghetti, D. R. (2015). Metolodogia de seleção de itens em testes
     adaptativos informatizados baseada em agrupamento por similaridade (Mestrado).
     Centro Universitário da FEI. Retrieved from
     https://www.researchgate.net/publication/283944553_Metodologia_de_selecao_de_itens_em_Testes_Adaptativos_Informatizados_baseada_em_Agrupamento_por_Similaridade

  Parameters
  ----------
  clusters : list[int]
      A list containing item cluster memberships (integers representing cluster IDs).
  method : str, optional
      One of the available methods for cluster selection. Given the estimated theta
      value at each step:

      - 'item_info': selects the cluster which has the item with maximum information
      - 'cluster_info': selects the cluster whose items' sum of information is maximum
      - 'weighted_info': selects the cluster whose weighted sum of information is
        maximum (weighted by the number of items in the cluster)

      Default is 'item_info'.
  r_max : float, optional
      Maximum exposure rate for items. Default is 1.
  r_control : str, optional
      Item exposure control method. If 'passive' and all items in the selected
      cluster have exposure rates > r_max, applies the item with maximum information.
      If 'aggressive', applies the item with smallest exposure rate. Default is 'passive'.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Cluster Selector"

  @property
  def r_max(self) -> float:
    """Return the maximum exposure rate for items the selector accepts."""
    return self._r_max

  @property
  def clusters(self) -> list[int]:
    """Return the clusters each item belongs to."""
    return self._clusters

  @property
  def method(self) -> str:
    """Return the method used for cluster selection."""
    return self._method

  @property
  def r_control(self) -> str:
    """Return the item exposure control method."""
    return self._r_control

  def __init__(
    self,
    clusters: list[int],
    method: str = "item_info",
    r_max: float = 1,
    r_control: str = "passive",
  ) -> None:
    """Initialize a ClusterSelector object.

    Parameters
    ----------
    clusters : list[int]
        List of integers defining item cluster associations.
    method : str, optional
        Cluster selection method, one of ['item_info', 'cluster_info', 'weighted_info'].
        Default is 'item_info'.
    r_max : float, optional
        Maximum item exposure rate. Default is 1.
    r_control : str, optional
        Item exposure control method, one of ['passive', 'aggressive']. Default is 'passive'.

    Raises
    ------
    ValueError
        If parameters are invalid (empty clusters list, negative cluster values, invalid
        method or r_control, or r_max not between 0 and 1).
    """
    if len(clusters) == 0:
      msg = "clusters list cannot be empty"
      raise ValueError(msg)
    if not all(c >= 0 for c in clusters):
      msg = "All cluster values must be non-negative integers"
      raise ValueError(msg)

    available_methods = ["item_info", "cluster_info", "weighted_info"]
    if method not in available_methods:
      msg = f"{method} is not a valid cluster selection method; choose one from {available_methods}"
      raise ValueError(msg)

    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)

    available_rcontrol = ["passive", "aggressive"]
    if r_control not in available_rcontrol:
      msg = f"{r_control} is not a valid item exposure control method; choose one from {available_rcontrol}"
      raise ValueError(msg)

    super().__init__()
    self._clusters = clusters
    self._method = method
    self._r_max = float(r_max)
    self._r_control = r_control

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    est_theta: float | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    items : numpy.ndarray or None, optional
        A matrix containing item parameters in the format that `catsim` understands
        (see: :py:meth:`ItemBank.generate_item_bank`). Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    int or None
        Index of the next item to be applied.
    """
    item_bank, administered_items, est_theta = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      **kwargs,
    )

    assert item_bank is not None
    assert administered_items is not None
    assert est_theta is not None

    selected_cluster = None
    existent_clusters = set(self._clusters)

    # this part of the code selects the cluster from which the item at
    # the current point of the test will be chosen
    if self._method == "item_info":
      # get the item indexes sorted by their information value
      infos = self._sort_by_info(item_bank, est_theta)

      evaluated_clusters = set()

      # iterate over every item in order of information value
      for i in range(item_bank.n_items):
        # get the current non-examined item
        max_info_item = infos[i]

        # if the cluster of the current item has already been fully examined, go to the next item
        if self._clusters[max_info_item] in evaluated_clusters:
          continue

        # get the indexes of all items in the same cluster as the current item
        items_in_cluster = numpy.nonzero([x == self._clusters[max_info_item] for x in self._clusters])[0]

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

    elif self._method in {"cluster_info", "weighted_info"}:
      # calculates the cluster information, depending on the method
      # selected
      if self._method == "cluster_info":
        cluster_infos = ClusterSelector.sum_cluster_infos(est_theta, item_bank, self._clusters)
      else:
        cluster_infos = ClusterSelector.weighted_cluster_infos(est_theta, item_bank, self._clusters)

      # sorts clusters descending by their information values
      # this type of sorting was seem on
      # http://stackoverflow.com/a/6618543
      sorted_clusters = numpy.array(
        [
          cluster
          for (inf_value, cluster) in sorted(
            zip(cluster_infos, set(self._clusters), strict=False),
            key=operator.itemgetter(0),
            reverse=True,
          )
        ],
        dtype=float,
      )

      # walks through the sorted clusters in order
      for i in range(len(sorted_clusters)):
        valid_indexes = numpy.nonzero([r == sorted_clusters[i] for r in item_bank.exposure_rates])[0]

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
      msg = "There are no more items to be applied."
      raise RuntimeError(msg)

    # in this part, an item is chosen from the cluster that was
    # selected above

    # gets the indexes and information values from the items in the
    # selected cluster that have not been administered
    valid_indexes = self._get_non_administered(
      numpy.nonzero([cluster == selected_cluster for cluster in self._clusters])[0],
      administered_items,
    )

    # gets the indexes and information values from the items in the
    # selected cluster with r < rmax that have not been administered
    valid_indexes_low_r = valid_indexes
    # ItemBank always validates and has 5 columns, so no need to check

    valid_indexes_low_r = [
      index
      for index in valid_indexes
      if item_bank.exposure_rates[index] < self._r_max and index not in administered_items
    ]

    if len(valid_indexes_low_r) > 0:
      # return the item with maximum information from the ones available
      inf_values = irt.inf_hpc(est_theta, item_bank.get_items(valid_indexes_low_r))
      selected_item = valid_indexes_low_r[numpy.nonzero(inf_values == max(inf_values))[0][0]]

    # if all items in the selected cluster have exceed their r values,
    # select the one with smallest r, regardless of information
    elif self._r_control == "passive":
      inf_values = irt.inf_hpc(est_theta, item_bank.get_items(valid_indexes))
      selected_item = valid_indexes[numpy.nonzero(inf_values == max(inf_values))[0][0]]
    else:
      selected_item = valid_indexes[item_bank.exposure_rates.index(min(item_bank.exposure_rates))]

    return selected_item

  @staticmethod
  def sum_cluster_infos(theta: float, item_bank: ItemBank, clusters: list[int]) -> numpy.ndarray:
    r"""Return the sum of item information values, separated by cluster.

    Parameters
    ----------
    theta : float
        An examinee's :math:`\theta` value.
    item_bank : ItemBank
        An ItemBank containing item parameters.
    clusters : list[int]
        A list containing item cluster memberships, represented by integers.

    Returns
    -------
    numpy.ndarray
        Array containing the sum of item information values for each cluster.
    """
    cluster_infos = numpy.zeros(len(set(clusters)))

    for cluster in set(clusters):
      cluster_indexes = numpy.nonzero([c == cluster for c in clusters])[0]

      for item in item_bank.get_items(cluster_indexes):
        cluster_infos[cluster] += irt.inf(theta, *item[:4])

    return cluster_infos

  @staticmethod
  def weighted_cluster_infos(theta: float, item_bank: ItemBank, clusters: list[int]) -> numpy.ndarray:
    r"""Return the weighted sum of item information values, separated by cluster.

    The weight is the number of items in each cluster, providing an average information
    value per cluster that accounts for cluster size.

    Parameters
    ----------
    theta : float
        An examinee's :math:`\theta` value.
    item_bank : ItemBank
        An ItemBank containing item parameters.
    clusters : list[int]
        A list containing item cluster memberships, represented by integers.

    Returns
    -------
    numpy.ndarray
        Array containing the average information values for each cluster (sum of
        item information divided by the number of items in each cluster).
    """
    cluster_infos = ClusterSelector.sum_cluster_infos(theta, item_bank, clusters)
    count = numpy.bincount(clusters)

    for i in range(len(cluster_infos)):
      cluster_infos[i] /= count[i]

    return cluster_infos

  @staticmethod
  def sum_cluster_params(item_bank: ItemBank, c: list[int]) -> NDArray[numpy.float64]:
    """Return the sum of item parameter values for each cluster.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    c : list[int]
        A list containing clustering memberships (cluster IDs for each item).

    Returns
    -------
    numpy.ndarray
        A matrix containing the sum of each parameter by cluster. Rows are clusters,
        columns are parameters (a, b, c, d).
    """
    averages = numpy.zeros((numpy.max(c) + 1, 4))

    for i in numpy.arange(0, numpy.size(c)):
      if c[i] == -1:
        continue
      averages[c[i], 0] += item_bank.discrimination[i]
      averages[c[i], 1] += item_bank.difficulty[i]
      averages[c[i], 2] += item_bank.pseudo_guessing[i]
      averages[c[i], 3] += item_bank.upper_asymptote[i]

    return averages

  @staticmethod
  def avg_cluster_params(item_bank: ItemBank, c: list[int]) -> NDArray[numpy.float64]:
    """Return the average values of item parameters by cluster.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    c : list[int]
        A list containing clustering memberships (cluster IDs for each item).

    Returns
    -------
    numpy.ndarray
        A matrix containing the average values of each parameter by cluster. Rows are
        clusters, columns are parameters (a, b, c, d).
    """
    averages = ClusterSelector.sum_cluster_params(item_bank, c)

    c_array = numpy.array(c)
    occurrences = numpy.bincount(numpy.delete(c_array, numpy.where(c_array == -1)).astype(numpy.int64))

    for counter, i in enumerate(occurrences):
      averages[counter, 0] /= i
      averages[counter, 1] /= i
      averages[counter, 2] /= i
      averages[counter, 3] /= i

    return averages


class StratifiedSelector(FiniteSelector):
  """Abstract class for stratified finite item selection strategies.

  Stratified selectors divide the item bank into strata and select items from different
  strata as the test progresses, helping to balance item exposure and test characteristics.

  Parameters
  ----------
  test_size : int
      Number of items in the test.
  sort_once : bool
      Whether the strategy allows for the item matrix to be presorted once at the
      beginning of the simulation (True) or requires resorting during the test (False).
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "General Stratified Selector"

  def __init__(self, test_size: int, sort_once: bool) -> None:
    """Initialize a StratifiedSelector.

    Parameters
    ----------
    test_size : int
        Number of items in the test.
    sort_once : bool
        Whether the strategy allows for the item matrix to be presorted.
    """
    super().__init__(test_size)
    self._sort_once = sort_once
    self._presorted_items = None

  @abstractmethod
  def presort_items(self, item_bank: ItemBank) -> numpy.ndarray:
    """Presort the item matrix according to the strategy employed by this selector.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.

    Returns
    -------
    numpy.ndarray
        Array of item indices sorted according to the strategy.
    """

  def postsort_items(
    self,
    item_bank: ItemBank,
    using_simulator_props: bool,
    **kwargs: Any,  # noqa: ARG002
  ) -> numpy.ndarray:
    """Sort the item matrix before selecting each new item.

    This default implementation simply returns the presorted items, or sorts them using
    the :py:func:`presort_items` method and returns them.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    using_simulator_props : bool
        Whether the selector is being executed inside a Simulator.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    -------
    numpy.ndarray
        Array of item indices sorted according to the strategy.
    """
    if using_simulator_props:
      return self._presorted_items
    return self.presort_items(item_bank)

  def preprocess(self) -> None:  # noqa: D102
    self._presorted_items = self.presort_items(self.simulator.item_bank)

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    item_bank : ItemBank or None, optional
        An ItemBank containing item parameters. Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more strata to get items from.
    """
    item_bank, administered_items, est_theta = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      **kwargs,
    )

    assert item_bank is not None
    assert administered_items is not None
    assert est_theta is not None

    # divide the item matrix into strata and get the stratum in which the examinee is
    stratum_index = len(administered_items)
    try:
      slices, pointer, max_pointer = self._get_stratum(item_bank, stratum_index)
    except IndexError as ierr:
      msg = (
        f"{self}: test size is larger than was informed to the selector\n"
        f"Length of administered items:\t{len(administered_items)}\n"
        f"Total length of the test:\t{self._test_size}\n"
        f"Number of slices:\t{len(slices)}"
      )
      raise RuntimeError(msg) from ierr

    using_simulator_props = index is not None

    if using_simulator_props and self._sort_once:
      # if running through a simulator and the selector allows presorting, get the presorted item matrix
      sorted_items = self._presorted_items
    else:
      # allow the selector to resort the item matrix at this point in the test
      kwargs["using_simulator_props"] = using_simulator_props
      sorted_items = self.postsort_items(item_bank, using_simulator_props, est_theta=est_theta)

    # if the selected item has already been administered, select the next one
    while sorted_items[pointer] in administered_items:
      pointer += 1
      if pointer == max_pointer:
        msg = f"There are no more items to be selected from stratum {slices[len(administered_items)]}"
        raise RuntimeError(msg)

    return sorted_items[pointer]

  def _get_stratum(self, item_bank: ItemBank, stratum_index: int) -> numpy.ndarray:
    slices = numpy.linspace(0, item_bank.n_items, self._test_size, endpoint=False, dtype="i")
    pointer = slices[stratum_index]
    max_pointer = item_bank.n_items if stratum_index == self._test_size - 1 else slices[stratum_index + 1]

    return slices, pointer, max_pointer


class AStratSelector(StratifiedSelector):
  r"""Implementation of the :math:`\alpha`-stratified selector proposed by [Chang99]_.

  In this selector, the item bank is sorted in ascending order according to the items'
  discrimination parameter and then separated into :math:`K` strata (:math:`K` being the
  test size), each stratum containing gradually higher average discrimination. The
  :math:`\alpha`-stratified selector then selects the first non-administered item from
  stratum :math:`k`, where :math:`k` represents the position in the test of the current
  item the examinee is being presented.

  This method helps control item exposure by ensuring items with different discrimination
  levels are distributed throughout the test.

  .. image:: ../sphinx/alpha-strat.*

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to create
      the correct number of strata.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "a-Stratified Selector"

  def __init__(self, test_size: int) -> None:
    """Initialize an AStratSelector object.

    Parameters
    ----------
    test_size : int
        Number of items the test contains.
    """
    super().__init__(test_size, True)

  def presort_items(self, item_bank: ItemBank) -> numpy.ndarray:  # noqa: PLR6301
    """Presort the item matrix in ascending order according to the discrimination of each item.

    Parameters
    ----------
    items : numpy.ndarray
        An item matrix.

    Returns
    -------
    numpy.ndarray
        Array of item indices sorted in ascending order by discrimination (a parameter).
    """
    return item_bank.discrimination.argsort()


class AStratBBlockSelector(StratifiedSelector):
  r"""Implementation of the :math:`\\alpha`-stratified selector with :math:`b` blocking proposed by [Chang2001]_.

  In this selector, the item bank is sorted in ascending order according to the items difficulty parameter and then
  separated into :math:`M` strata, each stratum containing gradually higher average difficulty.

  Each of the :math:`M` strata is then again separated into :math:`K` sub-strata (:math:`k` being the test size),
  according to their discrimination. The final item bank is then ordered such that the first sub-strata of each strata
  forms the first strata of the new ordered item bank, and so on. This method tries to balance the distribution of both
  parameters between all strata, after perceiving that they are correlated.

  .. image:: ../sphinx/b-blocking.*

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to
      create the correct number of strata.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "a-Stratified b-Blocking Selector"

  def __init__(self, test_size: int) -> None:
    """Initialize a AStratBBlockSelector object.

    Parameters
    ----------
    test_size : int
        Number of items the test contains.
    """
    super().__init__(test_size, True)

  def presort_items(self, item_bank: ItemBank) -> numpy.ndarray:
    """Presort items in ascending order of discrimination each item, then each strata according to item difficulty.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.

    Returns
    -------
    numpy.ndarray
        The sorted item matrix.
    """
    # sort items by their b values, in ascending order
    presorted_items = item_bank.difficulty.argsort()

    final_indices = []
    for stratum_index in range(self._test_size):
      _slices, pointer, max_pointer = self._get_stratum(item_bank, stratum_index)
      indices_current_stratum = presorted_items[pointer:max_pointer]
      items_current_stratum = item_bank.get_items(indices_current_stratum)
      sorted_indices_current_stratum = items_current_stratum[:, 0].argsort()
      # sort the items in the current stratum by their discrimination values, in ascending order
      global_sorted_indices_current_stratum = indices_current_stratum[sorted_indices_current_stratum]
      final_indices.extend(global_sorted_indices_current_stratum)

    # sort the item bank first by the items maximum information, ascending
    # then by their information to the examinee's cuirrent theta, descending
    return numpy.array(final_indices)


class MaxInfoStratSelector(StratifiedSelector):
  """Implementation of the maximum information stratification (MIS) selector proposed by [Bar06]_.

  In this selector, the item bank is sorted in ascending order according to the items'
  maximum information and then separated into :math:`K` strata (:math:`K` being the test
  size), each stratum containing items with gradually higher maximum information. The MIS
  selector then selects the first non-administered item from stratum :math:`k`, where
  :math:`k` represents the position in the test of the current item the examinee is being
  presented.

  .. image:: ../sphinx/mis.*

  This method claims to work better than the :math:`a`-stratified method by [Chang99]_ for
  the three-parameter logistic model of IRT, since item difficulty and maximum information
  are not positioned in the same place in the ability scale in 3PL.

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to create
      the correct number of strata.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Maximum Information Stratification Selector"

  def __init__(self, test_size: int) -> None:
    """Initialize a MaxInfoStratSelector object.

    Parameters
    ----------
    test_size : int
        Number of items the test contains.
    """
    super().__init__(test_size, False)

  def presort_items(self, item_bank: ItemBank) -> numpy.ndarray:  # noqa: PLR6301
    """Presort items in ascending order of maximum information.

    Parameters
    ----------
    items : numpy.ndarray
        An item matrix.

    Returns
    -------
    numpy.ndarray
        The sorted item matrix.
    """
    # Use cached max info values from ItemBank - this is a key optimization!
    # get the theta values in which items are maximally informative
    # get the information values for all items at their maximum points
    item_maxinfo = item_bank.max_info_values  # Using cached values!
    # globally sort item bank by item max information
    return item_maxinfo.argsort()

  def postsort_items(self, item_bank: ItemBank, using_simulator_props: bool, est_theta: float) -> numpy.ndarray:
    """Divide the item bank into strata and sort each one in descending order of information for the current theta.

    Parameters
    ----------
    items : numpy.ndarray
        The item matrix.
    using_simulator_props : bool
        Whether the selector is being executed inside a Simulator.
    est_theta : float
        The current estimate of the examinee's ability.

    Returns
    -------
    numpy.ndarray
        The sorted item matrix.
    """
    # recover items presorted by the first rule
    presorted_items = self._presorted_items if using_simulator_props else self.presort_items(item_bank)
    # run through each stratum and sort items in descending order according to
    # their information for the current theta value
    final_indices = []
    for stratum_index in range(self._test_size):
      # grab stratum pointers
      _slices, pointer, max_pointer = self._get_stratum(item_bank, stratum_index)
      item_indices_current_stratum = presorted_items[pointer:max_pointer]  # item indices for the current stratum
      items_current_stratum: numpy.ndarray = item_bank.get_items(
        item_indices_current_stratum
      )  # item params for the current stratum
      # their information for this theta
      info_items_current_stratum_current_theta: numpy.ndarray = irt.inf_hpc(est_theta, items_current_stratum)
      item_indices_current_stratum_sorted_by_info = item_indices_current_stratum[
        (-info_items_current_stratum_current_theta).argsort()
      ]
      final_indices.extend(item_indices_current_stratum_sorted_by_info)

    # sort the item bank first by the items maximum information, ascending
    # then by their information to the examinee's cuirrent theta, descending
    return numpy.array(final_indices)


class MaxInfoBBlockSelector(MaxInfoStratSelector):
  """Implementation of the maximum information stratification with :math:`b` blocking (MIS-B) selector [Bar06]_.

  In this selector, the item bank is sorted in ascending order according to the items difficulty parameter and then
  separated into :math:`M` strata, each stratum containing gradually higher average difficulty.

  Each of the :math:`M` strata is then again separated into :math:`K` sub-strata (:math:`k` being the test size),
  according to the items maximum information. The final item bank is then ordered such that the first sub-strata of each
  strata forms the first strata of the new ordered item bank, and so on. This method tries to balance the distribution
  of both parameters between all strata and works better than the :math:`a`-stratified with :math:`b` blocking method by
  [Chang2001]_ for the three-parameter logistic model of IRT, since item difficulty and maximum information are not
  positioned in the same place in the ability scale in 3PL. This may also apply, although not mentioned by the authors,
  for the 4PL.

  .. image:: ../sphinx/mis-b.*

  Parameters
  ----------
  test_size : int
      The number of items the test contains. The selector uses this parameter to
      create the correct number of strata.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Maximum Information Stratification with b-Blocking Selector"

  def presort_items(self, item_bank: ItemBank) -> numpy.ndarray:
    """Presort the item matrix according to the information of each item at their maximum.

    Parameters
    ----------
    items : numpy.ndarray
        An item matrix.

    Returns
    -------
    numpy.ndarray
        The sorted item matrix.
    """
    # Use cached max info values from ItemBank - key optimization!
    # get the theta values in which items are maximally informative
    theta_maxinfo = item_bank.max_info_thetas  # Using cached values!
    # sort items by theta
    presorted_items = theta_maxinfo.argsort()
    # get the information values for all items at their maximum points
    item_maxinfo = item_bank.max_info_values  # Using cached values!

    final_indices = []
    for stratum_index in range(self._test_size):
      _slices, pointer, max_pointer = self._get_stratum(item_bank, stratum_index)
      indices_current_stratum = presorted_items[pointer:max_pointer]
      # sort items in the current stratum by maximum information, in ascending order
      sorted_indices_current_stratum = item_maxinfo[indices_current_stratum].argsort()
      global_sorted_indices_current_stratum = indices_current_stratum[sorted_indices_current_stratum]
      final_indices.extend(global_sorted_indices_current_stratum)

    # sanity check to make sure all indices are present and unique
    assert len(final_indices) == len(set(final_indices))
    return numpy.array(final_indices)


class The54321Selector(FiniteSelector):
  r"""Implementation of the 5-4-3-2-1 selector proposed by [McBride83]_.

  In this selector, at each step :math:`k` of a test of size :math:`K`, an item is chosen from a bin containing the
  :math:`K-k` most informative items in the bank, given the current :math:`\\hat\\theta`. As the test progresses, the
  bin gets smaller and more informative items have a higher probability of being chosen by the end of the test, when the
  estimation of ':math:`\\hat\\theta` is more precise. The 5-4-3-2-1 selector can be viewed as a specialization of the
  :py:class:`catsim.selection.RandomesqueSelector`, in which the bin size of most informative items gets smaller as the
  test progresses.
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "5-4-3-2-1 Selector"

  def __init__(self, test_size: int) -> None:
    """Initialize a The54321Selector object.

    Parameters
    ----------
    test_size : int
        The number of items the test contains. The selector uses this parameter to set the bin size.
    """
    super().__init__(test_size)

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    est_theta: float | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    items : numpy.ndarray or None, optional
        A matrix containing item parameters in the format that `catsim` understands
        (see: :py:meth:`ItemBank.generate_item_bank`). Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs
        Additional keyword arguments. Notably:

        * **rng** (:py:class:`numpy.random.Generator`) -- Random number generator used by the object,
          guarantees reproducibility of outputs.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    item_bank, administered_items, est_theta, rng = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      return_rng=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      **kwargs,
    )

    assert est_theta is not None
    assert administered_items is not None
    assert item_bank is not None

    # sort item indexes by their information value descending and remove indexes of administered items
    ordered_items = self._sort_by_info(item_bank, est_theta)
    organized_items = self._get_non_administered(ordered_items, administered_items)

    if len(organized_items) == 0:
      msg = "There are no more items to apply."
      raise RuntimeError(msg)

    bin_size = self._test_size - len(administered_items)
    return rng.choice(organized_items[0:bin_size])


class RandomesqueSelector(Selector):
  """Implementation of the randomesque selector proposed by [Kingsbury89]_.

  In this selector, at each step of the test, an item is randomly chosen from the :math:`n` most informative items in
  the item bank, :math:`n` being a predefined value (originally 5, but user-defined in this implementation).
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Randomesque Selector"

  def __init__(self, bin_size: int) -> None:
    """Initialize a RandomesqueSelector object.

    Parameters
    ----------
    bin_size : int
        The number of most informative items to be taken into consideration when
        randomly selecting one of them.
    """
    super().__init__()
    self._bin_size = bin_size

  @property
  def bin_size(self) -> int:
    """Get the bin size."""
    return self._bin_size

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    est_theta: float | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    items : numpy.ndarray or None, optional
        A matrix containing item parameters in the format that `catsim` understands
        (see: :py:meth:`ItemBank.generate_item_bank`). Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs
        Additional keyword arguments. Notably:

        * **rng** (:py:class:`numpy.random.Generator`) -- Random number generator used by the object,
          guarantees reproducibility of outputs.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    item_bank, administered_items, est_theta, rng = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      return_rng=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      **kwargs,
    )

    assert est_theta is not None
    assert administered_items is not None
    assert item_bank is not None

    # sort item indexes by their information value descending and remove indexes of administered items
    ordered_items = self._sort_by_info(item_bank, est_theta)
    organized_items = self._get_non_administered(ordered_items, administered_items)

    if len(organized_items) == 0:
      msg = "There are no more items to apply."
      raise RuntimeError(msg)

    return rng.choice(list(organized_items)[: self._bin_size])


class IntervalInfoSelector(Selector):
  r"""Selects the item that maximizes the integral of the information function at a predetermined ``interval``.

  The interval is defined by a parameter :math:`\\delta` above and below the current :math:`\\hat\\theta`, like so:
  .. math:: argmax_{i \\in I} \\int_{\\hat\\theta - \\delta}^{\\hat\\theta + \\delta}I_i(\\hat\\theta)
  """

  def __str__(self) -> str:
    """Return the name of the selector."""
    return "Interval Information Selector"

  def __init__(self, interval: float | None = None) -> None:
    r"""Initialize an IntervalInfoSelector object.

    Parameters
    ----------
    interval : float or None, optional
        The interval of the integral. If no interval is passed, the integral is computed from
        :math:`[-\\infty, \\infty]`. Default is None.
    """
    super().__init__()
    self._interval = interval if interval is not None else numpy.inf

  @property
  def interval(self) -> float:
    """Get the size of the interval under which the integral of the information function will be computed."""
    return self._interval

  def select(
    self,
    index: int | None = None,
    item_bank: ItemBank | None = None,
    administered_items: list[int] | None = None,
    est_theta: float | None = None,
    **kwargs: Any,
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    index : int or None, optional
        The index of the current examinee in the simulator. Default is None.
    items : numpy.ndarray or None, optional
        A matrix containing item parameters in the format that `catsim` understands
        (see: :py:meth:`ItemBank.generate_item_bank`). Default is None.
    administered_items : list[int] or None, optional
        A list containing the indexes of items that were already administered. Default is None.
    est_theta : float or None, optional
        A float containing the current estimated ability. Default is None.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    int or None
        Index of the next item to be applied or `None` if there are no more items in the item bank.
    """
    item_bank, administered_items, est_theta = self._prepare_args(
      return_item_bank=True,
      return_administered_items=True,
      return_est_theta=True,
      index=index,
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      **kwargs,
    )

    assert est_theta is not None
    assert administered_items is not None
    assert item_bank is not None

    # compute the integral of the information function around an examinee's ability
    information_integral = numpy.array([
      quad(
        irt.inf,
        est_theta - self._interval,
        est_theta + self._interval,
        args=(item[0], item[1], item[2], item[3]),
      )[0]
      for item in item_bank.items
    ])
    # sort by that integral in descending order
    ordered_items = (-information_integral).argsort()
    # remove administered items
    organized_items = self._get_non_administered(ordered_items, administered_items)

    if len(organized_items) == 0:
      msg = "There are no more items to apply."
      raise RuntimeError(msg)

    return organized_items[0]
