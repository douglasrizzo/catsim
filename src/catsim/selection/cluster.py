"""Cluster-based selector implementation."""

import operator
from typing import Any

import numpy
from numpy.typing import NDArray

from .. import irt
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector


class ClusterSelector(BaseSelector):
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
    item_bank : ItemBank or None, optional
        An ItemBank containing item parameters. Default is None.
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

      evaluated_clusters: set[int] = set()

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
      raise NoItemsAvailableError(msg)

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
    valid_indexes_low_r = [
      idx for idx in valid_indexes if item_bank.exposure_rates[idx] < self._r_max and idx not in administered_items
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
      # Find the item with minimum exposure rate
      exposure_rates = [item_bank.exposure_rates[idx] for idx in valid_indexes]
      min_rate_idx = exposure_rates.index(min(exposure_rates))
      selected_item = valid_indexes[min_rate_idx]

    return selected_item

  @staticmethod
  def sum_cluster_infos(theta: float, item_bank: ItemBank, clusters: list[int]) -> NDArray[numpy.floating]:
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
  def weighted_cluster_infos(theta: float, item_bank: ItemBank, clusters: list[int]) -> NDArray[numpy.floating]:
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
