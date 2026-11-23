import operator

import numpy

from .. import irt
from .._typing import FloatArray
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector


class ClusterSelector(BaseSelector):
  """Cluster-based Item Selection Method.

  Parameters
  ----------
  clusters : list[int]
      A list containing item cluster memberships (integers representing cluster IDs).
  method : str, optional
      One of the available methods for cluster selection. Given the estimated theta
      value at each step:

      - 'item_info': selects the cluster which has the item with maximum information
      - 'cluster_info': selects the cluster whose items' sum of information is maximum
      - 'weighted_info': selects the cluster whose average item information is
        maximum (its sum of information divided by its number of items)

      Every score counts all the items in a cluster, including those already
      administered, as long as the cluster still has an item left. Default is
      'item_info'.
  r_max : float, optional
      Exposure-rate cap for choosing an item inside the selected cluster: items at
      or above it are passed over unless no other item in the cluster remains. It
      does not affect which cluster is chosen. Default is 1 (no cap).
  r_control : str, optional
      Item exposure control method, used when every remaining item in the selected
      cluster has an exposure rate at or above r_max. 'passive' applies the most
      informative of them; 'aggressive' applies the one with the smallest exposure
      rate. Default is 'passive'.

  Notes
  -----
  For full algorithmic details, see :doc:`/api/techniques/selection/cluster-selector`.
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
        Exposure-rate cap for choosing an item inside the selected cluster. Default is 1.
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
    item_bank: ItemBank,
    administered_items: list[int],
    est_theta: float,
    rng: numpy.random.Generator | None = None,
    exposure_rates: FloatArray | None = None,
    response_vector: list[bool] | None = None,  # noqa: ARG002
  ) -> int | None:
    """Return the index of the next item to be administered.

    Parameters
    ----------
    item_bank : ItemBank
        An ItemBank containing item parameters.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator, used only to break ties between equally good items in the chosen
        cluster.
    exposure_rates : numpy.ndarray or None, optional
        Run-level exposure-rate snapshot. ``None`` treats every item as unexposed.

    Returns
    -------
    int
        Index of the next item to be applied. Ties within the chosen cluster are broken uniformly at
        random (ADR-0008).

    Raises
    ------
    NoItemsAvailableError
        If every item has already been administered.
    ValueError
        If items in the chosen cluster tie and ``rng`` is None.
    """
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

          # if all clusters have been evaluated, the loop ends and the method raises NoItemsAvailableError below
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

      sorted_clusters = [
        cluster for cluster, _info in sorted(cluster_infos.items(), key=operator.itemgetter(1), reverse=True)
      ]

      # walks through the sorted clusters in order
      for cluster in sorted_clusters:
        valid_indexes = [
          idx
          for idx, item_cluster in enumerate(self._clusters)
          if item_cluster == cluster and idx not in administered_items
        ]

        # checks if at least one item from this cluster has not
        # been administered to this examinee yet
        if valid_indexes:
          selected_cluster = cluster
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

    # gets the indexes of the items in the selected cluster that have not been administered
    valid_indexes = self._get_non_administered(
      numpy.nonzero([cluster == selected_cluster for cluster in self._clusters])[0],
      administered_items,
    )

    # of those, gets the indexes of the items whose exposure rate is below r_max
    if exposure_rates is None:
      exposure_rates = numpy.zeros(item_bank.n_items, dtype=float)
    valid_indexes_low_r = [
      idx for idx in valid_indexes if exposure_rates[idx] < self._r_max and idx not in administered_items
    ]

    # each of the picks below breaks ties uniformly at random (ADR-0008)
    if len(valid_indexes_low_r) > 0:
      # the most informative of the items below r_max
      candidates = numpy.asarray(valid_indexes_low_r, dtype=numpy.intp)
      return self._pick_best(candidates, irt.inf_hpc(est_theta, item_bank.get_items(candidates)), rng)

    # every remaining item in the selected cluster is at or above r_max
    candidates = numpy.asarray(valid_indexes, dtype=numpy.intp)
    if self._r_control == "passive":
      # passive control ignores the cap: the most informative item in the cluster
      return self._pick_best(candidates, irt.inf_hpc(est_theta, item_bank.get_items(candidates)), rng)
    # aggressive control: the item with the smallest exposure rate, regardless of information
    return self._pick_best(candidates, -exposure_rates[candidates], rng)

  @staticmethod
  def sum_cluster_infos(theta: float, item_bank: ItemBank, clusters: list[int]) -> dict[int, float]:
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
    dict[int, float]
        The sum of item information values in each cluster, keyed by cluster ID.
    """
    cluster_infos: dict[int, float] = {}
    for cluster in sorted(set(clusters)):
      cluster_indexes = numpy.nonzero([c == cluster for c in clusters])[0]
      cluster_infos[cluster] = float(sum(irt.inf(theta, *item[:4]) for item in item_bank.get_items(cluster_indexes)))

    return cluster_infos

  @staticmethod
  def weighted_cluster_infos(theta: float, item_bank: ItemBank, clusters: list[int]) -> dict[int, float]:
    r"""Return the average item information in each cluster.

    Each cluster's sum of item information is divided by its number of items, so a
    cluster does not score higher only because it holds more items.

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
    dict[int, float]
        The average item information in each cluster (the sum of item information
        divided by the number of items in the cluster), keyed by cluster ID.
    """
    cluster_infos = ClusterSelector.sum_cluster_infos(theta, item_bank, clusters)
    count = numpy.bincount(clusters)

    for cluster in list(cluster_infos):
      cluster_infos[cluster] /= count[cluster]

    return cluster_infos
