"""Cluster-based item selection (Meneghetti, 2015).

:class:`ClusterSelector` chooses a cluster of items before it chooses an item. :class:`ClusterMethod`
names the ways it can score a cluster, and :class:`ExposureControl` what it administers when the chosen
cluster has no item below the exposure cap.
"""

from enum import StrEnum

import numpy

from .._typing import FloatArray, IntArray
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector


class ClusterMethod(StrEnum):
  """How :class:`ClusterSelector` scores a cluster at the current ability estimate.

  Every score counts all the items of a cluster, including those already administered.

  Attributes
  ----------
  ITEM_INFO : str
      The information of the cluster's most informative item.
  CLUSTER_INFO : str
      The sum of the information of the cluster's items.
  WEIGHTED_INFO : str
      The mean information of the cluster's items: their sum divided by the cluster's size.
  """

  ITEM_INFO = "item_info"
  CLUSTER_INFO = "cluster_info"
  WEIGHTED_INFO = "weighted_info"


class ExposureControl(StrEnum):
  """What :class:`ClusterSelector` administers when no remaining item of the chosen cluster is below ``r_max``.

  Attributes
  ----------
  PASSIVE : str
      The most informative remaining item of the cluster, as if there were no cap.
  AGGRESSIVE : str
      The remaining item of the cluster with the smallest exposure rate, regardless of information.
  """

  PASSIVE = "passive"
  AGGRESSIVE = "aggressive"


class ClusterSelector(BaseSelector):
  """Choose the best-scoring cluster of items, then the most informative remaining item in it.

  This is the cluster-based selection method of Meneghetti (2015). The caller labels each item with
  a cluster, for example by clustering the item parameters. At each step the selector scores every
  cluster that still has an item left, takes the best one, and administers that cluster's most
  informative remaining item whose exposure rate is below ``r_max``. Exact ties between clusters and
  between items are broken uniformly at random (ADR-0008).

  Parameters
  ----------
  clusters : list[int]
      One non-negative integer cluster label per item of the bank the selector is used with. The labels
      are checked against the bank's size at every selection.
  method : ClusterMethod or str, optional
      How clusters are scored; see :class:`ClusterMethod`. Every score counts all the items in a cluster,
      including those already administered. Default is ``ClusterMethod.ITEM_INFO``.
  r_max : float, optional
      Exposure-rate cap for choosing an item inside the selected cluster: items at or above it are
      passed over unless no other item in the cluster remains. It does not affect which cluster is
      chosen. Default is 1 (no cap).
  r_control : ExposureControl or str, optional
      What to administer when every remaining item in the selected cluster is at or above ``r_max``; see
      :class:`ExposureControl`. Default is ``ExposureControl.PASSIVE``.

  Notes
  -----
  The method has known limits. A cluster keeps the score of items already administered, so
  ``item_info`` can return to a cluster whose remaining items are much less informative than items
  elsewhere in the bank; in simulations it gave less test information and more test overlap than
  :class:`~catsim.selection.MaxInfoSelector`, and no scoring rule improved on
  :class:`~catsim.selection.MaxInfoSelector` or :class:`~catsim.selection.RandomesqueSelector`.
  Because ``r_max`` applies only inside the chosen cluster, items can end above the cap.

  For full algorithmic details and the simulation results, see
  :doc:`/api/techniques/selection/cluster-selector`.
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
    """Return the cluster label of each item."""
    return self._clusters.tolist()

  @property
  def method(self) -> ClusterMethod:
    """Return the method used for cluster selection."""
    return self._method

  @property
  def r_control(self) -> ExposureControl:
    """Return the item exposure control method."""
    return self._r_control

  def __init__(
    self,
    clusters: list[int],
    method: ClusterMethod | str = ClusterMethod.ITEM_INFO,
    r_max: float = 1,
    r_control: ExposureControl | str = ExposureControl.PASSIVE,
  ) -> None:
    """Initialize a ClusterSelector object.

    Parameters
    ----------
    clusters : list[int]
        One non-negative integer cluster label per item.
    method : ClusterMethod or str, optional
        Cluster-scoring method. Default is ``ClusterMethod.ITEM_INFO``.
    r_max : float, optional
        Exposure-rate cap for choosing an item inside the selected cluster. Default is 1.
    r_control : ExposureControl or str, optional
        Item exposure control method. Default is ``ExposureControl.PASSIVE``.

    Raises
    ------
    ValueError
        If parameters are invalid (empty clusters list, negative or non-integer cluster labels, invalid
        method or r_control, or r_max not between 0 and 1).
    """
    labels = numpy.array(clusters)
    if labels.size == 0:
      msg = "clusters list cannot be empty"
      raise ValueError(msg)
    if not numpy.issubdtype(labels.dtype, numpy.integer) or (labels < 0).any():
      msg = "All cluster values must be non-negative integers"
      raise ValueError(msg)

    try:
      method = ClusterMethod(method)
    except ValueError:
      msg = f"{method} is not a valid cluster selection method; choose one from {list(map(str, ClusterMethod))}"
      raise ValueError(msg) from None

    if not 0 <= r_max <= 1:
      msg = f"r_max must be between 0 and 1, got {r_max}"
      raise ValueError(msg)

    try:
      r_control = ExposureControl(r_control)
    except ValueError:
      msg = (
        f"{r_control} is not a valid item exposure control method; choose one from {list(map(str, ExposureControl))}"
      )
      raise ValueError(msg) from None

    super().__init__()
    self._clusters: IntArray = labels
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
        An ItemBank containing item parameters, one per cluster label.
    administered_items : list[int]
        A list containing the indexes of items that were already administered.
    est_theta : float
        The current estimated ability.
    rng : numpy.random.Generator or None, optional
        Random number generator, used only to break exact ties between clusters and between items in the
        chosen cluster.
    exposure_rates : numpy.ndarray or None, optional
        Run-level exposure-rate snapshot. ``None`` treats every item as unexposed.
    response_vector : list[bool] or None, optional
        Responses observed so far. Not used by this selector.

    Returns
    -------
    int
        Index of the next item to be applied.

    Raises
    ------
    ValueError
        If ``clusters`` does not hold one label per item of ``item_bank``, or if clusters or items in the
        chosen cluster tie and ``rng`` is None.
    NoItemsAvailableError
        If every item has already been administered.
    """
    if self._clusters.size != item_bank.n_items:
      msg = f"clusters holds {self._clusters.size} labels, but the item bank has {item_bank.n_items} items"
      raise ValueError(msg)

    remaining = numpy.ones(item_bank.n_items, dtype=bool)
    remaining[administered_items] = False
    open_clusters = numpy.unique(self._clusters[remaining])
    if open_clusters.size == 0:
      msg = "There are no more items to be applied."
      raise NoItemsAvailableError(msg)

    cluster = self._pick_best(open_clusters, self._score(est_theta, item_bank, open_clusters), rng)

    in_cluster = numpy.flatnonzero(remaining & (self._clusters == cluster))
    information = item_bank.information(est_theta)
    if exposure_rates is None:
      exposure_rates = numpy.zeros(item_bank.n_items)
    below_cap = in_cluster[exposure_rates[in_cluster] < self._r_max]
    if below_cap.size > 0:
      return self._pick_best(below_cap, information[below_cap], rng)
    if self._r_control is ExposureControl.PASSIVE:
      return self._pick_best(in_cluster, information[in_cluster], rng)
    return self._pick_best(in_cluster, -exposure_rates[in_cluster], rng)

  def _score(self, theta: float, item_bank: ItemBank, open_clusters: IntArray) -> FloatArray:
    """Return the score of each cluster in ``open_clusters`` under the selector's method.

    Parameters
    ----------
    theta : float
        The current estimated ability.
    item_bank : ItemBank
        The bank the cluster labels describe.
    open_clusters : numpy.ndarray
        Labels of the clusters to score.

    Returns
    -------
    numpy.ndarray
        One score per cluster in ``open_clusters``.
    """
    if self._method is ClusterMethod.ITEM_INFO:
      best = numpy.full(self._clusters.max() + 1, -numpy.inf)
      numpy.maximum.at(best, self._clusters, item_bank.information(theta))
      return best[open_clusters]
    scores = (
      self.sum_cluster_infos(theta, item_bank, self._clusters)
      if self._method is ClusterMethod.CLUSTER_INFO
      else self.weighted_cluster_infos(theta, item_bank, self._clusters)
    )
    return numpy.array([scores[label] for label in open_clusters.tolist()])

  @staticmethod
  def sum_cluster_infos(theta: float, item_bank: ItemBank, clusters: list[int] | IntArray) -> dict[int, float]:
    r"""Return the sum of item information values, separated by cluster.

    Parameters
    ----------
    theta : float
        An examinee's :math:`\theta` value.
    item_bank : ItemBank
        An ItemBank containing item parameters.
    clusters : list[int] or numpy.ndarray
        One non-negative integer cluster label per item of ``item_bank``.

    Returns
    -------
    dict[int, float]
        The sum of item information values in each cluster, keyed by cluster label.
    """
    sums = numpy.bincount(clusters, weights=item_bank.information(theta))
    return {label: float(sums[label]) for label in numpy.unique(clusters).tolist()}

  @staticmethod
  def weighted_cluster_infos(theta: float, item_bank: ItemBank, clusters: list[int] | IntArray) -> dict[int, float]:
    r"""Return the average item information in each cluster.

    Each cluster's sum of item information is divided by its number of items, so a
    cluster does not score higher only because it holds more items.

    Parameters
    ----------
    theta : float
        An examinee's :math:`\theta` value.
    item_bank : ItemBank
        An ItemBank containing item parameters.
    clusters : list[int] or numpy.ndarray
        One non-negative integer cluster label per item of ``item_bank``.

    Returns
    -------
    dict[int, float]
        The average item information in each cluster (the sum of item information
        divided by the number of items in the cluster), keyed by cluster label.
    """
    sizes = numpy.bincount(clusters)
    return {
      label: total / int(sizes[label])
      for label, total in ClusterSelector.sum_cluster_infos(theta, item_bank, clusters).items()
    }
