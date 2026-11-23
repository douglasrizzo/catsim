import numpy

from .._typing import FloatArray, IntArray
from ..exceptions import NoItemsAvailableError
from ..item_bank import ItemBank
from .base import BaseSelector


class ClusterSelector(BaseSelector):
  """Cluster-based Item Selection Method.

  Parameters
  ----------
  clusters : list[int]
      One non-negative integer cluster label per item of the bank the selector is used with. The labels
      are checked against the bank's size at every selection.
  method : str, optional
      One of the available methods for cluster selection. Given the estimated theta
      value at each step:

      - 'item_info': selects the cluster which has the item with maximum information
      - 'cluster_info': selects the cluster whose items' sum of information is maximum
      - 'weighted_info': selects the cluster whose average item information is
        maximum (its sum of information divided by its number of items)

      Every score counts all the items in a cluster, including those already
      administered, as long as the cluster still has an item left. Clusters whose
      scores tie exactly are chosen between uniformly at random (ADR-0008). Default is
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
    """Return the cluster label of each item."""
    return self._clusters.tolist()

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
        One non-negative integer cluster label per item.
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
    if self._r_control == "passive":
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
    if self._method == "item_info":
      best = numpy.full(self._clusters.max() + 1, -numpy.inf)
      numpy.maximum.at(best, self._clusters, item_bank.information(theta))
      return best[open_clusters]
    scores = (
      self.sum_cluster_infos(theta, item_bank, self._clusters)
      if self._method == "cluster_info"
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
