"""Cluster-based item selection (Meneghetti, 2015).

:class:`ClusterSelector` chooses a cluster of items before it chooses an item. :class:`ClusterMethod`
names the ways it can score a cluster, and :class:`CapFallback` what it administers when the chosen
cluster has no item below the exposure cap.
"""

from enum import StrEnum
from typing import override

import numpy

from ..context import StepContext
from ..item_bank import ItemBank
from ..typing import FloatArray, IntArray
from .base import BaseSelector, below_cap, check_r_max, pick_best, remaining


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


class CapFallback(StrEnum):
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

  This is the cluster-based selection method of :cite:t:`Meneghetti2015`. The caller labels each item with
  a cluster, for example by clustering the item parameters. At each step the selector scores every
  cluster that still has an item left, takes the best one, and administers that cluster's most
  informative remaining item whose exposure rate is below ``r_max``. Exact ties between clusters and
  between items are broken uniformly at random.

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
  r_control : CapFallback or str, optional
      What to administer when every remaining item in the selected cluster is at or above ``r_max``; see
      :class:`CapFallback`. Default is ``CapFallback.PASSIVE``.

  Raises
  ------
  ValueError
      If parameters are invalid (empty clusters list, negative or non-integer cluster labels, invalid
      method or r_control, or r_max not in (0, 1]).

  Notes
  -----
  Because ``r_max`` applies only inside the chosen cluster, items can end above the cap. For the
  method, its known limits and its simulation results, see
  :doc:`/api/techniques/selection/cluster-selector`.
  """

  @override
  def __repr__(self) -> str:
    """Return the selector as a constructor call that names every argument."""
    return (
      f"{type(self).__name__}(clusters={self.clusters!r}, method=ClusterMethod.{self._method.name}, "
      f"r_max={self._r_max!r}, r_control=CapFallback.{self._r_control.name})"
    )

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
  def r_control(self) -> CapFallback:
    """Return what the selector administers when no remaining item of the chosen cluster is below ``r_max``."""
    return self._r_control

  def __init__(
    self,
    clusters: list[int],
    method: ClusterMethod | str = ClusterMethod.ITEM_INFO,
    r_max: float = 1,
    r_control: CapFallback | str = CapFallback.PASSIVE,
  ) -> None:
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

    try:
      r_control = CapFallback(r_control)
    except ValueError:
      msg = f"{r_control} is not a valid cap fallback; choose one from {list(map(str, CapFallback))}"
      raise ValueError(msg) from None

    super().__init__()
    self._clusters: IntArray = labels
    # Scores are indexed by each label's rank among the distinct labels, not by the label itself, so
    # memory does not grow with the largest label (hashed or database IDs).
    self._label_values, self._label_ranks = numpy.unique(labels, return_inverse=True)
    self._method = method
    self._r_max = check_r_max(r_max)
    self._r_control = r_control

  @override
  def select(self, context: StepContext) -> int:
    """Return the most informative available item of the best-scoring cluster that has an item left.

    Parameters
    ----------
    context : StepContext
        The session to select for.

    Returns
    -------
    int
        Index of the selected item.

    Raises
    ------
    ValueError
        If ``clusters`` does not hold one label per item of ``context.item_bank``.
    """
    item_bank, theta, rng, exposure_rates = context.item_bank, context.theta, context.rng, context.exposure_rates
    if self._clusters.size != item_bank.n_items:
      msg = f"clusters holds {self._clusters.size} labels, but the item bank has {item_bank.n_items} items"
      raise ValueError(msg)

    candidates = remaining(context)
    open_clusters = numpy.unique(self._clusters[candidates])
    cluster = pick_best(open_clusters, self._score(theta, item_bank, open_clusters), rng)

    in_cluster = candidates[self._clusters[candidates] == cluster]
    # A cap of 1 is no cap (see below_cap), so the aggressive fallback never applies to it.
    if (
      self._r_control is CapFallback.AGGRESSIVE
      and exposure_rates is not None
      and self._r_max < 1
      and (exposure_rates[in_cluster] >= self._r_max).all()
    ):
      return pick_best(in_cluster, -exposure_rates[in_cluster], rng)
    allowed = below_cap(in_cluster, exposure_rates, self._r_max)
    return pick_best(allowed, item_bank.information(theta)[allowed], rng)

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
      best = numpy.full(self._label_values.size, -numpy.inf)
      numpy.maximum.at(best, self._label_ranks, item_bank.information(theta))
      return best[numpy.searchsorted(self._label_values, open_clusters)]
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
    labels, ranks = numpy.unique(clusters, return_inverse=True)
    sums = numpy.bincount(ranks, weights=item_bank.information(theta))
    return dict(zip(labels.tolist(), sums.tolist(), strict=True))

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
    labels, sizes = numpy.unique(clusters, return_counts=True)
    sums = ClusterSelector.sum_cluster_infos(theta, item_bank, clusters)
    return {label: sums[label] / size for label, size in zip(labels.tolist(), sizes.tolist(), strict=True)}
