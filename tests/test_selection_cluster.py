"""Tests for ClusterSelector."""

import numpy as np
import pytest

from catsim.item_bank import ItemBank
from catsim.selection import ClusterSelector
from tests.banks import two_pl_bank
from tests.contexts import make_context


class TestClusterSelector:
  """Tests for ClusterSelector."""

  # All difficulties are 0, so at theta = 0 an item's information is a^2 / 4. Cluster 0 holds the single most
  # informative item (1), cluster 1 the largest information sum (its best item is 7), and cluster 2 the largest
  # average information (its best item is 13).
  CLUSTERS = (0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2)
  DISCRIMINATIONS = (0.2, 2.0, 0.2, 0.2, 1.0, 1.0, 1.0, 1.05, 1.0, 1.0, 1.0, 1.0, 1.7, 1.8)

  def _select(
    self,
    administered: list[int],
    exposure_rates: np.ndarray | None = None,
    method: str = "item_info",
    r_max: float = 1.0,
    r_control: str = "passive",
    clusters: tuple[int, ...] = CLUSTERS,
  ) -> int | None:
    item_bank = two_pl_bank(list(self.DISCRIMINATIONS), [0.0] * len(self.DISCRIMINATIONS))
    selector = ClusterSelector(clusters=list(clusters), method=method, r_max=r_max, r_control=r_control)
    return selector.select(make_context(item_bank, administered, theta=0.0, exposure_rates=exposure_rates))

  @pytest.mark.parametrize(("method", "expected"), [("item_info", 1), ("cluster_info", 7), ("weighted_info", 13)])
  @pytest.mark.parametrize(
    "clusters", [CLUSTERS, tuple({0: 7, 1: 2**40, 2: 2**62}[label] for label in CLUSTERS)], ids=["small", "huge"]
  )
  def test_select_takes_the_best_item_of_the_cluster_each_method_prefers(
    self, method: str, expected: int, clusters: tuple[int, ...]
  ) -> None:
    """Each method ranks clusters differently; the pick is the most informative item of the chosen cluster.

    Labels need not be small: hashed or database IDs must pick the same items. Scoring arrays indexed by label value
    allocated one slot per value up to the largest label, which fails outright for 2**62.
    """
    assert self._select([], method=method, clusters=clusters) == expected

  def test_select_moves_on_once_a_cluster_is_fully_administered(self) -> None:
    """With cluster 0 used up, item_info turns to the cluster of the next most informative item."""
    assert self._select([0, 1, 2, 3]) == 13

  def test_item_info_scores_a_cluster_by_its_administered_items_too(self) -> None:
    """Item 0 is given, yet its cluster keeps its score, so item 1 (0.037) is picked over item 2 (1.0).

    This is the published method (see the technique page). Scoring only the remaining items would pick item 2.
    """
    item_bank = ItemBank(np.array([[2.5, 0.0, 0.0, 1.0], [0.5, 3.0, 0.0, 1.0], [2.0, 0.0, 0.0, 1.0]]))

    assert ClusterSelector(clusters=[0, 0, 1]).select(make_context(item_bank, [0], theta=0.0)) == 1

  @pytest.mark.parametrize("method", ["item_info", "cluster_info", "weighted_info"])
  @pytest.mark.parametrize("n_labels", [len(CLUSTERS) - 1, len(CLUSTERS) + 1])
  def test_select_rejects_labels_that_do_not_match_the_bank(self, method: str, n_labels: int) -> None:
    """Labels for another bank size raised IndexError, or with cluster_info silently skipped the unlabelled items."""
    item_bank = two_pl_bank(list(self.DISCRIMINATIONS), [0.0] * len(self.DISCRIMINATIONS))
    selector = ClusterSelector(clusters=[*self.CLUSTERS, 2][:n_labels], method=method)

    with pytest.raises(ValueError, match=f"clusters holds {n_labels} labels, but the item bank has 14 items"):
      selector.select(make_context(item_bank, [], theta=0.0))

  def test_cluster_sums_and_means_match_the_hand_computed_values(self) -> None:
    """Each cluster's information sum and mean must be a^2 / 4 summed or averaged over its items, gaps in labels too.

    The three clusters are relabelled 0, 3 and 5, so labels 1, 2 and 4 hold no item and must not appear.
    """
    item_bank = two_pl_bank(list(self.DISCRIMINATIONS), [0.0] * len(self.DISCRIMINATIONS))
    clusters = [{0: 0, 1: 3, 2: 5}[label] for label in self.CLUSTERS]
    sums = {0: (3 * 0.2**2 + 2.0**2) / 4, 3: (7 * 1.0**2 + 1.05**2) / 4, 5: (1.7**2 + 1.8**2) / 4}

    assert ClusterSelector.sum_cluster_infos(0.0, item_bank, clusters) == pytest.approx(sums, rel=1e-12)
    assert ClusterSelector.weighted_cluster_infos(0.0, item_bank, clusters) == pytest.approx(
      {0: sums[0] / 4, 3: sums[3] / 8, 5: sums[5] / 2}, rel=1e-12
    )

  @pytest.mark.parametrize(
    ("kwargs", "match"),
    [
      ({"clusters": []}, "cannot be empty"),
      ({"clusters": [0, -1]}, "non-negative integers"),
      ({"clusters": [0.0, 1.5]}, "non-negative integers"),
      ({"clusters": [0, 1], "method": "max_info"}, "not a valid cluster selection method"),
      ({"clusters": [0, 1], "r_control": "strict"}, "not a valid cap fallback"),
    ],
  )
  def test_constructor_rejects_invalid_arguments(self, kwargs: dict[str, object], match: str) -> None:
    """A bad label, method or fallback must fail at construction; float labels used to be accepted."""
    with pytest.raises(ValueError, match=match):
      ClusterSelector(**kwargs)

  def test_select_prefers_items_below_the_exposure_cap(self) -> None:
    """Inside the chosen cluster, an item at the cap is passed over for a less informative one below it."""
    exposure_rates = np.zeros(len(self.CLUSTERS))
    exposure_rates[13] = 0.9

    assert self._select([], exposure_rates, method="weighted_info", r_max=0.5) == 12

  def test_the_default_cap_of_1_holds_back_no_item(self) -> None:
    """r_max = 1 applies no cap inside the cluster, even to an item every earlier examinee received (rate 1).

    Fails if the strict ``rate < r_max`` test passes item 13 over for the less informative item 12.
    """
    exposure_rates = np.zeros(len(self.CLUSTERS))
    exposure_rates[13] = 1.0

    assert self._select([], exposure_rates, method="weighted_info") == 13

  @pytest.mark.parametrize(
    ("rates", "r_control", "expected"),
    [
      ((0.8, 0.9), "passive", 13),
      ((0.8, 0.9), "aggressive", 12),
      # item 13 is now the least exposed and still the more informative: not the first or least informative item
      ((0.9, 0.8), "aggressive", 13),
    ],
  )
  def test_r_control_decides_when_the_whole_cluster_is_over_the_cap(
    self, rates: tuple[float, float], r_control: str, expected: int
  ) -> None:
    """Passive control still administers the most informative item; aggressive control the least exposed one."""
    exposure_rates = np.zeros(len(self.CLUSTERS))
    exposure_rates[[12, 13]] = rates

    assert self._select([], exposure_rates, method="weighted_info", r_max=0.5, r_control=r_control) == expected

  def test_aggressive_control_breaks_exposure_ties_at_random(self) -> None:
    """Items tied for the smallest exposure rate are drawn uniformly (ADR-0008)."""
    item_bank = ItemBank.generate(12, rng=4)
    rates = np.full(12, 0.5)
    rates[[3, 7, 9]] = 0.2
    selector = ClusterSelector(clusters=[0] * 12, r_max=0.1, r_control="aggressive")

    picks = {
      selector.select(make_context(item_bank, [], theta=0.0, rng=np.random.default_rng(seed), exposure_rates=rates))
      for seed in range(40)
    }

    assert picks == {3, 7, 9}
