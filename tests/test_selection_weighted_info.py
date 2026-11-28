"""Tests for MLWISelector."""

import numpy as np
import pytest
from irtlib import ItemBank, irt

from catsim.estimation import NumericalSearchEstimator
from catsim.selection import MaxInfoSelector, MLWISelector
from tests.contexts import make_context


class TestMLWISelector:
  """Tests for MLWISelector."""

  # two administered 2PL items near 0, then an easy, a middle, a hard and a flat candidate
  HAND_BUILT = np.array([
    [1.0, 0.0, 0.0, 1.0],
    [1.0, 0.5, 0.0, 1.0],
    [1.6, -2.0, 0.0, 1.0],
    [1.3, 0.3, 0.0, 1.0],
    [1.6, 2.0, 0.0, 1.0],
    [0.5, 0.0, 0.0, 1.0],
  ])

  @pytest.mark.parametrize(
    ("responses", "expected"),
    [([True, True], 4), ([False, False], 2), ([True, False], 3)],
  )
  def test_select_matches_brute_force_weighted_information(self, responses: list[bool], expected: int) -> None:
    """The pick has the largest sum of likelihood times information over the 41 nodes, and it moves with the responses.

    The reference multiplies the response probabilities node by node. Ignoring the likelihood ties the easy and the
    hard item on their total information, and reading the responses reversed swaps them.
    """
    nodes = np.linspace(-6.0, 6.0, 41)
    likelihood = np.ones_like(nodes)
    for item, correct in zip(self.HAND_BUILT[:2], responses, strict=True):
      p = irt.icc(nodes, item)
      likelihood *= p if correct else 1.0 - p
    brute_force = [float(np.sum(likelihood * irt.information(nodes, item))) for item in self.HAND_BUILT[2:]]

    selected = MLWISelector().select(make_context(ItemBank(self.HAND_BUILT), [0, 1], responses))

    assert selected == 2 + int(np.argmax(brute_force)) == expected

  def test_select_falls_back_when_exposure_cap_blocks_all_candidates(self) -> None:
    """With every remaining item over r_max, the cap is ignored and the best remaining item is still chosen."""
    selector = MLWISelector(r_max=0.5)

    selected = selector.select(
      make_context(ItemBank(self.HAND_BUILT), [0, 1], [True, False], exposure_rates=np.ones(len(self.HAND_BUILT)))
    )

    # the pick of test_select_matches_brute_force_weighted_information for these responses
    assert selected == 3

  def test_select_with_no_data_prefers_the_most_discriminating_item(self) -> None:
    """With flat weights a 2PL item's integrated information is its discrimination a, whatever its difficulty."""
    item_bank = ItemBank(
      np.array(
        [[0.8, 2.0, 0.0, 1.0], [1.9, -2.5, 0.0, 1.0], [1.2, 0.0, 0.0, 1.0], [1.5, 1.0, 0.0, 1.0]],
        dtype=float,
      )
    )

    selected = MLWISelector().select(make_context(item_bank, [], [], theta=0.0))

    assert selected == 1

  def test_select_after_responses_to_extreme_items(self) -> None:
    """Items far from every node leave the likelihood flat inside the grid, so the most discriminating item is chosen.

    A wrong answer to an item at b = -6 and a right one to an item at b = 6, both with a = 200, have a log-likelihood
    of -2400 at every inner node. Computing log(1 - P) after P rounds to 1 would make it -inf at every node, and the
    selection would raise.
    """
    item_bank = ItemBank(
      np.array([
        [200.0, -6.0, 0.0, 1.0],
        [200.0, 6.0, 0.0, 1.0],
        [0.8, 2.0, 0.0, 1.0],
        [1.9, -2.5, 0.0, 1.0],
        [1.2, 0.0, 0.0, 1.0],
      ])
    )

    assert MLWISelector().select(make_context(item_bank, [0, 1], [False, True])) == 3

  @pytest.mark.parametrize("bounds", [(-np.inf, 6.0), (-6.0, np.inf), (np.nan, 6.0)])
  def test_init_rejects_non_finite_bounds(self, bounds: tuple[float, float]) -> None:
    """A non-finite bound would spread NaN nodes over the grid and fail only at the first selection."""
    with pytest.raises(ValueError, match=r"^bounds must be finite, got \("):
      MLWISelector(bounds=bounds)

  def test_init_rejects_non_increasing_bounds(self) -> None:
    """Reversed bounds would flip the sign of every score and select the least informative item."""
    with pytest.raises(ValueError, match="bounds must be increasing"):
      MLWISelector(bounds=(4.0, -4.0))
    with pytest.raises(ValueError, match="bounds must be increasing"):
      MLWISelector(bounds=(1.0, 1.0))

  def test_select_converges_to_max_info_on_long_test(self) -> None:
    """On a long test, MLWI should agree with MaxInfo near the MLE.

    As the likelihood sharpens, MLWI should concentrate on the MLE neighborhood,
    so disagreement here would signal broken weighting or integration.
    """
    item_bank = ItemBank.generate(30, rng=1)
    theta_true = 0.7
    rng = np.random.default_rng(5)
    administered_items = list(range(20))
    responses = [bool(rng.random() < irt.icc(theta_true, item_bank.items[idx])) for idx in administered_items]
    mle_theta = (
      NumericalSearchEstimator()
      .estimate(make_context(item_bank, administered_items, responses, theta=theta_true))
      .theta
    )

    mlwi = MLWISelector(n_nodes=81)
    max_info = MaxInfoSelector()

    mlwi_choice = mlwi.select(make_context(item_bank, administered_items, responses, theta=mle_theta))
    max_info_choice = max_info.select(make_context(item_bank, administered_items, theta=mle_theta))

    assert mlwi_choice == max_info_choice
