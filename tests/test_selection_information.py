"""Tests for MaxInfoSelector, UrrySelector and IntervalInfoSelector."""

import numpy as np
import pytest
from irtlib import ItemBank, irt
from scipy.integrate import quad

from catsim.selection import IntervalInfoSelector, MaxInfoSelector, UrrySelector
from tests.banks import two_pl_bank
from tests.contexts import make_context


class TestMaxInfoSelector:
  """Tests for MaxInfoSelector."""

  # all difficulties are 0, so at theta = 0 information is a^2 / 4 and ranks items 1, 2, 0, 3
  BANK_A = (1.0, 2.0, 1.5, 0.5)

  def test_select_returns_the_most_informative_remaining_item(self) -> None:
    """The pick is the remaining item with the largest information at the estimate."""
    item_bank = two_pl_bank(list(self.BANK_A), [0.0] * 4)
    selector = MaxInfoSelector()

    assert selector.select(make_context(item_bank, [], theta=0.0)) == 1
    assert selector.select(make_context(item_bank, [1], theta=0.0)) == 2

  def test_select_skips_items_at_the_exposure_cap(self) -> None:
    """An item whose exposure rate has reached r_max is passed over for the next most informative one."""
    item_bank = two_pl_bank(list(self.BANK_A), [0.0] * 4)
    exposure_rates = np.array([0.0, 0.5, 0.0, 0.0])

    selected = MaxInfoSelector(r_max=0.5).select(make_context(item_bank, [], theta=0.0, exposure_rates=exposure_rates))

    assert selected == 2

  def test_the_default_cap_of_1_holds_back_no_item(self) -> None:
    """r_max = 1 applies no cap, as in catR, even to an item every earlier examinee received.

    Such an item has a rate of exactly 1. Fails if the strict ``rate < r_max`` test passes it over, which made the
    default selector give the second examinee none of the first examinee's items.
    """
    item_bank = two_pl_bank(list(self.BANK_A), [0.0] * 4)

    selected = MaxInfoSelector().select(
      make_context(item_bank, [], theta=0.0, exposure_rates=np.array([0.0, 1.0, 0.0, 0.0]))
    )

    assert selected == 1

  def test_select_ignores_the_cap_when_every_item_is_over_it(self) -> None:
    """With every remaining item at or over r_max, the most informative one is still administered."""
    item_bank = two_pl_bank(list(self.BANK_A), [0.0] * 4)

    selected = MaxInfoSelector(r_max=0.5).select(make_context(item_bank, [], theta=0.0, exposure_rates=np.ones(4)))

    assert selected == 1


class TestUrrySelector:
  """Tests for UrrySelector."""

  def test_select_matches_difficulty_to_the_estimate_whatever_the_discrimination(self) -> None:
    """The pick is the remaining item whose difficulty is closest to the estimate, not the most informative one."""
    # at theta = 0.8, item 0 is the most informative but item 1's difficulty is the closest
    item_bank = two_pl_bank([2.5, 0.5, 1.0, 1.0], [0.3, 0.9, -1.0, 2.5])
    selector = UrrySelector()

    assert MaxInfoSelector().select(make_context(item_bank, [], theta=0.8)) == 0
    assert selector.select(make_context(item_bank, [], theta=0.8)) == 1
    assert selector.select(make_context(item_bank, [1], theta=0.8)) == 0


class TestIntervalInfoSelector:
  """Tests for IntervalInfoSelector."""

  @pytest.mark.parametrize(("kwargs", "match"), [({"interval": 0.0}, "interval"), ({"confidence": 1.0}, "confidence")])
  def test_init_rejects_invalid_arguments(self, kwargs: dict[str, float], match: str) -> None:
    """The half-width must be positive and the confidence level strictly between 0 and 1."""
    with pytest.raises(ValueError, match=match):
      IntervalInfoSelector(**kwargs)

  def test_default_interval_is_scaled_by_the_context_standard_error(self) -> None:
    """A selector that ignored context.se, such as one that recomputed the Fisher standard error, picks one item twice.

    Over [-w, w] the 2PL integral is a [sigma(a (w - b)) - sigma(a (-w - b))]. With w = 1.96 * 0.15 = 0.29, item 0
    (a = 2, b = 0) scores 0.572 and item 1 (a = 3, b = 1.5) 0.065; with w = 1.96 * 1.5 = 2.94, 1.989 and 2.961.
    """
    item_bank = ItemBank(np.array([[2.0, 0.0, 0.0, 1.0], [3.0, 1.5, 0.0, 1.0]]))

    narrow = IntervalInfoSelector().select(make_context(item_bank, theta=0.0, se=0.15))
    wide = IntervalInfoSelector().select(make_context(item_bank, theta=0.0, se=1.5))

    assert (narrow, wide) == (0, 1)

  def test_default_interval_with_the_fisher_standard_error_is_equation_11(self) -> None:
    """With the MLE's standard error 1 / sqrt(I(theta_hat)), the interval must be Veerkamp & Berger's (1994) Eq. 11.

    The quadrature over theta_hat +/- 1.96 SE picks item 51; the whole scale, an interval of the same width centred at 0
    and one 2.576 SE wide each pick item 24.
    """
    item_bank = ItemBank.generate(60, model="3PL", rng=7)
    administered = [3, 11, 25, 40, 52]
    theta = 0.7
    se = 1 / np.sqrt(irt.total_information(theta, item_bank.items[administered]))
    half_width = 1.959963984540054 * se
    remaining = [i for i in range(item_bank.n_items) if i not in administered]
    integrals = [
      quad(irt.information, theta - half_width, theta + half_width, args=(item_bank.items[i],))[0] for i in remaining
    ]

    selected = IntervalInfoSelector().select(make_context(item_bank, administered, theta=theta, se=se))

    assert selected == remaining[int(np.argmax(integrals))]

  def test_default_interval_is_the_whole_scale_while_the_standard_error_is_infinite(self) -> None:
    """An infinite standard error must integrate over the whole ability scale (Equation 16), where a 2PL item scores a.

    Item 1 (a = 2, b = 5) then scores 2 against item 0's (a = 1, b = 0) 1. Over any interval narrower than
    theta_hat +/- 4.98, item 0 scores more.
    """
    item_bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [2.0, 5.0, 0.0, 1.0]]))

    assert IntervalInfoSelector().select(make_context(item_bank, theta=0.0, se=np.inf)) == 1

  @pytest.mark.parametrize("se", [5e-324, 1e-300, 1e-16])
  def test_a_vanishing_interval_selects_the_most_informative_item_at_the_estimate(self, se: float) -> None:
    """A near-zero standard error selects by the items' information at theta_hat, the interval's limit.

    At theta = 0.3 the 2PL items (a, b) = (2, 0), (1, 3) and (3, -0.2) give a^2 P (1 - P) = 0.915, 0.059 and 1.34,
    so item 2. Fails if the selector integrates over the collapsed interval: every integral is 0 (a random tie) or
    rounding noise, which picks another item for some seeds.
    """
    item_bank = ItemBank(np.array([[2.0, 0.0, 0.0, 1.0], [1.0, 3.0, 0.0, 1.0], [3.0, -0.2, 0.0, 1.0]]))

    selected = {
      IntervalInfoSelector().select(make_context(item_bank, theta=0.3, se=se, rng=np.random.default_rng(seed)))
      for seed in range(20)
    }

    assert selected == {2}

  def test_fixed_interval_matches_numerical_integration(self) -> None:
    """A fixed half-width integrates information over theta_hat +/- interval."""
    item_bank = ItemBank.generate(40, rng=8)
    integrals = [quad(irt.information, -1.3, -0.3, args=(item,))[0] for item in item_bank.items]

    selected = IntervalInfoSelector(interval=0.5).select(make_context(item_bank, [], theta=-0.8))

    assert selected == int(np.argmax(integrals))
