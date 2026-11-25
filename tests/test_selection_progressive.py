"""Tests for ProgressiveSelector and ProportionalSelector."""

from collections.abc import Callable
from fractions import Fraction

import numpy as np
import pytest

from catsim.item_bank import ItemBank
from catsim.selection import BaseSelector, ProgressiveSelector, ProportionalSelector
from catsim.typing import FloatArray
from tests.banks import two_pl_bank
from tests.contexts import make_context
from tests.generators import DeterministicRNG


def _progressive_item_bank() -> ItemBank:
  """Create a small bank with a clear information ranking at theta=0."""
  return ItemBank(
    np.array(
      [
        [1.0, -3.0, 0.0, 1.0],
        [1.0, -1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 1.0],
      ],
      dtype=float,
    )
  )


class TestPositionWeight:
  """Tests for the position weight, the schedule ProgressiveSelector and ProportionalSelector share.

  ProportionalSelector draws each remaining item with probability proportional to its information raised to the test
  length times the weight, so the probabilities it hands the generator give the weight away.
  """

  # Eight 1PL items, whose information at theta = 0 is e^b / (1 + e^b)^2 in closed form.
  DIFFICULTIES = np.linspace(-1.5, 2.0, 8)

  def _draw_probabilities(self, test_size: int, acceleration: float, position: int) -> FloatArray:
    """Return the probabilities ProportionalSelector draws with at a position, the earlier items administered."""
    rng = DeterministicRNG([])
    ProportionalSelector(test_size=test_size, acceleration=acceleration).select(
      make_context(two_pl_bank([1.0] * 8, self.DIFFICULTIES.tolist()), list(range(position - 1)), theta=0.0, rng=rng)
    )
    ((_, probs),) = rng.choice_calls
    assert probs is not None
    return probs

  def _probabilities_for(self, position: int, exponent: float) -> FloatArray:
    """Return the remaining items' information raised to the exponent, normalized, from the closed form."""
    difficulties = self.DIFFICULTIES[position - 1 :]
    weights = (np.exp(difficulties) / (1 + np.exp(difficulties)) ** 2) ** exponent
    return weights / weights.sum()

  @pytest.mark.parametrize(
    ("test_size", "acceleration", "position", "expected"),
    [
      pytest.param(5, 1.0, 1, 0.0, id="first_position"),
      pytest.param(5, 0.0, 3, 2 / 4, id="zero_is_linear"),
      pytest.param(5, 1.0, 3, (1 + 2) / (1 + 2 + 3 + 4), id="positive_delays"),
      pytest.param(5, 2.0, 4, (1 + 4 + 9) / (1 + 4 + 9 + 16), id="larger_delays_more"),
      pytest.param(5, -1.0, 2, 1 / (1 + 1 / 2 + 1 / 3 + 1 / 4), id="negative_hastens"),
      pytest.param(5, 1.0, 5, 1.0, id="last_position"),
      pytest.param(5, 1.0, 7, 1.0, id="clamped_past_test_size"),
      pytest.param(1, 1.0, 1, 1.0, id="one_item_test"),
    ],
  )
  def test_follows_barrada_et_al_equation_12(
    self, test_size: int, acceleration: float, position: int, expected: float
  ) -> None:
    """At position n of N the weight is the sum of k ** t for k < n over the same sum for k < N.

    The expected values are those sums worked by hand from Barrada et al.'s (2008) Equation 12, which catR's nextItem
    computes too. A power schedule, ((n - 1) / (N - 1)) ** t, would give 0.5, 0.5625 and 4 in the t = 1, 2 and -1
    rows. Past N the weight stays 1: the stopper, not the selector, ends the test, and a weight above 1 would give the
    random component a negative coefficient. A one-item test is at its last position from the start.
    """
    np.testing.assert_allclose(
      self._draw_probabilities(test_size, acceleration, position),
      self._probabilities_for(position, test_size * expected),
      rtol=1e-12,
    )

  def test_large_acceleration_does_not_overflow(self) -> None:
    """3.0 ** 1000 and 4.0 ** 1000 overflow to inf, which would make the weight inf / inf and every draw NaN.

    The acceleration is finite, so the selector accepts it. The expected weight is the ratio of the sums in exact
    integer arithmetic.
    """
    expected = Fraction(sum(m**1000 for m in range(1, 4)), sum(m**1000 for m in range(1, 5)))

    np.testing.assert_allclose(
      self._draw_probabilities(5, 1000.0, 4), self._probabilities_for(4, 5 * float(expected)), rtol=1e-12
    )


class TestProgressiveSelector:
  """Tests for ProgressiveSelector."""

  @pytest.mark.parametrize(
    ("administered", "draws", "expected"),
    [
      pytest.param([], [0.1, 0.9, 0.2, 0.3], 1, id="first_position_follows_the_draws"),
      pytest.param([0, 2], [0.1, 0.99, 0.2, 0.0], 3, id="last_position_follows_the_information"),
    ],
  )
  def test_select_moves_from_the_draws_to_the_information(
    self, administered: list[int], draws: list[float], expected: int
  ) -> None:
    """The largest random draw wins at the first position, the most informative item at the last.

    Item 3 (a = 2, b = 0) is the most informative at theta = 0, with scaled information 1, against
    0.197 for item 1. A weight of 0.5 at either end would flip the pick: 0.548 for item 1 against 0.65
    for item 3 with the first draws, and 0.594 against 0.5 with the last ones.
    """
    selector = ProgressiveSelector(test_size=3)

    selected = selector.select(
      make_context(_progressive_item_bank(), administered, theta=0.0, rng=DeterministicRNG(draws))
    )

    assert selected == expected

  def test_select_respects_exposure_cap(self) -> None:
    """Item 3, the most informative, is at the cap, so the last position, scored by information only, passes it over."""
    item_bank = _progressive_item_bank()
    selector = ProgressiveSelector(test_size=3, r_max=0.5)
    exposure_rates = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    selected = selector.select(
      make_context(
        item_bank, [0, 1], theta=0.0, rng=DeterministicRNG([0.1, 0.2, 0.3, 0.4]), exposure_rates=exposure_rates
      )
    )

    assert selected == 2

  def test_random_component_is_scaled_to_available_items(self) -> None:
    """Both score components share the scale of the most informative available item.

    Item 0 dominates the bank's information at theta=0 but is already administered. Normalizing
    by the bank-wide maximum would shrink the remaining items' information toward 0 and let the
    random draw decide (item 2); normalizing by the available maximum keeps the halfway blend
    balanced, so the far more informative item 1 wins.
    """
    item_bank = ItemBank(
      np.array(
        [
          [3.0, 0.0, 0.0, 1.0],
          [1.0, 0.0, 0.0, 1.0],
          [0.5, 0.0, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    selector = ProgressiveSelector(test_size=3)

    selected = selector.select(make_context(item_bank, [0], theta=0.0, rng=DeterministicRNG([0.9, 0.1, 0.4])))

    assert selected == 1


class TestProportionalSelector:
  """Tests for ProportionalSelector."""

  # Information of _progressive_item_bank's items at theta = 0, in closed form: L (1 - L) with L = 1 / (1 + e^b) for
  # the items with a = 1, and a^2 / 4 for item 3.
  INFO_AT_ZERO = np.array([np.e**3 / (1 + np.e**3) ** 2, np.e / (1 + np.e) ** 2, np.e / (1 + np.e) ** 2, 1.0])

  @pytest.mark.parametrize(
    ("test_size", "acceleration", "administered", "exponent"),
    [
      pytest.param(3, 0.0, [], 0.0, id="first_position_is_uniform"),
      pytest.param(4, 1.0, [0, 1], 4 * (1 + 2) / (1 + 2 + 3), id="positive_acceleration"),
      pytest.param(3, -1.0, [0], 3 * 1 / (1 + 1 / 2), id="negative_acceleration"),
      pytest.param(3, 0.0, [0, 1], 3.0, id="last_position_reaches_the_test_length"),
    ],
  )
  def test_select_draws_by_information_raised_to_the_scheduled_exponent(
    self, test_size: int, acceleration: float, administered: list[int], exponent: float
  ) -> None:
    """Each remaining item is drawn with probability proportional to its information raised to N times the weight.

    That exponent is Barrada et al.'s (2008) Equation 13, worked by hand here from the weight's sums: 0 at the first
    position and the test length N at the last. An exponent of 6 ((n - 1) / (N - 1)) ** t would raise the information
    to 4 in the positive_acceleration row, and requiring a positive acceleration would reject the other rows.
    """
    rng = DeterministicRNG([0.0, 0.0, 0.0, 0.0])

    ProportionalSelector(test_size=test_size, acceleration=acceleration).select(
      make_context(_progressive_item_bank(), administered, theta=0.0, rng=rng)
    )

    ((candidates, probs),) = rng.choice_calls
    assert probs is not None
    remaining_items = [item for item in range(4) if item not in administered]
    assert candidates.tolist() == remaining_items
    weights = self.INFO_AT_ZERO[remaining_items] ** exponent
    np.testing.assert_allclose(probs, weights / weights.sum())

  def test_large_exponent_does_not_underflow_to_uniform_sampling(self) -> None:
    """Far from every item, the information raised to the exponent underflows to 0 for every item.

    At theta = -400 the information of an item with a = 1 is about e^(-400 - b), and its square, the exponent at the
    last position of a two-item test, is below the smallest float. Unscaled weights would all be 0, and the draw would
    fall back to uniform sampling, or divide 0 by 0, where it should favor the most informative item.
    """
    rng = DeterministicRNG([0.0, 0.0, 0.0, 0.0])

    ProportionalSelector(test_size=2).select(
      make_context(two_pl_bank([1.0] * 4, [0.0, 1.0, 2.0, 3.0]), [3], theta=-400.0, rng=rng)
    )

    ((_, probs),) = rng.choice_calls
    assert probs is not None
    # This far out the information ratios are e^-b to double precision, and the exponent is the test length, 2.
    weights = np.exp(-2.0 * np.array([0.0, 1.0, 2.0]))
    np.testing.assert_allclose(probs, weights / weights.sum())


@pytest.mark.parametrize("acceleration", [float("nan"), float("inf")])
@pytest.mark.parametrize("selector_class", [ProgressiveSelector, ProportionalSelector])
def test_scheduled_selectors_reject_non_finite_acceleration(
  selector_class: Callable[..., BaseSelector], acceleration: float
) -> None:
  """A NaN or infinite acceleration makes the weight NaN; an infinite one makes both of its sums infinite.

  Without the check the selector would construct, then fail at its first select() on NaN scores or probabilities.
  """
  with pytest.raises(ValueError, match="acceleration must be finite"):
    selector_class(test_size=10, acceleration=acceleration)


@pytest.mark.parametrize(
  "make_selector",
  [
    pytest.param(lambda: ProgressiveSelector(test_size=8), id="progressive"),
    pytest.param(lambda: ProportionalSelector(test_size=8), id="proportional"),
  ],
)
def test_randomized_selectors_repeat_with_the_same_seed(make_selector: Callable[[], BaseSelector]) -> None:
  """Two runs seeded alike choose the same items at every position, as both selectors' pages guarantee."""
  item_bank = ItemBank.generate(40, rng=5)

  def run(seed: int) -> list[int | None]:
    selector, rng, administered = make_selector(), np.random.default_rng(seed), []
    for _ in range(8):
      administered.append(selector.select(make_context(item_bank, administered, theta=0.3, rng=rng)))
    return administered

  assert run(11) == run(11)
  assert run(11) != run(12)
