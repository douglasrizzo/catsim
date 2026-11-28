"""Tests for KLSelector."""

import numpy as np
import pytest
from irtlib import ItemBank
from scipy.integrate import quad

from catsim.selection import KLSelector
from tests.contexts import make_context


def _kl_reference(items: np.ndarray, theta_hat: float, half_width: float) -> list[float]:
  """Return each item's Bernoulli divergence from theta_hat integrated over theta_hat +- half_width by scipy's quad.

  P and Q are written as sums of positive terms, so they stay accurate at any logit, and quad splits the window at
  the difficulty, where a steep item's divergence turns.
  """
  lower, upper = theta_hat - half_width, theta_hat + half_width

  def divergence(theta: float, item: np.ndarray) -> float:
    a, b, c, d = item
    p_hat, q_hat = (
      c + (d - c) / (1 + np.exp(-a * (theta_hat - b))),
      (1 - d) + (d - c) / (1 + np.exp(a * (theta_hat - b))),
    )
    p, q = c + (d - c) / (1 + np.exp(-a * (theta - b))), (1 - d) + (d - c) / (1 + np.exp(a * (theta - b)))
    return p_hat * np.log(p_hat / p) + q_hat * np.log(q_hat / q)

  return [
    quad(
      divergence,
      lower,
      upper,
      args=(item,),
      epsabs=0.0,
      epsrel=1e-12,
      limit=200,
      points=[item[1]] if lower < item[1] < upper else None,
    )[0]
    for item in items
  ]


class TestKLSelector:
  """Tests for KLSelector."""

  @pytest.mark.parametrize(
    ("kwargs", "match"),
    [
      pytest.param({"window": 0.0}, "window must be positive", id="zero_window"),
      pytest.param({"window": -1.0}, "window must be positive", id="negative_window"),
    ],
  )
  def test_init_rejects_invalid_arguments(self, kwargs: dict[str, float], match: str) -> None:
    """A window of zero width would make every item's KL integral 0 and the pick a random tie."""
    with pytest.raises(ValueError, match=match):
      KLSelector(**kwargs)

  @pytest.mark.parametrize(
    "items",
    [
      *(
        pytest.param(ItemBank.generate(30, model=model, rng=7).items, id=model)
        for model in ("1PL", "2PL", "3PL", "4PL")
      ),
      pytest.param(
        np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 0.3, 0.2, 1.0], [1.2, -0.5, 0.0, 0.9], [1.8, 0.4, 0.15, 0.85]]),
        id="every_branch",
      ),
    ],
  )
  @pytest.mark.parametrize(("theta_hat", "half_width"), [(-1.2, 3.0), (0.4, 3.0 / np.sqrt(5)), (2.0, 0.5), (0.0, 6.0)])
  def test_kl_integrals_match_adaptive_quadrature(self, items: np.ndarray, theta_hat: float, half_width: float) -> None:
    """Each item's integral matches scipy's adaptive quadrature of the Bernoulli divergence to 1e-9.

    Nodes not scaled to the window, a missing half-width factor or the two abilities swapped in the divergence miss
    every item; a fixed 24-node rule misses the generated items over theta_hat +- 6.
    """
    reference = _kl_reference(items, theta_hat, half_width)

    integrals = KLSelector._kl_global(theta_hat, items, half_width)  # noqa: SLF001 - select shows only the largest
    np.testing.assert_allclose(integrals, reference, rtol=1e-9)

  @pytest.mark.parametrize(
    "items",
    [
      pytest.param(np.array([[4.0, 0.3, 0.0, 1.0], [6.0, -0.6, 0.0, 1.0], [10.0, 0.8, 0.0, 1.0]]), id="2pl"),
      pytest.param(np.array([[4.0, -0.2, 0.25, 0.9], [6.0, 0.4, 0.3, 0.85], [10.0, -0.5, 0.2, 0.95]]), id="4pl"),
    ],
  )
  @pytest.mark.parametrize(("theta_hat", "half_width"), [(0.0, 1.5), (0.3, 3.0), (-0.2, 6.0)])
  def test_kl_integrals_of_steep_items_match_adaptive_quadrature(
    self, items: np.ndarray, theta_hat: float, half_width: float
  ) -> None:
    """Items with a = 4 to 10 get integrals within 1e-10 of adaptive quadrature, over narrow and wide windows.

    A rule whose node count ignores the discrimination misses them: 24 nodes are off by up to 2e-3, enough to change
    which of two steep items is selected.
    """
    reference = _kl_reference(items, theta_hat, half_width)

    integrals = KLSelector._kl_global(theta_hat, items, half_width)  # noqa: SLF001 - select shows only the largest
    np.testing.assert_allclose(integrals, reference, rtol=1e-10)

  def test_select_ranks_steep_items_by_their_integrals(self) -> None:
    """Of two steep items whose KL integrals over 0 +- 3 differ by 4e-6 of their size, the larger one is selected.

    A fixed 24-node rule errs by more than that margin on them and selects the other item.
    """
    # the second discrimination is tuned so that the two integrals differ by a relative 4e-6
    items = np.array([[6.0, -0.6, 0.0, 1.0], [5.011988691929875, 0.3, 0.0, 1.0]])
    reference = _kl_reference(items, 0.0, 3.0)

    assert reference[0] > reference[1]
    assert KLSelector().select(make_context(ItemBank(items))) == 0

  @pytest.mark.parametrize("n_administered", [0, 1, 4, 9])
  def test_select_maximizes_kl_over_a_window_shrinking_with_the_test(self, n_administered: int) -> None:
    """The pick has the largest KL integral over theta_hat +- c / sqrt(max(1, n)) among the remaining items.

    The reference integrates the Bernoulli divergence, written out in the test, with the trapezoid rule.
    It fails if the window ignores the number of administered items, shrinks it by n instead of sqrt(n),
    or integrates over the wrong interval.
    """
    rng = np.random.default_rng(n_administered)
    for _ in range(10):
      item_bank = ItemBank.generate(25, model="3PL", rng=int(rng.integers(1_000_000)))
      administered = [int(idx) for idx in rng.choice(25, n_administered, replace=False)]
      theta_hat = float(rng.uniform(-1.5, 1.5))
      half_width = 2.0 / np.sqrt(max(1, n_administered))
      grid = np.linspace(theta_hat - half_width, theta_hat + half_width, 4001)
      a, b, c, d = (item_bank.items[:, [k]] for k in range(4))
      p = c + (d - c) / (1.0 + np.exp(-a * (grid - b)))
      p_hat = c + (d - c) / (1.0 + np.exp(-a * (theta_hat - b)))
      divergence = p_hat * np.log(p_hat / p) + (1.0 - p_hat) * np.log((1.0 - p_hat) / (1.0 - p))
      reference = np.trapezoid(divergence, grid, axis=1)
      reference[administered] = -np.inf

      picked = KLSelector(window=2.0).select(
        make_context(item_bank, administered, theta=theta_hat, rng=np.random.default_rng(0))
      )

      assert reference[picked] == pytest.approx(reference.max(), rel=1e-6)

  @pytest.mark.parametrize(("administered", "expected"), [([2], 1), ([2, 3, 4, 5], 0)])
  def test_window_shrinks_with_the_square_root_of_the_test_length(self, administered: list[int], expected: int) -> None:
    """With window c = 2 the window is theta_hat +- 2 after one item and +- 1 after four.

    At theta_hat = 0, item 0 (a = 1, b = 0) and item 1 (a = 4, b = 1.5) trade places as the window narrows:
    their KL integrals are 0.610 and 0.869 over +- 2, 0.225 and 0.146 over +- 1.41, and 0.081 and 0.028
    over +- 1. A window of c / sqrt(n + 1) would already prefer item 0 after one item.
    """
    filler = [1.0, 3.0, 0.0, 1.0]
    item_bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0], [4.0, 1.5, 0.0, 1.0], filler, filler, filler, filler]))

    assert (
      KLSelector(window=2.0).select(make_context(item_bank, administered, theta=0.0, rng=np.random.default_rng(0)))
      == expected
    )

  def test_select_skips_items_at_the_exposure_cap(self) -> None:
    """The item with the most KL information is passed over while its exposure rate is at r_max."""
    item_bank = ItemBank(np.array([[1.0, -2.0, 0.0, 1.0], [1.0, 0.0, 0.0, 1.0], [1.0, 1.5, 0.0, 1.0]]))
    selector = KLSelector(window=0.5, r_max=0.5)

    uncapped = selector.select(make_context(item_bank, [], theta=0.0, rng=np.random.default_rng(0)))
    capped = selector.select(
      make_context(item_bank, [], theta=0.0, rng=np.random.default_rng(0), exposure_rates=np.array([0.0, 0.5, 0.0]))
    )

    assert (uncapped, capped) == (1, 2)
