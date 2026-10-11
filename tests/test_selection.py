"""Tests for catsim.selection module."""

import numpy as np
import pytest
from scipy.integrate import quad

from catsim import irt
from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  BaseSelector,
  ClusterSelector,
  FiniteSelector,
  IntervalInfoSelector,
  LinearSelector,
  MaxInfoBBlockSelector,
  MaxInfoSelector,
  MaxInfoStratSelector,
  RandomesqueSelector,
  RandomSelector,
  The54321Selector,
  UrrySelector,
)


class TestMaxInfoSelector:
  """Tests for MaxInfoSelector."""

  def test_init_default(self) -> None:
    """Test default initialization."""
    selector = MaxInfoSelector()
    assert selector.r_max == pytest.approx(1.0)

  def test_init_with_r_max(self) -> None:
    """Test initialization with r_max."""
    selector = MaxInfoSelector(r_max=0.5)
    assert selector.r_max == pytest.approx(0.5)

  def test_init_invalid_r_max_raises(self) -> None:
    """Test that invalid r_max raises ValueError."""
    with pytest.raises(ValueError, match="between 0 and 1"):
      MaxInfoSelector(r_max=1.5)
    with pytest.raises(ValueError, match="between 0 and 1"):
      MaxInfoSelector(r_max=-0.1)

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    selector = MaxInfoSelector()

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 50

  def test_select_excludes_administered(self) -> None:
    """Test that select excludes administered items."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    selector = MaxInfoSelector()

    administered = [0, 1, 2, 3, 4]
    selected = selector.select(
      item_bank=item_bank,
      administered_items=administered,
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert selected not in administered

  def test_select_raises_when_exhausted(self) -> None:
    """Test that select raises NoItemsAvailableError when all items administered."""
    item_bank = ItemBank.generate_item_bank(5, seed=42)
    selector = MaxInfoSelector()

    administered = list(range(5))  # All items
    with pytest.raises(NoItemsAvailableError, match="no more items"):
      selector.select(
        item_bank=item_bank,
        administered_items=administered,
        est_theta=0.0,
        rng=np.random.default_rng(42),
      )


class TestRandomSelector:
  """Tests for RandomSelector."""

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    selector = RandomSelector()

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 50

  def test_select_is_random(self) -> None:
    """Test that selection is random across calls with different RNGs."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    selector = RandomSelector()

    selections = set()
    for seed in range(20):
      selected = selector.select(
        item_bank=item_bank,
        administered_items=[],
        est_theta=0.0,
        rng=np.random.default_rng(seed),
      )
      if selected is not None:
        selections.add(selected)

    # With 20 different seeds, we should get multiple different selections
    assert len(selections) > 1


class TestLinearSelector:
  """Tests for LinearSelector."""

  def test_init_with_indices(self) -> None:
    """Test initialization with item indices."""
    indices = [0, 5, 10, 15, 20]
    selector = LinearSelector(indices)
    assert selector.test_size == 5

  def test_select_returns_items_in_order(self) -> None:
    """Test that select returns items in the specified order."""
    indices = [5, 10, 15]
    selector = LinearSelector(indices)
    item_bank = ItemBank.generate_item_bank(50, seed=42)

    # First selection
    selected1 = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )
    assert selected1 == 5

    # Second selection
    selected2 = selector.select(
      item_bank=item_bank,
      administered_items=[5],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )
    assert selected2 == 10

    # Third selection
    selected3 = selector.select(
      item_bank=item_bank,
      administered_items=[5, 10],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )
    assert selected3 == 15


class TestUrrySelector:
  """Tests for UrrySelector."""

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    selector = UrrySelector()

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 50


class TestRandomesqueSelector:
  """Tests for RandomesqueSelector."""

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    selector = RandomesqueSelector(5)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 50


class TestThe54321Selector:
  """Tests for The54321Selector."""

  def test_init_with_test_size(self) -> None:
    """Test initialization with test size."""
    selector = The54321Selector(15)
    assert selector.test_size == 15

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    selector = The54321Selector(15)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 50


class TestAStratSelector:
  """Tests for AStratSelector."""

  def test_init_with_test_size(self) -> None:
    """Test initialization with test size."""
    selector = AStratSelector(20)
    assert selector.test_size == 20

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(100, seed=42)
    selector = AStratSelector(20)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 100


class TestAStratBBlockSelector:
  """Tests for AStratBBlockSelector."""

  def test_init_with_test_size(self) -> None:
    """Test initialization with test size."""
    selector = AStratBBlockSelector(20)
    assert selector.test_size == 20

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(100, seed=42)
    selector = AStratBBlockSelector(20)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 100


class TestMaxInfoStratSelector:
  """Tests for MaxInfoStratSelector."""

  def test_init_with_test_size(self) -> None:
    """Test initialization with test size."""
    selector = MaxInfoStratSelector(20)
    assert selector.test_size == 20

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(100, seed=42)
    selector = MaxInfoStratSelector(20)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 100


class TestMaxInfoBBlockSelector:
  """Tests for MaxInfoBBlockSelector."""

  def test_init_with_test_size(self) -> None:
    """Test initialization with test size."""
    selector = MaxInfoBBlockSelector(20)
    assert selector.test_size == 20

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(100, seed=42)
    selector = MaxInfoBBlockSelector(20)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 100


class TestIntervalInfoSelector:
  """Tests for IntervalInfoSelector."""

  def test_init_with_interval(self) -> None:
    """Test initialization with interval."""
    selector = IntervalInfoSelector(interval=2.0)
    assert selector.interval == pytest.approx(2.0)

  def test_init_default(self) -> None:
    """By default there is no fixed interval: the confidence interval of the estimate is used."""
    selector = IntervalInfoSelector()
    assert selector.interval is None
    assert selector.confidence == pytest.approx(0.95)

  @pytest.mark.parametrize(("kwargs", "match"), [({"interval": 0.0}, "interval"), ({"confidence": 1.0}, "confidence")])
  def test_init_rejects_invalid_arguments(self, kwargs: dict[str, float], match: str) -> None:
    """The half-width must be positive and the confidence level strictly between 0 and 1."""
    with pytest.raises(ValueError, match=match):
      IntervalInfoSelector(**kwargs)

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    selector = IntervalInfoSelector(interval=2.0)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 50

  def test_default_interval_is_the_confidence_interval_of_the_estimate(self) -> None:
    """Veerkamp & Berger (1997) integrate over theta_hat +/- z / sqrt(I(theta_hat)) of the administered items."""
    item_bank = ItemBank.generate_item_bank(60, itemtype="3PL", seed=7)
    administered = [3, 11, 25, 40, 52]
    theta = 0.7
    half_width = 1.959963984540054 / np.sqrt(irt.test_info(theta, item_bank.items[administered]))
    remaining = [i for i in range(item_bank.n_items) if i not in administered]
    integrals = [
      quad(irt.inf, theta - half_width, theta + half_width, args=tuple(item_bank.items[i, :4]))[0] for i in remaining
    ]

    selected = IntervalInfoSelector().select(item_bank=item_bank, administered_items=administered, est_theta=theta)

    assert selected == remaining[int(np.argmax(integrals))]

  def test_default_selection_adapts_to_the_estimate(self) -> None:
    """The default no longer integrates over the whole ability scale, which ignored the estimate.

    Those integrals were NaN, so earlier versions administered items in bank order. Before any item is
    administered the confidence interval is still unbounded, and the 2PL integral over it is a. With
    equal discriminations, the item whose difficulty is nearest the estimate has the largest integral
    over any interval centered on it.
    """
    items = np.column_stack([np.linspace(0.6, 2.0, 41), np.zeros(41), np.zeros(41), np.ones(41)])
    assert IntervalInfoSelector().select(item_bank=ItemBank(items), administered_items=[], est_theta=0.0) == 40

    items[:, 0], items[:, 1] = 1.2, np.linspace(-3.0, 3.0, 41)
    item_bank = ItemBank(items)
    administered = [18, 19, 20, 21, 22]
    low = IntervalInfoSelector().select(item_bank=item_bank, administered_items=administered, est_theta=-2.0)
    high = IntervalInfoSelector().select(item_bank=item_bank, administered_items=administered, est_theta=2.0)

    assert item_bank.difficulty[low] == pytest.approx(-1.95)
    assert item_bank.difficulty[high] == pytest.approx(1.95)

  def test_fixed_interval_matches_numerical_integration(self) -> None:
    """A fixed half-width integrates information over theta_hat +/- interval."""
    item_bank = ItemBank.generate_item_bank(40, seed=8)
    integrals = [quad(irt.inf, -1.3, -0.3, args=tuple(item[:4]))[0] for item in item_bank.items]

    selected = IntervalInfoSelector(interval=0.5).select(item_bank=item_bank, administered_items=[], est_theta=-0.8)

    assert selected == int(np.argmax(integrals))

  def test_select_excludes_administered_and_raises_when_exhausted(self) -> None:
    """Administered items are never returned; an exhausted bank raises NoItemsAvailableError."""
    item_bank = ItemBank.generate_item_bank(3, seed=9)
    selector = IntervalInfoSelector(interval=1.0)

    assert selector.select(item_bank=item_bank, administered_items=[0, 2], est_theta=0.0) == 1
    with pytest.raises(NoItemsAvailableError):
      selector.select(item_bank=item_bank, administered_items=[0, 1, 2], est_theta=0.0)


class TestClusterSelector:
  """Tests for ClusterSelector."""

  def test_select_returns_item(self) -> None:
    """Test that select returns a valid item index."""
    item_bank = ItemBank.generate_item_bank(50, seed=42)
    # Create cluster assignments
    clusters = [i % 5 for i in range(50)]
    selector = ClusterSelector(clusters=clusters)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected is not None
    assert 0 <= selected < 50


class TestBaseSelectorAbstract:
  """Tests for BaseSelector abstract base class."""

  def test_cannot_instantiate_base_selector(self) -> None:
    """Test that BaseSelector cannot be instantiated directly."""
    with pytest.raises(TypeError):
      BaseSelector()  # type: ignore[abstract]


class TestFiniteSelectorAbstract:
  """Tests for FiniteSelector abstract base class."""

  def test_cannot_instantiate_finite_selector(self) -> None:
    """Test that FiniteSelector cannot be instantiated directly."""
    with pytest.raises(TypeError):
      FiniteSelector(10)  # type: ignore[abstract]


_N_ITEMS = 30
SELECTOR_FACTORIES = {
  "MaxInfoSelector": MaxInfoSelector,
  "UrrySelector": UrrySelector,
  "IntervalInfoSelector": IntervalInfoSelector,
  "RandomSelector": RandomSelector,
  "RandomesqueSelector": lambda: RandomesqueSelector(bin_size=5),
  "The54321Selector": lambda: The54321Selector(test_size=10),
  "LinearSelector": lambda: LinearSelector(list(range(_N_ITEMS))),
  "ClusterSelector": lambda: ClusterSelector(clusters=[i % 3 for i in range(_N_ITEMS)]),
  "AStratSelector": lambda: AStratSelector(test_size=10),
  "AStratBBlockSelector": lambda: AStratBBlockSelector(test_size=10),
  "MaxInfoStratSelector": lambda: MaxInfoStratSelector(test_size=10),
  "MaxInfoBBlockSelector": lambda: MaxInfoBBlockSelector(test_size=10),
}


@pytest.mark.parametrize("make_selector", SELECTOR_FACTORIES.values(), ids=SELECTOR_FACTORIES.keys())
def test_select_returns_builtin_int(make_selector) -> None:
  """select() promises ``int | None``; numpy integers must not leak into session state."""
  item_bank = ItemBank.generate_item_bank(_N_ITEMS, seed=3)

  item_id = make_selector().select(
    item_bank=item_bank,
    administered_items=[0, 1],
    est_theta=0.3,
    rng=np.random.default_rng(0),
  )

  assert type(item_id) is int
