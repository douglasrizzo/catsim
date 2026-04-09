"""Tests for catsim.selection module."""

import importlib

import numpy as np
import pytest

from catsim.exceptions import NoItemsAvailableError
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  BaseSelector,
  ClusterSelector,
  FiniteSelector,
  IntervalInfoSelector,
  KLSelector,
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
    assert str(selector) == "Maximum Information Selector"

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

  def test_init_with_bin_size(self) -> None:
    """Test initialization with bin size."""
    selector = RandomesqueSelector(5)
    assert str(selector) == "Randomesque Selector"

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
    assert str(selector) == "Interval Information Selector"

  def test_init_default(self) -> None:
    """Test default initialization (infinite interval)."""
    selector = IntervalInfoSelector()
    assert selector.interval == np.inf

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


class TestKLSelector:
  """Tests for KLSelector."""

  @staticmethod
  def _bank() -> ItemBank:
    """Create a small bank with a clear central item."""
    return ItemBank(
      np.array([
        [1.0, -2.0, 0.0, 1.0],
        [1.0, 0.0, 0.0, 1.0],
        [1.0, 2.0, 0.0, 1.0],
      ])
    )

  def test_init_default(self) -> None:
    """Test default initialization."""
    selector = KLSelector()
    assert selector.c == pytest.approx(3.0)
    assert selector.r_max == pytest.approx(1.0)
    assert str(selector) == "Kullback-Leibler Selector (c=3.0)"

  def test_kl_integrand_is_zero_at_theta_hat(self) -> None:
    """Test that the KL integrand vanishes when theta equals theta_hat."""
    kl_module = importlib.import_module("catsim.selection.kl")
    value = kl_module.__dict__["_kl_integrand"](0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    assert value == pytest.approx(0.0, abs=1e-12)

  def test_wider_integration_window_accumulates_more_kl_mass(self) -> None:
    """Test that a wider integration window yields more KL mass."""
    item = self._bank().items[1]
    kl_global = KLSelector.__dict__["_kl_global"].__func__
    narrow = kl_global(theta_hat=0.0, item=item, half_width=0.25)
    wide = kl_global(theta_hat=0.0, item=item, half_width=1.0)

    assert narrow >= 0.0
    assert wide > narrow

  def test_select_uses_half_width_c_over_sqrt_administered(
    self,
    monkeypatch: pytest.MonkeyPatch,
  ) -> None:
    """Test that select uses c / sqrt(max(1, n_administered)) for the window width.

    This spies on `_kl_global` instead of the local `half_width` variable so the
    assertion stays tied to the public `select()` contract while remaining stable.
    """
    item_bank = self._bank()
    selector = KLSelector(c=2.0)
    observed_half_widths: list[float] = []

    def spy(_theta_hat: float, item: np.ndarray, half_width: float) -> float:
      observed_half_widths.append(half_width)
      return float(2.0 - abs(float(item[1])))

    monkeypatch.setattr(KLSelector, "_kl_global", staticmethod(spy))

    selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )
    selector.select(
      item_bank=item_bank,
      administered_items=[0, 1],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert all(half_width == pytest.approx(2.0) for half_width in observed_half_widths[: item_bank.n_items])
    assert all(
      half_width == pytest.approx(2.0 / np.sqrt(2.0)) for half_width in observed_half_widths[item_bank.n_items :]
    )

  def test_select_prefers_central_item(self) -> None:
    """Test that KL selection prefers the item centered on the current theta."""
    item_bank = self._bank()
    selector = KLSelector(c=0.5)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=np.random.default_rng(42),
    )

    assert selected == 1

  def test_select_excludes_administered_and_honors_exposure_cap(self) -> None:
    """Test that selection excludes administered items and falls back when cap blocks all candidates."""
    item_bank = self._bank()
    selector = KLSelector(c=0.5, r_max=0.0)
    exposure_rates = np.ones(item_bank.n_items, dtype=float)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[1],
      est_theta=0.0,
      rng=np.random.default_rng(42),
      exposure_rates=exposure_rates,
    )

    assert selected is not None
    assert selected != 1
    assert selected in {0, 2}

  def test_select_raises_when_exhausted(self) -> None:
    """Test that select raises NoItemsAvailableError when all items are administered."""
    item_bank = self._bank()
    selector = KLSelector()

    with pytest.raises(NoItemsAvailableError, match="no more items"):
      selector.select(
        item_bank=item_bank,
        administered_items=[0, 1, 2],
        est_theta=0.0,
        rng=np.random.default_rng(42),
      )


class TestClusterSelector:
  """Tests for ClusterSelector."""

  def test_init_with_clusters(self) -> None:
    """Test initialization with clusters."""
    clusters = [0, 0, 1, 1, 2, 2, 3, 3]
    selector = ClusterSelector(clusters=clusters)
    assert selector is not None

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


class TestAllSelectorsBasicFunctionality:
  """Integration tests for all selectors."""

  @pytest.fixture
  def item_bank(self) -> ItemBank:
    """Create a test item bank."""
    return ItemBank.generate_item_bank(100, seed=42)

  @pytest.fixture
  def rng(self) -> np.random.Generator:
    """Create a random number generator."""
    return np.random.default_rng(42)

  def test_max_info_selector(self, item_bank: ItemBank, rng: np.random.Generator) -> None:
    """Test MaxInfoSelector in sequence."""
    selector = MaxInfoSelector()
    administered: list[int] = []

    for _ in range(10):
      selected = selector.select(
        item_bank=item_bank,
        administered_items=administered,
        est_theta=0.0,
        rng=rng,
      )
      assert selected is not None
      assert selected not in administered
      administered.append(selected)

  def test_random_selector(self, item_bank: ItemBank, rng: np.random.Generator) -> None:
    """Test RandomSelector in sequence."""
    selector = RandomSelector()
    administered: list[int] = []

    for _ in range(10):
      selected = selector.select(
        item_bank=item_bank,
        administered_items=administered,
        est_theta=0.0,
        rng=rng,
      )
      assert selected is not None
      assert selected not in administered
      administered.append(selected)

  def test_linear_selector_sequence(self, item_bank: ItemBank, rng: np.random.Generator) -> None:
    """Test LinearSelector returns items in exact order."""
    indices = [0, 10, 20, 30, 40]
    selector = LinearSelector(indices)
    administered: list[int] = []

    for expected_idx in indices:
      selected = selector.select(
        item_bank=item_bank,
        administered_items=administered,
        est_theta=0.0,
        rng=rng,
      )
      assert selected == expected_idx
      administered.append(selected)
