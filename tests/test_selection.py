"""Tests for catsim.selection module."""

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
  LinearSelector,
  MaxInfoBBlockSelector,
  MaxInfoSelector,
  MaxInfoStratSelector,
  ProgressiveSelector,
  ProportionalSelector,
  RandomesqueSelector,
  RandomSelector,
  The54321Selector,
  UrrySelector,
)


class _DeterministicRNG:
  """Deterministic stand-in for numpy's Generator in selector tests."""

  def __init__(self, uniform_values: list[float] | np.ndarray) -> None:
    self._uniform_values = np.asarray(uniform_values, dtype=float)
    self.choice_calls: list[tuple[np.ndarray, np.ndarray | None]] = []

  def uniform(
    self,
    low: float = 0.0,  # noqa: ARG002
    high: float = 1.0,  # noqa: ARG002
    size: int | tuple[int, ...] | None = None,
  ) -> float | np.ndarray:
    if size is None:
      return float(self._uniform_values[0])
    expected_size = int(np.prod(size)) if isinstance(size, tuple) else int(size)
    if expected_size != self._uniform_values.size:
      msg = f"expected {expected_size} uniform values, got {self._uniform_values.size}"
      raise ValueError(msg)
    return self._uniform_values.copy()

  def choice(self, a: np.ndarray | list[int], p: np.ndarray | None = None) -> int:
    candidate_indices = np.asarray(a, dtype=int)
    probabilities = None if p is None else np.asarray(p, dtype=float)
    self.choice_calls.append((
      candidate_indices.copy(),
      probabilities.copy() if probabilities is not None else None,
    ))
    if probabilities is None:
      return int(candidate_indices[0])
    return int(candidate_indices[int(probabilities.argmax())])


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


class TestProgressiveSelector:
  """Tests for ProgressiveSelector."""

  def test_init_default(self) -> None:
    """Test default initialization."""
    selector = ProgressiveSelector(test_size=10)
    assert selector.test_size == 10
    assert str(selector) == "Progressive Selector (s=1.0)"

  def test_select_blends_random_and_information(self) -> None:
    """Test that early positions behave randomly and late positions use information."""
    item_bank = _progressive_item_bank()
    selector = ProgressiveSelector(test_size=3, acceleration=1.0)

    early_rng = _DeterministicRNG([0.1, 0.9, 0.2, 0.3])
    early_selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=early_rng,
    )
    assert early_selected == 1

    late_rng = _DeterministicRNG([0.9, 0.1, 0.2, 0.3])
    late_selected = selector.select(
      item_bank=item_bank,
      administered_items=[0, 1],
      est_theta=0.0,
      rng=late_rng,
    )
    assert late_selected == 3

  def test_select_respects_exposure_cap(self) -> None:
    """Test that the exposure cap is applied when possible."""
    item_bank = _progressive_item_bank()
    selector = ProgressiveSelector(test_size=3, acceleration=1.0, r_max=0.5)
    exposure_rates = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[0, 1],
      est_theta=0.0,
      rng=_DeterministicRNG([0.1, 0.2, 0.3, 0.4]),
      exposure_rates=exposure_rates,
    )

    assert selected == 2


class TestProportionalSelector:
  """Tests for ProportionalSelector."""

  def test_init_default(self) -> None:
    """Test default initialization."""
    selector = ProportionalSelector(test_size=10)
    assert selector.test_size == 10
    assert str(selector) == "Proportional Selector (s=1.0, k=6.0)"

  def test_select_uses_weighted_information(self) -> None:
    """Test that selection probabilities follow the weighted information formula."""
    item_bank = _progressive_item_bank()
    selector = ProportionalSelector(test_size=3, acceleration=1.0, sharpness=2.0)
    rng = _DeterministicRNG([0.4, 0.3, 0.2, 0.1])

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[0, 1],
      est_theta=0.0,
      rng=rng,
    )

    assert selected == 3
    assert len(rng.choice_calls) == 1
    candidates, probs = rng.choice_calls[0]
    assert np.array_equal(candidates, np.array([2, 3]))
    assert probs is not None

    info = item_bank.information(0.0)[candidates]
    position = len([0, 1]) + 1
    exponent = 2.0 * ((position - 1) / (selector.test_size - 1))
    expected = np.power(np.maximum(info, 0.0), exponent)
    expected /= expected.sum()
    np.testing.assert_allclose(probs, expected)

  def test_select_can_fall_back_to_uniform(self) -> None:
    """Test that zero sharpness produces a uniform distribution."""
    item_bank = _progressive_item_bank()
    selector = ProportionalSelector(test_size=3, sharpness=0.0)
    rng = _DeterministicRNG([0.9, 0.8, 0.7, 0.6])

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      rng=rng,
    )

    assert selected == 0
    _, probs = rng.choice_calls[0]
    assert probs is not None
    np.testing.assert_allclose(probs, np.full(4, 0.25))


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
