"""Tests for the ItemBank class."""

import numpy as np
import pytest

from catsim import ItemBank, irt


class TestItemBankCreation:
  """Tests for ItemBank initialization and creation."""

  def test_create_from_4pl_params(self) -> None:
    """Test creating ItemBank from 4PL parameters."""
    items = np.array([
      [1.0, 0.0, 0.0, 1.0],
      [1.2, -0.5, 0.1, 1.0],
      [0.8, 1.0, 0.05, 0.95],
    ])
    bank = ItemBank(items)

    assert bank.n_items == 3
    assert bank.items.shape == (3, 5)
    assert bank.model == 4  # 4PL due to d parameter != 1 in third item

  def test_create_from_3pl_params(self) -> None:
    """Test creating ItemBank from 3PL parameters."""
    items = np.array([
      [1.0, 0.0, 0.0],
      [1.2, -0.5, 0.1],
    ])
    bank = ItemBank(items)

    assert bank.n_items == 2
    assert bank.items.shape == (2, 5)
    # d should be added as 1.0
    assert np.all(bank.upper_asymptote == 1.0)

  def test_create_from_2pl_params(self) -> None:
    """Test creating ItemBank from 2PL parameters."""
    items = np.array([
      [1.0, 0.0],
      [1.2, -0.5],
    ])
    bank = ItemBank(items)

    assert bank.n_items == 2
    assert bank.model == 2  # 2PL
    # c should be 0, d should be 1
    assert np.all(bank.pseudo_guessing == 0.0)
    assert np.all(bank.upper_asymptote == 1.0)

  def test_create_from_1pl_params(self) -> None:
    """Test creating ItemBank from 1PL (difficulty only) parameters."""
    items = np.array([[0.0], [-0.5], [1.0]])
    bank = ItemBank(items)

    assert bank.n_items == 3
    assert bank.model == 1  # 1PL (Rasch)
    # a should be 1, c should be 0, d should be 1
    assert np.all(bank.discrimination == 1.0)
    assert np.all(bank.pseudo_guessing == 0.0)
    assert np.all(bank.upper_asymptote == 1.0)

  def test_exposure_rates_initialized_to_zero(self) -> None:
    """Test that exposure rates are initialized to zero."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)

    assert np.all(bank.exposure_rates == 0.0)

  def test_validation_enabled_by_default(self) -> None:
    """Test that validation is enabled by default and catches invalid params."""
    # Invalid: discrimination < 0
    items = np.array([[-1.0, 0.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="discrimination < 0"):
      ItemBank(items)

  def test_validation_can_be_disabled(self) -> None:
    """Test that validation can be disabled."""
    # Invalid but should not raise when validate=False
    items = np.array([[-1.0, 0.0, 0.0, 1.0]])

    bank = ItemBank(items, validate=False)
    assert bank.n_items == 1


class TestItemBankGenerate:
  """Tests for ItemBank.generate() factory method."""

  def test_generate_4pl(self) -> None:
    """Test generating 4PL item bank."""
    bank = ItemBank.generate(50, item_type="4PL", seed=42)

    assert bank.n_items == 50
    assert bank.model == 4
    # Check distributions are reasonable
    assert 0.5 < np.mean(bank.discrimination) < 2.0
    assert -2 < np.mean(bank.difficulty) < 2
    assert np.min(bank.pseudo_guessing) >= 0
    assert 0.94 <= np.min(bank.upper_asymptote) <= 1.0

  def test_generate_3pl(self) -> None:
    """Test generating 3PL item bank."""
    bank = ItemBank.generate(30, item_type="3PL", seed=42)

    assert bank.n_items == 30
    assert bank.model == 3
    assert np.all(bank.upper_asymptote == 1.0)
    assert np.any(bank.pseudo_guessing > 0)

  def test_generate_2pl(self) -> None:
    """Test generating 2PL item bank."""
    bank = ItemBank.generate(20, item_type="2PL", seed=42)

    assert bank.n_items == 20
    assert bank.model == 2
    assert np.all(bank.pseudo_guessing == 0.0)
    assert np.all(bank.upper_asymptote == 1.0)

  def test_generate_1pl(self) -> None:
    """Test generating 1PL (Rasch) item bank."""
    bank = ItemBank.generate(15, item_type="1PL", seed=42)

    assert bank.n_items == 15
    assert bank.model == 1
    assert np.all(bank.discrimination == 1.0)
    assert np.all(bank.pseudo_guessing == 0.0)
    assert np.all(bank.upper_asymptote == 1.0)

  def test_generate_with_correlation(self) -> None:
    """Test generating items with correlated a and b parameters."""
    bank = ItemBank.generate(100, item_type="2PL", corr=0.5, seed=42)

    # Check that there's positive correlation
    correlation = np.corrcoef(bank.discrimination, bank.difficulty)[0, 1]
    assert correlation > 0.3  # Should be somewhat positive

  def test_generate_case_insensitive(self) -> None:
    """Test that item_type strings are case insensitive."""
    bank1 = ItemBank.generate(10, item_type="3pl", seed=42)
    bank2 = ItemBank.generate(10, item_type="3PL", seed=42)
    bank3 = ItemBank.generate(10, item_type="3Pl", seed=42)

    assert bank1.model == bank2.model == bank3.model == 3

  def test_generate_invalid_type_raises(self) -> None:
    """Test that invalid item_type raises ValueError."""
    with pytest.raises(ValueError, match="Invalid itemtype"):
      ItemBank.generate(10, item_type="5PL")

  def test_generate_with_num_params_enum(self) -> None:
    """Test generating with irt.NumParams enum."""
    bank = ItemBank.generate(10, item_type=irt.NumParams.PL3, seed=42)
    assert bank.model == 3


class TestItemBankProperties:
  """Tests for ItemBank properties."""

  def test_discrimination_property(self) -> None:
    """Test discrimination property returns column 0."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    assert np.array_equal(bank.discrimination, np.array([1.0, 1.5]))

  def test_difficulty_property(self) -> None:
    """Test difficulty property returns column 1."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    assert np.array_equal(bank.difficulty, np.array([0.0, 1.0]))

  def test_pseudo_guessing_property(self) -> None:
    """Test pseudo_guessing property returns column 2."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    assert np.array_equal(bank.pseudo_guessing, np.array([0.0, 0.1]))

  def test_upper_asymptote_property(self) -> None:
    """Test upper_asymptote property returns column 3."""
    items = np.array([[1.0, 0.0, 0.0, 0.95], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    assert np.array_equal(bank.upper_asymptote, np.array([0.95, 1.0]))

  def test_exposure_rates_property(self) -> None:
    """Test exposure_rates property returns column 4."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)
    bank.items[0, 4] = 0.5

    assert bank.exposure_rates[0] == 0.5

  def test_model_property(self) -> None:
    """Test model property returns correct IRT model."""
    items_1pl = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    items_2pl = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.0, 1.0]])
    items_3pl = np.array([[1.0, 0.0, 0.1, 1.0], [1.5, 1.0, 0.05, 1.0]])
    items_4pl = np.array([[1.0, 0.0, 0.1, 0.98], [1.5, 1.0, 0.05, 0.95]])

    assert ItemBank(items_1pl).model == 1
    assert ItemBank(items_2pl).model == 2
    assert ItemBank(items_3pl).model == 3
    assert ItemBank(items_4pl).model == 4


class TestItemBankCachedValues:
  """Tests for cached/precomputed values."""

  def test_max_info_thetas_cached(self) -> None:
    """Test that max_info_thetas are precomputed and cached."""
    items = np.array([[1.0, 0.5, 0.0, 1.0], [1.2, -0.3, 0.1, 1.0]])
    bank = ItemBank(items)

    # Should be cached
    assert bank.max_info_thetas is not None
    assert len(bank.max_info_thetas) == 2

    # For 2PL item (c=0, d=1), max info theta should equal b
    assert abs(bank.max_info_thetas[0] - 0.5) < 0.01

  def test_max_info_values_cached(self) -> None:
    """Test that max_info_values are precomputed and cached."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)

    assert bank.max_info_values is not None
    assert len(bank.max_info_values) == 1
    # For a=1, b=0, c=0, d=1, max info should be 0.25
    assert abs(bank.max_info_values[0] - 0.25) < 0.01

  def test_cached_values_consistent_with_direct_computation(self) -> None:
    """Test that cached values match direct computation."""
    items = np.array([[1.2, 0.5, 0.05, 0.98], [0.9, -0.3, 0.0, 1.0]])
    bank = ItemBank(items)

    # Compute directly
    direct_max_thetas = irt.max_info_hpc(items)
    direct_max_info = irt.inf_hpc(direct_max_thetas, items)

    # Compare with cached
    np.testing.assert_array_almost_equal(bank.max_info_thetas, direct_max_thetas)
    np.testing.assert_array_almost_equal(bank.max_info_values, direct_max_info)


class TestItemBankMethods:
  """Tests for ItemBank methods."""

  def test_get_item(self) -> None:
    """Test getting a single item by index."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    item = bank.get_item(1)
    assert len(item) == 5
    assert item[0] == 1.5  # discrimination
    assert item[1] == 1.0  # difficulty

  def test_get_item_out_of_bounds_raises(self) -> None:
    """Test that out of bounds index raises IndexError."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)

    with pytest.raises(IndexError, match="out of bounds"):
      bank.get_item(5)

    with pytest.raises(IndexError, match="out of bounds"):
      bank.get_item(-1)

  def test_get_items(self) -> None:
    """Test getting multiple items by indices."""
    items = np.array([
      [1.0, 0.0, 0.0, 1.0],
      [1.5, 1.0, 0.1, 1.0],
      [0.8, -0.5, 0.05, 1.0],
    ])
    bank = ItemBank(items)

    selected = bank.get_items([0, 2])
    assert selected.shape == (2, 5)
    assert selected[0, 0] == 1.0
    assert selected[1, 0] == 0.8

  def test_update_exposure_rate(self) -> None:
    """Test updating exposure rate for a specific item."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)

    bank.update_exposure_rate(0, 0.75)
    assert bank.exposure_rates[0] == 0.75

  def test_update_exposure_rate_invalid_index_raises(self) -> None:
    """Test that invalid index raises IndexError."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)

    with pytest.raises(IndexError):
      bank.update_exposure_rate(5, 0.5)

  def test_update_exposure_rate_invalid_value_raises(self) -> None:
    """Test that invalid exposure rate raises ValueError."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)

    with pytest.raises(ValueError, match="between 0 and 1"):
      bank.update_exposure_rate(0, 1.5)

    with pytest.raises(ValueError, match="between 0 and 1"):
      bank.update_exposure_rate(0, -0.1)

  def test_reset_exposure_rates(self) -> None:
    """Test resetting all exposure rates to zero."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    # Set some exposure rates
    bank.update_exposure_rate(0, 0.5)
    bank.update_exposure_rate(1, 0.3)

    # Reset
    bank.reset_exposure_rates()
    assert np.all(bank.exposure_rates == 0.0)

  def test_icc_scalar_theta(self) -> None:
    """Test computing ICC for a scalar theta."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    probs = bank.icc(0.0)
    assert len(probs) == 2
    assert 0 <= probs[0] <= 1
    assert 0 <= probs[1] <= 1

  def test_icc_array_theta(self) -> None:
    """Test computing ICC for an array of thetas."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)

    thetas = np.array([-1.0, 0.0, 1.0])
    probs = bank.icc(thetas)
    assert probs.shape == (3, 1)
    # Probabilities should increase with theta for items with b=0
    assert probs[0, 0] < probs[1, 0] < probs[2, 0]

  def test_information_scalar_theta(self) -> None:
    """Test computing information for a scalar theta."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    info = bank.information(0.0)
    assert len(info) == 2
    assert info[0] > 0
    assert info[1] > 0

  def test_information_array_theta(self) -> None:
    """Test computing information for an array of thetas."""
    items = np.array([[1.0, 0.0, 0.0, 1.0]])
    bank = ItemBank(items)

    thetas = np.array([-1.0, 0.0, 1.0])
    info = bank.information(thetas)
    assert info.shape == (3, 1)
    # Information should be highest at theta=b=0
    assert info[1, 0] > info[0, 0]
    assert info[1, 0] > info[2, 0]

  def test_test_information_all_items(self) -> None:
    """Test computing test information using all items."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0]])
    bank = ItemBank(items)

    test_info = bank.test_information(0.0)
    # Should be sum of individual item information values
    expected = irt.test_info(0.0, items)
    assert abs(test_info - expected) < 0.001

  def test_test_information_subset_items(self) -> None:
    """Test computing test information using subset of items."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 0.0, 1.0], [1.0, -1.0, 0.0, 1.0]])
    bank = ItemBank(items)

    test_info = bank.test_information(0.0, item_indices=[0, 2])
    # Should only use items 0 and 2
    expected = irt.test_info(0.0, items[[0, 2], :])
    assert abs(test_info - expected) < 0.001


class TestItemBankReset:
  """Tests for ItemBank reset functionality."""

  def test_reset_exposure_rates(self) -> None:
    """Test that reset_exposure_rates clears exposure rates."""
    bank = ItemBank(np.array([[1.0, 0.0, 0.0, 1.0]]))

    # Set some exposure rates
    bank.update_exposure_rate(0, 0.5)
    assert bank.exposure_rates[0] == 0.5

    # Reset
    bank.reset_exposure_rates()
    assert bank.exposure_rates[0] == 0.0

  def test_reset_clears_exposure_rates(self) -> None:
    """Test that reset() clears all exposure rates."""
    bank = ItemBank.generate_item_bank(10)

    # Simulate some usage
    for i in range(10):
      bank.update_exposure_rate(i, 0.1 * (i + 1))

    # Verify exposure rates were set
    assert np.any(bank.exposure_rates > 0)

    # Reset
    bank.reset()

    # Verify all exposure rates are zero
    assert np.all(bank.exposure_rates == 0)

  def test_reset_preserves_parameters(self) -> None:
    """Test that reset() preserves item parameters."""
    bank = ItemBank.generate_item_bank(5, itemtype="3PL", seed=42)

    # Store original parameters
    original_params = bank.items[:, :4].copy()

    # Set exposure rates
    for i in range(5):
      bank.update_exposure_rate(i, 0.2)

    # Reset
    bank.reset()

    # Verify parameters unchanged
    np.testing.assert_array_equal(bank.items[:, :4], original_params)

  def test_reset_preserves_cached_properties(self) -> None:
    """Test that reset() preserves precomputed cached properties."""
    bank = ItemBank.generate_item_bank(10, seed=42)

    # Access cached properties to ensure they're computed
    max_info_thetas_before = bank.max_info_thetas.copy()
    max_info_values_before = bank.max_info_values.copy()

    # Reset
    bank.reset()

    # Verify cached properties are still available and unchanged
    np.testing.assert_array_equal(bank.max_info_thetas, max_info_thetas_before)
    np.testing.assert_array_equal(bank.max_info_values, max_info_values_before)

  def test_reset_multiple_times(self) -> None:
    """Test that reset() can be called multiple times."""
    bank = ItemBank.generate_item_bank(5)

    for _ in range(3):
      # Set exposure rates
      for i in range(5):
        bank.update_exposure_rate(i, 0.3)

      # Reset
      bank.reset()

      # Verify reset worked
      assert np.all(bank.exposure_rates == 0)


class TestItemBankDunderMethods:
  """Tests for ItemBank dunder methods."""

  def test_repr(self) -> None:
    """Test __repr__ method."""
    items = np.array([[1.0, 0.0, 0.1, 1.0]])
    bank = ItemBank(items)

    repr_str = repr(bank)
    assert "ItemBank" in repr_str
    assert "n_items=1" in repr_str
    assert "model=3" in repr_str

  def test_len(self) -> None:
    """Test __len__ method."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    assert len(bank) == 2

  def test_getitem_int(self) -> None:
    """Test __getitem__ with integer index."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0]])
    bank = ItemBank(items)

    item = bank[1]
    assert item[0] == 1.5

  def test_getitem_slice(self) -> None:
    """Test __getitem__ with slice."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0], [0.8, -0.5, 0.0, 1.0]])
    bank = ItemBank(items)

    subset = bank[0:2]
    assert subset.shape == (2, 5)

  def test_getitem_list(self) -> None:
    """Test __getitem__ with list of indices."""
    items = np.array([[1.0, 0.0, 0.0, 1.0], [1.5, 1.0, 0.1, 1.0], [0.8, -0.5, 0.0, 1.0]])
    bank = ItemBank(items)

    subset = bank[[0, 2]]
    assert subset.shape == (2, 5)
    assert subset[0, 0] == 1.0
    assert subset[1, 0] == 0.8
