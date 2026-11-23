"""Tests for catsim.initialization module."""

import numpy as np
import pytest

from catsim.initialization import (
  FixedPointInitializer,
  InitializationDistribution,
  RandomInitializer,
)
from catsim.item_bank import ItemBank


class TestRandomInitializerInit:
  """Tests for RandomInitializer initialization."""

  def test_init_invalid_dist_type_raises(self) -> None:
    """Test that invalid dist_type raises TypeError."""
    with pytest.raises(TypeError, match="must be of type InitializationDistribution"):
      RandomInitializer(dist_type="uniform")  # type: ignore[arg-type]

  def test_init_invalid_dist_params_not_tuple_raises(self) -> None:
    """Test that non-tuple dist_params raises ValueError."""
    with pytest.raises(ValueError, match="tuple of exactly 2 values"):
      RandomInitializer(dist_params=[-4, 4])  # type: ignore[arg-type]

  def test_init_invalid_dist_params_wrong_length_raises(self) -> None:
    """Test that wrong length dist_params raises ValueError."""
    with pytest.raises(ValueError, match="tuple of exactly 2 values"):
      RandomInitializer(dist_params=(-4, 0, 4))

  def test_init_uniform_equal_bounds_raises(self) -> None:
    """Test that uniform with equal bounds raises ValueError."""
    with pytest.raises(ValueError, match="different min and max values"):
      RandomInitializer(
        dist_type=InitializationDistribution.UNIFORM,
        dist_params=(0, 0),
      )

  def test_init_normal_nonpositive_std_raises(self) -> None:
    """Test that normal with non-positive std raises ValueError."""
    with pytest.raises(ValueError, match="standard deviation must be positive"):
      RandomInitializer(
        dist_type=InitializationDistribution.NORMAL,
        dist_params=(0, 0),
      )

    with pytest.raises(ValueError, match="standard deviation must be positive"):
      RandomInitializer(
        dist_type=InitializationDistribution.NORMAL,
        dist_params=(0, -1),
      )


class TestRandomInitializerInitialize:
  """Tests for RandomInitializer.initialize() method."""

  def test_initialize_defaults_start_near_the_center(self) -> None:
    """Defaults draw from U(-0.5, 0.5), as in Barrada et al. (2006), or N(0, 1); they used to be U(-5, 5)."""
    rng = np.random.default_rng(42)
    item_bank = ItemBank.generate_item_bank(10, seed=42)

    uniform = [RandomInitializer().initialize(item_bank=item_bank, rng=rng) for _ in range(1000)]
    normal_initializer = RandomInitializer(dist_type=InitializationDistribution.NORMAL)
    normal = [normal_initializer.initialize(item_bank=item_bank, rng=rng) for _ in range(5000)]

    assert -0.5 <= min(uniform) < max(uniform) <= 0.5
    assert np.mean(normal) == pytest.approx(0.0, abs=0.05)
    assert np.std(normal) == pytest.approx(1.0, abs=0.05)

  def test_initialize_uniform_reversed_params(self) -> None:
    """Test that uniform handles reversed params correctly."""
    rng = np.random.default_rng(42)
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    initializer = RandomInitializer(
      dist_type=InitializationDistribution.UNIFORM,
      dist_params=(3, -3),  # Reversed order
    )

    for _ in range(100):
      theta = initializer.initialize(item_bank=item_bank, rng=rng)
      assert -3 <= theta <= 3

  def test_initialize_reproducibility_with_rng(self) -> None:
    """Test that same RNG produces same results."""
    initializer = RandomInitializer()
    item_bank = ItemBank.generate_item_bank(10, seed=42)

    rng1 = np.random.default_rng(42)
    values1 = [initializer.initialize(item_bank=item_bank, rng=rng1) for _ in range(10)]

    rng2 = np.random.default_rng(42)
    values2 = [initializer.initialize(item_bank=item_bank, rng=rng2) for _ in range(10)]

    assert values1 == values2


class TestFixedPointInitializerInit:
  """Tests for FixedPointInitializer construction."""

  @pytest.mark.parametrize("start", [float("nan"), float("inf"), -float("inf")])
  def test_init_non_finite_start_raises(self, start: float) -> None:
    """A NaN or infinite start was accepted and became every session's first estimate."""
    with pytest.raises(ValueError, match="start must be finite"):
      FixedPointInitializer(start)

  @pytest.mark.parametrize("start", ["abc", None])
  def test_init_non_numeric_start_raises(self, start: object) -> None:
    """A non-numeric start was accepted and returned as the initial estimate, despite the documented TypeError."""
    with pytest.raises(TypeError, match="must be real number"):
      FixedPointInitializer(start)  # type: ignore[arg-type]


class TestFixedPointInitializerInitialize:
  """Tests for FixedPointInitializer.initialize() method."""

  @pytest.mark.parametrize(("start", "expected"), [(1.5, 1.5), (0, 0.0), (np.float64(-1.25), -1.25)])
  def test_initialize_returns_start_as_builtin_float(self, start: float, expected: float) -> None:
    """The start was returned as given, so an int or a NumPy scalar reached the session's theta history."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    theta = FixedPointInitializer(start).initialize(item_bank=item_bank, rng=np.random.default_rng(42))

    assert type(theta) is float
    assert theta == expected
