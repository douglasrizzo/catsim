"""Tests for catsim.initialization module."""

import numpy as np
import pytest

from catsim.initialization import (
  BaseInitializer,
  FixedPointInitializer,
  InitializationDistribution,
  RandomInitializer,
)


class TestInitializationDistribution:
  """Tests for InitializationDistribution enum."""

  def test_uniform_value(self) -> None:
    """Test uniform distribution value."""
    assert InitializationDistribution.UNIFORM.value == "uniform"

  def test_normal_value(self) -> None:
    """Test normal distribution value."""
    assert InitializationDistribution.NORMAL.value == "normal"


class TestRandomInitializerInit:
  """Tests for RandomInitializer initialization."""

  def test_init_default(self) -> None:
    """Test default initialization."""
    initializer = RandomInitializer()
    assert str(initializer) == "Random Initializer"

  def test_init_with_uniform(self) -> None:
    """Test initialization with uniform distribution."""
    initializer = RandomInitializer(
      dist_type=InitializationDistribution.UNIFORM,
      dist_params=(-4, 4),
    )
    assert initializer is not None

  def test_init_with_normal(self) -> None:
    """Test initialization with normal distribution."""
    initializer = RandomInitializer(
      dist_type=InitializationDistribution.NORMAL,
      dist_params=(0, 1),
    )
    assert initializer is not None

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

  def test_initialize_uniform_in_range(self) -> None:
    """Test that uniform initialization produces values in range."""
    rng = np.random.default_rng(42)
    initializer = RandomInitializer(
      dist_type=InitializationDistribution.UNIFORM,
      dist_params=(-3, 3),
    )

    for _ in range(100):
      theta = initializer.initialize(rng=rng)
      assert -3 <= theta <= 3

  def test_initialize_uniform_reversed_params(self) -> None:
    """Test that uniform handles reversed params correctly."""
    rng = np.random.default_rng(42)
    initializer = RandomInitializer(
      dist_type=InitializationDistribution.UNIFORM,
      dist_params=(3, -3),  # Reversed order
    )

    for _ in range(100):
      theta = initializer.initialize(rng=rng)
      assert -3 <= theta <= 3

  def test_initialize_normal_distribution(self) -> None:
    """Test that normal initialization produces reasonable values."""
    rng = np.random.default_rng(42)
    initializer = RandomInitializer(
      dist_type=InitializationDistribution.NORMAL,
      dist_params=(0, 1),
    )

    values = [initializer.initialize(rng=rng) for _ in range(1000)]

    # Check mean is close to 0
    assert abs(np.mean(values)) < 0.1

    # Check std is close to 1
    assert abs(np.std(values) - 1) < 0.1

  def test_initialize_reproducibility_with_rng(self) -> None:
    """Test that same RNG produces same results."""
    initializer = RandomInitializer()

    rng1 = np.random.default_rng(42)
    values1 = [initializer.initialize(rng=rng1) for _ in range(10)]

    rng2 = np.random.default_rng(42)
    values2 = [initializer.initialize(rng=rng2) for _ in range(10)]

    assert values1 == values2


class TestFixedPointInitializerInit:
  """Tests for FixedPointInitializer initialization."""

  def test_init_with_zero(self) -> None:
    """Test initialization with zero starting point."""
    initializer = FixedPointInitializer(0.0)
    assert str(initializer) == "Fixed Point Initializer"

  def test_init_with_positive(self) -> None:
    """Test initialization with positive starting point."""
    initializer = FixedPointInitializer(1.5)
    assert initializer is not None

  def test_init_with_negative(self) -> None:
    """Test initialization with negative starting point."""
    initializer = FixedPointInitializer(-2.0)
    assert initializer is not None


class TestFixedPointInitializerInitialize:
  """Tests for FixedPointInitializer.initialize() method."""

  def test_initialize_returns_fixed_value(self) -> None:
    """Test that initialize always returns the fixed value."""
    start = 1.5
    initializer = FixedPointInitializer(start)

    for _ in range(10):
      theta = initializer.initialize()
      assert theta == start

  def test_initialize_ignores_index(self) -> None:
    """Test that initialize ignores the index parameter."""
    initializer = FixedPointInitializer(0.0)

    for i in range(10):
      theta = initializer.initialize(index=i)
      assert theta == 0.0

  def test_initialize_zero(self) -> None:
    """Test initialization with zero."""
    initializer = FixedPointInitializer(0.0)
    assert initializer.initialize() == 0.0

  def test_initialize_negative(self) -> None:
    """Test initialization with negative value."""
    initializer = FixedPointInitializer(-3.0)
    assert initializer.initialize() == -3.0


class TestBaseInitializerAbstract:
  """Tests for BaseInitializer abstract base class."""

  def test_cannot_instantiate_base_initializer(self) -> None:
    """Test that BaseInitializer cannot be instantiated directly."""
    with pytest.raises(TypeError):
      BaseInitializer()  # type: ignore[abstract]

  def test_subclass_must_implement_initialize(self) -> None:
    """Test that subclass must implement initialize method."""

    class IncompleteInitializer(BaseInitializer):
      pass

    with pytest.raises(TypeError):
      IncompleteInitializer()  # type: ignore[abstract]

  def test_custom_initializer(self) -> None:
    """Test creating a custom initializer."""

    class ConstantInitializer(BaseInitializer):
      def initialize(self, **_kwargs: object) -> float:
        return 42.0

    initializer = ConstantInitializer()
    assert initializer.initialize() == 42.0
