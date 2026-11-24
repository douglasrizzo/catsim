"""Tests for catsim.initialization module."""

import importlib
import inspect
import math

import numpy as np
import pytest

from catsim.initialization import FixedPointInitializer, NormalInitializer, UniformInitializer
from catsim.initialization.base import BaseInitializer
from catsim.item_bank import ItemBank

NAN = float("nan")
INF = float("inf")
# Enough draws for the sample mean and sd to land within 4 standard errors of their closed forms;
# each statistic misses by chance with probability below 1e-4.
N_DRAWS = 10_000


def sample(initializer: BaseInitializer, seed: int = 42, n: int = N_DRAWS) -> np.ndarray:
  """Draw ``n`` initial abilities from one generator seeded with ``seed``."""
  item_bank = ItemBank.generate(10, rng=42)
  rng = np.random.default_rng(seed)
  return np.array([initializer.initialize(item_bank=item_bank, rng=rng) for _ in range(n)])


class TestUniformInitializerInit:
  """Tests for UniformInitializer construction."""

  @pytest.mark.parametrize(("low", "high"), [(0.0, 0.0), (1.0, -1.0)])
  def test_init_empty_or_reversed_interval_raises(self, low: float, high: float) -> None:
    """An empty interval was accepted, or reversed bounds were silently swapped instead of rejected."""
    with pytest.raises(ValueError, match="high must be greater than low"):
      UniformInitializer(low=low, high=high)

  @pytest.mark.parametrize(("low", "high"), [(NAN, 1.0), (0.0, NAN), (-INF, 0.0), (0.0, INF)])
  def test_init_non_finite_bound_raises(self, low: float, high: float) -> None:
    """A NaN or infinite bound was accepted and failed only at draw time, with NumPy's OverflowError."""
    with pytest.raises(ValueError, match="low and high must be finite"):
      UniformInitializer(low=low, high=high)


class TestNormalInitializerInit:
  """Tests for NormalInitializer construction."""

  @pytest.mark.parametrize("mean", [NAN, INF, -INF])
  def test_init_non_finite_mean_raises(self, mean: float) -> None:
    """A NaN or infinite mean was accepted, and every draw was NaN or infinite."""
    with pytest.raises(ValueError, match="mean must be finite"):
      NormalInitializer(mean=mean)

  @pytest.mark.parametrize("sd", [0.0, -1.0, NAN, INF])
  def test_init_invalid_sd_raises(self, sd: float) -> None:
    """A zero, negative, NaN or infinite sd was accepted; NaN and infinity then drew NaN or infinite abilities."""
    with pytest.raises(ValueError, match="sd must be positive and finite"):
      NormalInitializer(sd=sd)


class TestRandomInitializersInitialize:
  """Tests for the draws of UniformInitializer and NormalInitializer."""

  @pytest.mark.parametrize(
    ("initializer", "low", "high"),
    [(UniformInitializer(), -0.5, 0.5), (UniformInitializer(low=-3.0, high=1.0), -3.0, 1.0)],
    ids=["default", "custom"],
  )
  def test_uniform_draws_cover_the_interval(self, initializer: UniformInitializer, low: float, high: float) -> None:
    """Draws left the interval or did not spread uniformly over it, or the default is not Barrada's U(-0.5, 0.5)."""
    draws = sample(initializer)
    width = high - low
    # U(low, high) has mean (low + high) / 2 and sd width / sqrt(12). Over n draws the sample mean's
    # standard error is width / sqrt(12 n) and the sample sd's is width / sqrt(60 n).
    mean_se = width / math.sqrt(12 * N_DRAWS)
    sd_se = width / math.sqrt(60 * N_DRAWS)

    assert low <= draws.min() < draws.max() <= high
    assert draws.mean() == pytest.approx((low + high) / 2, abs=4 * mean_se)
    assert draws.std() == pytest.approx(width / math.sqrt(12), abs=4 * sd_se)

  @pytest.mark.parametrize(
    ("initializer", "mean", "sd"),
    [(NormalInitializer(), 0.0, 1.0), (NormalInitializer(mean=2.0, sd=0.5), 2.0, 0.5)],
    ids=["default", "custom"],
  )
  def test_normal_draws_have_the_configured_mean_and_sd(
    self, initializer: NormalInitializer, mean: float, sd: float
  ) -> None:
    """Draws had the wrong centre or spread (mean and sd swapped, variance used as sd), or the default isn't N(0, 1)."""
    draws = sample(initializer)
    # Over n draws the sample mean's standard error is sd / sqrt(n) and the sample sd's about sd / sqrt(2 n).
    mean_se = sd / math.sqrt(N_DRAWS)
    sd_se = sd / math.sqrt(2 * N_DRAWS)

    assert draws.mean() == pytest.approx(mean, abs=4 * mean_se)
    assert draws.std() == pytest.approx(sd, abs=4 * sd_se)

  @pytest.mark.parametrize("initializer", [UniformInitializer(), NormalInitializer()], ids=["uniform", "normal"])
  def test_initialize_draws_from_the_given_rng(self, initializer: BaseInitializer) -> None:
    """The initializer drew from another source (NumPy's global state, a generator of its own) instead of ``rng``."""
    first = sample(initializer, seed=7, n=10)

    assert np.array_equal(first, sample(initializer, seed=7, n=10))
    assert not np.array_equal(first, sample(initializer, seed=8, n=10))


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
    item_bank = ItemBank.generate(10, rng=42)
    theta = FixedPointInitializer(start).initialize(item_bank=item_bank, rng=np.random.default_rng(42))

    assert type(theta) is float
    assert theta == expected


INITIALIZER_REPRS = {
  "UniformInitializer": "UniformInitializer(low=-1.0, high=2.0)",
  "NormalInitializer": "NormalInitializer(mean=0.5, sd=2.0)",
  "FixedPointInitializer": "FixedPointInitializer(start=-0.25)",
}


@pytest.mark.parametrize("source", INITIALIZER_REPRS.values(), ids=INITIALIZER_REPRS.keys())
def test_repr_is_the_constructor_call(source: str) -> None:
  """An initializer's repr names every constructor argument with its value, so evaluating it rebuilds the initializer.

  Initializers printed only a display name such as "Uniform Initializer", and their default repr showed an object
  address, so a printed strategy did not say how it was configured.
  """
  namespace = vars(importlib.import_module("catsim.initialization"))

  assert repr(eval(source, namespace)) == source


def test_repr_table_covers_every_exported_initializer() -> None:
  """An initializer left out of INITIALIZER_REPRS could hide an argument from its repr without any test noticing."""
  exported = {
    name
    for name, cls in vars(importlib.import_module("catsim.initialization")).items()
    if isinstance(cls, type) and issubclass(cls, BaseInitializer) and not inspect.isabstract(cls)
  }

  assert exported == set(INITIALIZER_REPRS)
