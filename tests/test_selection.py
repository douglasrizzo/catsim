"""Tests for catsim.selection module."""

import importlib
import inspect

import numpy as np
import pytest

from catsim import irt
from catsim.estimation import BaseEstimator
from catsim.estimation.numerical import NumericalSearchEstimator
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
  MEISelector,
  MLWISelector,
  ProgressiveSelector,
  ProportionalSelector,
  RandomesqueSelector,
  RandomSelector,
  The54321Selector,
  UrrySelector,
)


class RecordingEstimator(BaseEstimator):
  """Estimator stub that records the history it receives."""

  def __init__(self) -> None:
    super().__init__()
    self.records: list[tuple[list[int], list[bool], float]] = []

  def estimate(
    self,
    item_bank: ItemBank,
    administered_items: list[int],
    response_vector: list[bool],
    est_theta: float,
  ) -> float:
    del item_bank
    self.records.append((list(administered_items), list(response_vector), est_theta))
    return float(len(response_vector))


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


class TestMEISelector:
  """Tests for MEISelector."""

  def test_select_requires_response_vector(self) -> None:
    """The selector needs the current response history."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    selector = MEISelector()

    with pytest.raises(ValueError, match="response_vector"):
      selector.select(
        item_bank=item_bank,
        administered_items=[],
        est_theta=0.0,
        rng=np.random.default_rng(42),
      )

  def test_select_forwards_response_history_to_estimator(self) -> None:
    """One-step-ahead scoring should append both hypothetical responses."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    estimator = RecordingEstimator()
    selector = MEISelector(estimator=estimator)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[0],
      est_theta=0.0,
      rng=np.random.default_rng(42),
      response_vector=[True],
    )

    assert selected is not None
    assert selected != 0
    assert estimator.records
    assert all(call[0][:1] == [0] for call in estimator.records)
    assert {tuple(call[1]) for call in estimator.records} == {(True, True), (True, False)}

  def test_select_falls_back_when_exposure_cap_blocks_all_candidates(self) -> None:
    """MEI should still choose from non-administered items when every candidate exceeds r_max."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    selector = MEISelector(estimator=RecordingEstimator(), r_max=0.0)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[0],
      est_theta=0.0,
      rng=np.random.default_rng(42),
      exposure_rates=np.ones(item_bank.n_items, dtype=float),
      response_vector=[True],
    )

    assert selected is not None
    assert selected != 0

  def test_select_diverges_from_max_info_when_one_step_estimates_shift(self) -> None:
    """MEI should not collapse to MaxInfo when the response history changes theta.

    This is the key behavioral check for a one-step-ahead selector: if MEI
    always agreed with MaxInfo here, it would be ignoring the estimated post-response shift.
    """
    # Frozen bank parameters captured from a verified divergence scenario.
    item_bank = ItemBank(
      np.array(
        [
          [1.16697378, 0.12573022, 0.23911482, 0.97692311],
          [1.22622503, 0.64042265, 0.243674, 0.96302065],
          [1.29039876, -0.53566937, 0.25823261, 0.9998326],
          [1.43677024, 1.30400005, 0.27085027, 0.99885012],
          [0.88364463, -0.70373524, 0.24742931, 0.98113252],
          [1.21033149, -0.62327446, 0.27732927, 0.97902756],
          [1.14530208, -2.32503077, 0.23669611, 0.9813068],
          [1.01693316, -1.24591095, 0.2570302, 0.96333529],
        ],
        dtype=float,
      )
    )
    administered_items = [0]
    response_vector = [True]
    est_theta = 0.0

    mei = MEISelector(estimator=NumericalSearchEstimator())
    max_info = MaxInfoSelector()

    mei_choice = mei.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      rng=np.random.default_rng(0),
      response_vector=response_vector,
    )
    max_info_choice = max_info.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      rng=np.random.default_rng(0),
    )

    assert mei_choice == 1
    assert max_info_choice == 2
    assert mei_choice != max_info_choice

  def test_select_matches_manual_expected_information(self) -> None:
    """MEI scores should match the manual P*I_correct + (1-P)*I_incorrect calculation.

    The manual score comparison makes sure both response branches are weighted
    correctly instead of merely exercising the estimator wiring.
    """
    item_bank = ItemBank(
      np.array(
        [
          [1.5, 0.0, 0.0, 1.0],
          [1.2, 0.5, 0.0, 1.0],
          [0.8, -0.5, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    administered_items = [0]
    response_vector = [True]
    est_theta = 0.4
    estimator = NumericalSearchEstimator()
    selector = MEISelector(estimator=estimator)

    candidate_mask = np.ones(item_bank.n_items, dtype=bool)
    candidate_mask[administered_items] = False
    candidates = np.flatnonzero(candidate_mask)
    manual_scores = np.array(
      [
        float(
          (
            irt.icc(est_theta, *item_bank.items[int(idx), :4])
            * item_bank.test_information(
              estimator.estimate(
                item_bank=item_bank,
                administered_items=[*administered_items, int(idx)],
                response_vector=[*response_vector, True],
                est_theta=est_theta,
              ),
              [*administered_items, int(idx)],
            )
          )
          + (
            (1.0 - irt.icc(est_theta, *item_bank.items[int(idx), :4]))
            * item_bank.test_information(
              estimator.estimate(
                item_bank=item_bank,
                administered_items=[*administered_items, int(idx)],
                response_vector=[*response_vector, False],
                est_theta=est_theta,
              ),
              [*administered_items, int(idx)],
            )
          )
        )
        for idx in candidates
      ],
      dtype=float,
    )

    selected = selector.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      rng=np.random.default_rng(0),
      response_vector=response_vector,
    )

    assert selected == 1
    assert selected == int(candidates[int(manual_scores.argmax())])


class TestMLWISelector:
  """Tests for MLWISelector."""

  def test_select_requires_response_vector(self) -> None:
    """The selector needs the current response history."""
    item_bank = ItemBank.generate_item_bank(10, seed=42)
    selector = MLWISelector()

    with pytest.raises(ValueError, match="response_vector"):
      selector.select(
        item_bank=item_bank,
        administered_items=[],
        est_theta=0.0,
        rng=np.random.default_rng(42),
      )

  def test_select_matches_manual_weighted_information(self) -> None:
    """Likelihood-weighted scores should match the manual integration."""
    item_bank = ItemBank.generate_item_bank(8, seed=42)
    selector = MLWISelector(n_nodes=11)
    administered_items = [0, 1]
    response_vector = [True, False]
    candidates = np.array([idx for idx in range(item_bank.n_items) if idx not in administered_items])

    selected = selector.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=0.0,
      rng=np.random.default_rng(42),
      response_vector=response_vector,
    )

    nodes = np.linspace(-6.0, 6.0, 11)
    delta = (6.0 - (-6.0)) / (11 - 1)
    log_likelihood = np.array(
      [irt.log_likelihood(float(theta), response_vector, item_bank.items[administered_items, :4]) for theta in nodes],
      dtype=float,
    )
    log_likelihood -= float(log_likelihood.max())
    weights = np.exp(log_likelihood)
    manual_scores = np.array(
      [
        float(
          np.sum(np.array([irt.inf(float(theta), *item_bank.items[int(idx), :4]) for theta in nodes]) * weights) * delta
        )
        for idx in candidates
      ],
      dtype=float,
    )
    expected = int(candidates[int(manual_scores.argmax())])

    assert selected == expected

  def test_select_falls_back_when_exposure_cap_blocks_all_candidates(self) -> None:
    """MLWI should still choose from non-administered items when every candidate exceeds r_max."""
    item_bank = ItemBank.generate_item_bank(8, seed=42)
    selector = MLWISelector(n_nodes=11, r_max=0.0)

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[0, 1],
      est_theta=0.0,
      rng=np.random.default_rng(42),
      exposure_rates=np.ones(item_bank.n_items, dtype=float),
      response_vector=[True, False],
    )

    assert selected is not None
    assert selected not in {0, 1}

  def test_select_with_empty_response_vector_reduces_to_information_integral(self) -> None:
    """With no responses, MLWI should reduce to the integral of item information.

    This verifies the no-data degeneracy case where likelihood weights become
    uniform; if that fallback changed, MLWI would stop matching its mathematical definition.
    """
    item_bank = ItemBank.generate_item_bank(8, seed=0)
    n_nodes = 21
    selector = MLWISelector(n_nodes=n_nodes)
    nodes = np.linspace(-6.0, 6.0, n_nodes)
    delta = (6.0 - (-6.0)) / (n_nodes - 1)

    manual_scores = np.array(
      [
        float(np.sum(np.array([irt.inf(float(theta), *item_bank.items[idx, :4]) for theta in nodes])) * delta)
        for idx in range(item_bank.n_items)
      ],
      dtype=float,
    )

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[],
      est_theta=0.0,
      response_vector=[],
    )

    assert selected == int(manual_scores.argmax())

  def test_select_converges_to_max_info_on_long_test(self) -> None:
    """On a long test, MLWI should agree with MaxInfo near the MLE.

    As the likelihood sharpens, MLWI should concentrate on the MLE neighborhood,
    so disagreement here would signal broken weighting or integration.
    """
    item_bank = ItemBank.generate_item_bank(30, seed=1)
    theta_true = 0.7
    rng = np.random.default_rng(5)
    administered_items = list(range(20))
    response_vector = [
      bool(rng.random() < irt.icc(theta_true, *item_bank.items[idx, :4])) for idx in administered_items
    ]
    mle_theta = NumericalSearchEstimator().estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=response_vector,
      est_theta=theta_true,
    )

    mlwi = MLWISelector(n_nodes=81)
    max_info = MaxInfoSelector()

    mlwi_choice = mlwi.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=mle_theta,
      response_vector=response_vector,
    )
    max_info_choice = max_info.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=mle_theta,
    )

    assert mlwi_choice == max_info_choice


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
    """Test that the KL integrand vanishes when theta equals theta_hat.

    Self-divergence must be zero, so this catches sign flips and argument-order
    mistakes in the KL formula before they distort the global integral.
    """
    kl_module = importlib.import_module("catsim.selection.kl")
    value = kl_module.__dict__["_kl_integrand"](0.0, 0.0, 1.0, 0.0, 0.0, 1.0)
    assert value == pytest.approx(0.0, abs=1e-12)

  def test_wider_integration_window_accumulates_more_kl_mass(self) -> None:
    """Test that a wider integration window yields more KL mass.

    This isolates the integration-window effect, which is the mechanism that
    makes KL behave globally early and more locally as the test progresses.
    """
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
    If the sqrt schedule or guard changed, KL could stop converging toward a
    local-information rule late in the test.
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

  def test_weight_is_zero_at_first_position(self) -> None:
    """At position 1 the blend weight must be exactly 0.

    This guards the pure-random starting condition; an off-by-one here would
    silently make the selector exposure-heavy from the first item.
    """
    selector = ProgressiveSelector(test_size=10, acceleration=1.0)
    assert selector._weight(1) == pytest.approx(0.0)  # noqa: SLF001

  def test_weight_is_one_at_last_position(self) -> None:
    """At the final position the blend weight must be exactly 1.

    This protects the asymptotic handoff to information-based selection at the
    end of a fixed-length test.
    """
    selector = ProgressiveSelector(test_size=10, acceleration=1.0)
    assert selector._weight(10) == pytest.approx(1.0)  # noqa: SLF001

  def test_weight_formula_linear_schedule(self) -> None:
    """Midpoint weight should be 0.5 with s=1 and test_size=9.

    The direct formula check catches exponent or indexing mistakes that a
    selection-outcome test could miss.
    """
    selector = ProgressiveSelector(test_size=9, acceleration=1.0)
    assert selector._weight(5) == pytest.approx(0.5)  # noqa: SLF001

  def test_weight_formula_accelerated_schedule(self) -> None:
    """s=2 should delay the transition.

    This verifies that larger acceleration values postpone the move toward
    information-based behavior instead of speeding it up.
    """
    selector = ProgressiveSelector(test_size=9, acceleration=2.0)
    assert selector._weight(5) == pytest.approx(0.25)  # noqa: SLF001

  def test_weight_formula_fast_schedule(self) -> None:
    """s=0.5 should accelerate the transition.

    This checks the opposite regime, where smaller acceleration values should
    pull the selector toward information earlier in the test.
    """
    selector = ProgressiveSelector(test_size=9, acceleration=0.5)
    assert selector._weight(5) == pytest.approx((4 / 8) ** 0.5)  # noqa: SLF001

  def test_first_item_is_independent_of_information(self) -> None:
    """At position 1, high-information items should not be selected more often.

    This catches regressions where the random component stops dominating the
    first step and high-information items start being overused immediately.
    """
    item_bank = ItemBank(
      np.array(
        [
          [5.0, 0.0, 0.0, 1.0],
          [0.3, -1.0, 0.0, 1.0],
          [0.3, 0.0, 0.0, 1.0],
          [0.3, 1.0, 0.0, 1.0],
          [0.3, 2.0, 0.0, 1.0],
        ],
        dtype=float,
      )
    )
    selector = ProgressiveSelector(test_size=10, acceleration=1.0)
    rng = np.random.default_rng(99)

    selections = [selector.select(item_bank, [], 0.0, rng=rng) for _ in range(200)]
    rate = selections.count(0) / len(selections)

    assert 0.10 < rate < 0.40

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

  def test_first_position_is_uniform_regardless_of_sharpness(self) -> None:
    """Test that the first position is uniform for any sharpness.

    Because the exponent is zero at the first step, the sampling distribution
    must ignore item information even when sharpness is large.
    """
    item_bank = _progressive_item_bank()

    for sharpness in (0.0, 12.0):
      selector = ProportionalSelector(test_size=3, sharpness=sharpness)
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

  def test_late_position_becomes_near_deterministic_with_high_sharpness(self) -> None:
    """Test that late positions collapse toward the highest-information item.

    This validates the upper end of the schedule, where proportional sampling
    should behave almost deterministically on the best item.
    """
    item_bank = _progressive_item_bank()
    selector = ProportionalSelector(test_size=3, acceleration=1.0, sharpness=12.0)
    rng = _DeterministicRNG([0.2, 0.1, 0.3, 0.4])

    selected = selector.select(
      item_bank=item_bank,
      administered_items=[0, 1],
      est_theta=0.0,
      rng=rng,
    )

    assert selected == 3
    _, probs = rng.choice_calls[0]
    assert probs is not None
    assert probs[1] > 0.999999
    assert probs[0] < 1e-6


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
  "KLSelector": KLSelector,
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
  "MEISelector": MEISelector,
  "MLWISelector": MLWISelector,
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
    response_vector=[True, False],
  )

  assert type(item_id) is int


def test_every_selector_accepts_the_engine_selection_inputs() -> None:
  """CatEngine always passes response_vector, so every concrete selector must accept it."""
  selectors = [
    cls
    for cls in vars(importlib.import_module("catsim.selection")).values()
    if isinstance(cls, type) and issubclass(cls, BaseSelector) and not inspect.isabstract(cls)
  ]

  missing = [cls.__name__ for cls in selectors if "response_vector" not in inspect.signature(cls.select).parameters]

  assert len(selectors) > 10
  assert missing == []
