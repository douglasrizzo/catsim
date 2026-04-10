"""Tests for catsim.selection module."""

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
  LinearSelector,
  MaxInfoBBlockSelector,
  MaxInfoSelector,
  MaxInfoStratSelector,
  MEISelector,
  MLWISelector,
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
