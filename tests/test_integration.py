"""Integration tests for catsim simulation pipeline.

These tests run full CAT simulations with various configurations to verify
that all components work correctly together.
"""

import random

import numpy as np
import pytest
from sklearn.cluster import KMeans

from catsim import cat, irt
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import BaseInitializer, InitializationDistribution, RandomInitializer
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  BaseSelector,
  ClusterSelector,
  LinearSelector,
  MaxInfoBBlockSelector,
  MaxInfoSelector,
  MaxInfoStratSelector,
  RandomesqueSelector,
  RandomSelector,
  The54321Selector,
  UrrySelector,
)
from catsim.simulation import Simulator
from catsim.stopping import (
  BaseStopper,
  ConfidenceIntervalStopper,
  MinErrorStopper,
)


def one_simulation(
  items: ItemBank,
  examinees: int,
  initializer: BaseInitializer,
  selector: BaseSelector,
  estimator: NumericalSearchEstimator,
  stopper: BaseStopper,
) -> Simulator:
  """Test a single simulation.

  Returns
  -------
  Simulator
      The simulator instance after running the simulation.
  """
  simulator = Simulator(items, examinees, initializer, selector, estimator, stopper)
  simulator.simulate(verbose=True)

  # Verify simulation ran successfully
  assert simulator.latest_estimations is not None, "Simulation did not produce estimations"
  assert len(simulator.latest_estimations) == examinees, "Incorrect number of examinees"
  assert simulator.administered_items is not None, "No items were administered"

  return simulator


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parallel
@pytest.mark.parametrize("examinees", [100])
@pytest.mark.parametrize("bank_size", [500])
@pytest.mark.parametrize("initializer", [RandomInitializer(InitializationDistribution.UNIFORM, (-5, 5))])
@pytest.mark.parametrize("estimator", [NumericalSearchEstimator()])
@pytest.mark.parametrize(
  "stopper",
  [
    MinErrorStopper(0.4, max_items=30),
    MinErrorStopper(0.4, min_items=10, max_items=30),
  ],
)
def test_cism(
  examinees: int,
  bank_size: int,
  initializer: BaseInitializer,
  estimator: NumericalSearchEstimator,
  stopper: BaseStopper,
) -> None:
  """Test the cluster-based item selection method."""
  item_bank = ItemBank.generate_item_bank(bank_size)
  clusters = list(KMeans(n_clusters=8, n_init="auto").fit_predict(item_bank.items))
  ClusterSelector.weighted_cluster_infos(0, item_bank, clusters)
  ClusterSelector.avg_cluster_params(item_bank, clusters)
  selector = ClusterSelector(clusters=clusters, r_max=0.2)
  one_simulation(item_bank, examinees, initializer, selector, estimator, stopper)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parallel
@pytest.mark.parametrize("examinees", [100])
@pytest.mark.parametrize("bank_size", [500])
@pytest.mark.parametrize(
  "estimator", [NumericalSearchEstimator(method=m) for m in sorted(NumericalSearchEstimator.available_methods())]
)
def test_estimators(
  examinees: int,
  bank_size: int,
  estimator: NumericalSearchEstimator,
) -> None:
  """Test all NumericalSearchEstimator methods with simple selector configurations."""
  rng = np.random.default_rng(1337)
  item_bank = ItemBank.generate_item_bank(bank_size, itemtype=irt.NumParams.PL4)
  initializer = RandomInitializer(InitializationDistribution.UNIFORM, (-5, 5))
  stopper = MinErrorStopper(0.4, max_items=30)
  test_size = 30

  # Test with a finite selector (LinearSelector)
  finite_selector = LinearSelector(list(rng.choice(bank_size, size=test_size, replace=False)))
  one_simulation(item_bank, examinees, initializer, finite_selector, estimator, stopper)

  # Test with an infinite selector (RandomSelector)
  infinite_selector = RandomSelector()
  one_simulation(item_bank, examinees, initializer, infinite_selector, estimator, stopper)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parallel
@pytest.mark.parametrize("examinees", [100])
@pytest.mark.parametrize("test_size", [30])
@pytest.mark.parametrize("bank_size", [500])
def test_finite_selectors(
  examinees: int,
  test_size: int,
  bank_size: int,
) -> None:
  """Test all finite selectors with brent and bounded estimators."""
  rng = np.random.default_rng(1337)
  item_bank = ItemBank.generate_item_bank(bank_size, itemtype=irt.NumParams.PL4)
  initializer = RandomInitializer(InitializationDistribution.UNIFORM, (-5, 5))
  stopper = MinErrorStopper(0.4, max_items=test_size)

  finite_selectors = [
    LinearSelector(list(rng.choice(bank_size, size=test_size, replace=False))),
    AStratSelector(test_size),
    AStratBBlockSelector(test_size),
    MaxInfoStratSelector(test_size),
    MaxInfoBBlockSelector(test_size),
    The54321Selector(test_size),
    RandomesqueSelector(test_size // 6),
  ]

  estimators = [
    NumericalSearchEstimator(method="brent"),
    NumericalSearchEstimator(method="bounded"),
  ]

  for selector in finite_selectors:
    for estimator in estimators:
      rng = np.random.default_rng(1337)
      responses = cat.random_response_vector(random.randint(1, test_size - 1))
      administered_items = list(rng.choice(bank_size, len(responses), replace=False))
      est_theta = initializer.initialize(rng=rng)
      selector.select(item_bank=item_bank, administered_items=administered_items, est_theta=est_theta, rng=rng)
      estimator.estimate(
        item_bank=item_bank,
        administered_items=administered_items,
        response_vector=responses,
        est_theta=est_theta,
      )
      stopper.stop(
        _item_bank=item_bank, administered_items=item_bank.get_items(administered_items), theta=est_theta, rng=rng
      )

      one_simulation(item_bank, examinees, initializer, selector, estimator, stopper)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parallel
@pytest.mark.parametrize("examinees", [100])
@pytest.mark.parametrize("bank_size", [5000])
@pytest.mark.parametrize(
  "selector",
  [
    MaxInfoSelector(),
    RandomSelector(),
    UrrySelector(),
  ],
)
def test_infinite_selectors(
  examinees: int,
  bank_size: int,
  selector: BaseSelector,
) -> None:
  """Test all infinite selectors with brent and bounded estimators."""
  rng = np.random.default_rng(1337)
  item_bank = ItemBank.generate_item_bank(bank_size, itemtype=irt.NumParams.PL4)
  initializer = RandomInitializer(InitializationDistribution.UNIFORM, (-5, 5))
  stopper = MinErrorStopper(0.4, max_items=30)

  estimators = [
    NumericalSearchEstimator(method="brent"),
    NumericalSearchEstimator(method="bounded"),
  ]

  for estimator in estimators:
    responses = cat.random_response_vector(random.randint(1, 30))
    administered_items = list(rng.choice(bank_size, len(responses), replace=False))
    est_theta = initializer.initialize(rng=rng)
    selector.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      rng=rng,
    )
    estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=responses,
      est_theta=est_theta,
    )
    stopper.stop(
      _item_bank=item_bank,
      administered_items=item_bank.get_items(administered_items),
      theta=est_theta,
    )
    one_simulation(item_bank, examinees, initializer, selector, estimator, stopper)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parallel
@pytest.mark.parametrize("examinees", [100])
@pytest.mark.parametrize("bank_size", [5000])
@pytest.mark.parametrize(
  "stopper",
  [
    MinErrorStopper(0.4, max_items=30),
    MinErrorStopper(0.4, min_items=10, max_items=30),
    ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.80, max_items=50),
    ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.80, min_items=10, max_items=50),
  ],
)
def test_stoppers(
  examinees: int,
  bank_size: int,
  stopper: BaseStopper,
) -> None:
  """Test all stopper configurations with brent and bounded estimators."""
  rng = np.random.default_rng(1337)
  item_bank = ItemBank.generate_item_bank(bank_size, itemtype=irt.NumParams.PL4)
  initializer = RandomInitializer(InitializationDistribution.UNIFORM, (-5, 5))
  selector = MaxInfoSelector()

  # Extract max items from stopper
  max_administered_items = stopper.max_items if stopper.max_items is not None else bank_size

  estimators = [
    NumericalSearchEstimator(method="brent"),
    NumericalSearchEstimator(method="bounded"),
  ]

  for estimator in estimators:
    responses = cat.random_response_vector(random.randint(1, max_administered_items))
    administered_items = list(rng.choice(bank_size, len(responses), replace=False))
    est_theta = initializer.initialize(rng=rng)
    selector.select(
      item_bank=item_bank,
      administered_items=administered_items,
      est_theta=est_theta,
      rng=rng,
    )
    estimator.estimate(
      item_bank=item_bank,
      administered_items=administered_items,
      response_vector=responses,
      est_theta=est_theta,
    )
    stopper.stop(
      _item_bank=item_bank,
      administered_items=item_bank.get_items(administered_items),
      theta=est_theta,
    )
    one_simulation(item_bank, examinees, initializer, selector, estimator, stopper)
