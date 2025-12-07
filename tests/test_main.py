import random

import numpy as np
import pytest
from matplotlib.pyplot import close
from sklearn.cluster import KMeans

from catsim import cat, irt, plot
from catsim.estimation import NumericalSearchEstimator
from catsim.initialization import FixedPointInitializer, InitializationDistribution, RandomInitializer
from catsim.item_bank import ItemBank
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
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
from catsim.simulation import Estimator, Initializer, Selector, Simulator, Stopper
from catsim.stopping import (
  CombinationStopper,
  CombinationStrategy,
  ConfidenceIntervalStopper,
  ItemBankLengthStopper,
  MaxItemStopper,
  MinErrorStopper,
)


def one_simulation(
  items: ItemBank,
  examinees: int,
  initializer: Initializer,
  selector: Selector,
  estimator: Estimator,
  stopper: Stopper,
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
    CombinationStopper([MaxItemStopper(30), ItemBankLengthStopper()], strategy=CombinationStrategy.OR),
    CombinationStopper([MinErrorStopper(0.4), ItemBankLengthStopper()], strategy=CombinationStrategy.OR),
    CombinationStopper(
      [MaxItemStopper(30), MinErrorStopper(0.4), ItemBankLengthStopper()], strategy=CombinationStrategy.OR
    ),
  ],
)
def test_cism(
  examinees: int,
  bank_size: int,
  initializer: Initializer,
  estimator: Estimator,
  stopper: Stopper,
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
@pytest.mark.parametrize("test_size", [30])
@pytest.mark.parametrize("bank_size", [500])
@pytest.mark.parametrize("logistic_model", [irt.NumParams.PL4])
@pytest.mark.parametrize(
  "initializer",
  [
    RandomInitializer(InitializationDistribution.UNIFORM, (-5, 5)),
    FixedPointInitializer(0),
  ],
)
@pytest.mark.parametrize(
  "estimator", [NumericalSearchEstimator(method=m) for m in NumericalSearchEstimator.available_methods()]
)
def test_finite_selectors(
  examinees: int,
  test_size: int,
  bank_size: int,
  logistic_model: irt.NumParams,
  initializer: Initializer,
  estimator: Estimator,
) -> None:
  """Test finite selectors."""
  rng = np.random.default_rng()
  finite_selectors = [
    LinearSelector(list(rng.choice(bank_size, size=test_size, replace=False))),
    AStratSelector(test_size),
    AStratBBlockSelector(test_size),
    MaxInfoStratSelector(test_size),
    MaxInfoBBlockSelector(test_size),
    The54321Selector(test_size),
    RandomesqueSelector(test_size // 6),
  ]
  stopper = CombinationStopper([MaxItemStopper(test_size), ItemBankLengthStopper()], strategy=CombinationStrategy.OR)

  for selector in finite_selectors:
    rng = np.random.default_rng(1337)
    item_bank = ItemBank.generate_item_bank(bank_size, itemtype=logistic_model)
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
    stopper.stop(administered_items=item_bank.get_items(administered_items), theta=est_theta, rng=rng)

    one_simulation(item_bank, examinees, initializer, selector, estimator, stopper)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parallel
@pytest.mark.parametrize("examinees", [100])
@pytest.mark.parametrize("bank_size", [5000])
@pytest.mark.parametrize("logistic_model", [irt.NumParams.PL4])
@pytest.mark.parametrize(
  "initializer",
  [
    RandomInitializer(InitializationDistribution.UNIFORM, (-5, 5)),
    FixedPointInitializer(0),
  ],
)
@pytest.mark.parametrize(
  "selector",
  [
    MaxInfoSelector(),
    RandomSelector(),
    UrrySelector(),
  ],
)
@pytest.mark.parametrize(
  "estimator", [NumericalSearchEstimator(method=m) for m in NumericalSearchEstimator.available_methods()]
)
@pytest.mark.parametrize(
  "stopper",
  [
    CombinationStopper([MaxItemStopper(30), ItemBankLengthStopper()], strategy=CombinationStrategy.OR),
    CombinationStopper([MinErrorStopper(0.4), ItemBankLengthStopper()], strategy=CombinationStrategy.OR),
    CombinationStopper(
      [MaxItemStopper(30), MinErrorStopper(0.4), ItemBankLengthStopper()], strategy=CombinationStrategy.OR
    ),
    CombinationStopper(
      [ConfidenceIntervalStopper([-2.0, 0.0, 2.0], confidence=0.80), ItemBankLengthStopper()],
      strategy=CombinationStrategy.OR,
    ),
  ],
)
def test_infinite_selectors(
  examinees: int,
  bank_size: int,
  logistic_model: irt.NumParams,
  initializer: Initializer,
  selector: Selector,
  estimator: Estimator,
  stopper: Stopper,
) -> None:
  """Test infinite selectors."""
  rng = np.random.default_rng()
  item_bank = ItemBank.generate_item_bank(bank_size, itemtype=logistic_model)

  # Extract max items from stopper (check if it's a CombinationStopper with MaxItemStopper inside)
  max_administered_items = bank_size
  if isinstance(stopper, CombinationStopper):
    for sub_stopper in stopper.stoppers:
      if isinstance(sub_stopper, MaxItemStopper):
        max_administered_items = min(max_administered_items, sub_stopper.max_itens)
  elif isinstance(stopper, MaxItemStopper):
    max_administered_items = stopper.max_itens

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


@pytest.mark.parametrize(
  "logistic_model",
  [
    irt.NumParams.PL1,
    irt.NumParams.PL2,
    irt.NumParams.PL3,
    irt.NumParams.PL4,
  ],
)
@pytest.mark.parametrize(
  "correlation",
  [0, 0.3],
)
def test_item_bank_generation(logistic_model: irt.NumParams, correlation: float) -> None:
  """Test item bank generation function."""
  item_bank = ItemBank.generate_item_bank(5, logistic_model, corr=correlation)
  irt.validate_item_bank(item_bank.items, raise_err=True)

  # Verify basic properties of generated items
  assert len(item_bank) == 5, "Incorrect number of items generated"
  assert item_bank.items.shape[1] >= 4, "Items should have at least 4 parameters"


@pytest.mark.slow
def test_plots() -> None:
  """Test plot functionalities."""
  initializer = RandomInitializer()
  selector = MaxInfoSelector()
  estimator = NumericalSearchEstimator()
  stopper = CombinationStopper([MaxItemStopper(20), ItemBankLengthStopper()], strategy=CombinationStrategy.OR)
  s = Simulator(ItemBank.generate_item_bank(100), 10)
  s.simulate(initializer, selector, estimator, stopper, verbose=True)

  # Verify simulation produced results before plotting
  assert s.items is not None, "Simulation did not produce items"
  assert len(s.items) > 0, "No items in simulation"

  for item in s.items[0:10]:
    plot.item_curve(item[0], item[1], item[2], item[3], title="Test plot", ptype=plot.PlotType.ICC, max_info=False)
    plot.item_curve(item[0], item[1], item[2], item[3], title="Test plot", ptype=plot.PlotType.IIC, max_info=True)
    plot.item_curve(item[0], item[1], item[2], item[3], title="Test plot", ptype=plot.PlotType.BOTH, max_info=True)
    close("all")

  plot.gen3d_dataset_scatter(s.item_bank)
  plot.test_progress(
    title="Test progress",
    simulator=s,
    index=0,
    info=True,
    see=True,
    reliability=True,
  )
  plot.item_exposure(simulator=s)
  plot.item_exposure(simulator=s, par="a")
  plot.item_exposure(simulator=s, par="b")
  plot.item_exposure(simulator=s, par="c")
  plot.item_exposure(simulator=s, par="d")
  plot.item_exposure(simulator=s, hist=True)

  # close all plots after testing
  close("all")
