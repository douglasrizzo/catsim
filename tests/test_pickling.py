"""Tests that every built-in strategy, an engine built from them, and a run's result survive pickling (ADR-0015)."""

import copy
import importlib
import inspect
import pickle

import numpy as np
import pytest
from irtlib import ItemBank
from irtlib.quadrature import QuadratureGrid, normal_log_prior, uniform_log_prior

from catsim import CatEngine, StepContext, simulate
from catsim.estimation import (
  BaseEstimator,
  EAPEstimator,
  MAPEstimator,
  MLEstimator,
  WarmLikelihoodEstimator,
)
from catsim.exposure import BaseExposureControl
from catsim.initialization import BaseInitializer, FixedPointInitializer, NormalInitializer, UniformInitializer
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  BaseSelector,
  ClusterSelector,
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
from catsim.stopping import (
  BaseStopper,
  ConfidenceIntervalStopper,
  MaxLengthStopper,
  StandardErrorStopper,
  StopRule,
)
from tests.contexts import make_context

STRATEGY_PACKAGES = [
  "catsim.selection",
  "catsim.estimation",
  "catsim.initialization",
  "catsim.stopping",
  "catsim.exposure",
]
STRATEGY_BASES = (BaseSelector, BaseEstimator, BaseInitializer, BaseStopper, BaseExposureControl)
N_ITEMS = 30
CLUSTERS = [item % 4 for item in range(N_ITEMS)]

# Each built-in strategy with its defaults and, where it has settings, with other values for them.
STRATEGIES = [
  pytest.param(MaxInfoSelector(), id="MaxInfoSelector"),
  pytest.param(MaxInfoSelector(r_max=0.3), id="MaxInfoSelector-capped"),
  pytest.param(UrrySelector(), id="UrrySelector"),
  pytest.param(KLSelector(), id="KLSelector"),
  pytest.param(KLSelector(window=1.5, r_max=0.3), id="KLSelector-settings"),
  pytest.param(IntervalInfoSelector(), id="IntervalInfoSelector"),
  pytest.param(IntervalInfoSelector(interval=1.0, confidence=0.9), id="IntervalInfoSelector-settings"),
  pytest.param(RandomSelector(), id="RandomSelector"),
  pytest.param(RandomesqueSelector(bin_size=4), id="RandomesqueSelector"),
  pytest.param(The54321Selector(), id="The54321Selector"),
  pytest.param(The54321Selector(initial_bin_size=3), id="The54321Selector-settings"),
  pytest.param(LinearSelector([4, 2, 9, 0, 17, 5, 11, 3]), id="LinearSelector"),
  pytest.param(ProgressiveSelector(test_size=10), id="ProgressiveSelector"),
  pytest.param(ProgressiveSelector(test_size=10, acceleration=2.0, r_max=0.3), id="ProgressiveSelector-settings"),
  pytest.param(ProportionalSelector(test_size=10), id="ProportionalSelector"),
  pytest.param(ProportionalSelector(test_size=10, acceleration=2.0, r_max=0.3), id="ProportionalSelector-settings"),
  pytest.param(ClusterSelector(CLUSTERS), id="ClusterSelector"),
  pytest.param(
    ClusterSelector(CLUSTERS, method="weighted_info", r_max=0.3, r_control="aggressive"), id="ClusterSelector-settings"
  ),
  pytest.param(AStratSelector(test_size=10), id="AStratSelector"),
  pytest.param(AStratSelector(test_size=10, n_strata=2), id="AStratSelector-settings"),
  pytest.param(AStratBBlockSelector(test_size=10, n_strata=2), id="AStratBBlockSelector"),
  pytest.param(MaxInfoStratSelector(test_size=10, n_strata=2), id="MaxInfoStratSelector"),
  pytest.param(MaxInfoBBlockSelector(test_size=10, n_strata=2), id="MaxInfoBBlockSelector"),
  pytest.param(MEISelector(), id="MEISelector"),
  pytest.param(MEISelector(estimator=EAPEstimator(), r_max=0.3), id="MEISelector-eap"),
  pytest.param(MLWISelector(), id="MLWISelector"),
  pytest.param(MLWISelector(n_nodes=21, bounds=(-4.0, 4.0), r_max=0.3), id="MLWISelector-settings"),
  pytest.param(MLEstimator(), id="MLEstimator"),
  pytest.param(MLEstimator(tol=1e-3, bounds=(-4.0, 4.0)), id="MLEstimator-settings"),
  pytest.param(EAPEstimator(), id="EAPEstimator"),
  pytest.param(
    EAPEstimator(grid=QuadratureGrid.uniform(31, -4.0, 4.0), log_prior=normal_log_prior(0.5, 2.0)),
    id="EAPEstimator-normal",
  ),
  pytest.param(EAPEstimator(log_prior=uniform_log_prior(-3.0, 3.0)), id="EAPEstimator-uniform"),
  pytest.param(MAPEstimator(), id="MAPEstimator"),
  pytest.param(MAPEstimator(log_prior=normal_log_prior(0.5, 2.0)), id="MAPEstimator-normal"),
  pytest.param(
    MAPEstimator(log_prior=uniform_log_prior(-3.0, 3.0), tol=1e-3, bounds=(-4.0, 4.0)), id="MAPEstimator-uniform"
  ),
  pytest.param(WarmLikelihoodEstimator(), id="WarmLikelihoodEstimator"),
  pytest.param(WarmLikelihoodEstimator(tol=1e-3, bounds=(-4.0, 4.0)), id="WarmLikelihoodEstimator-settings"),
  pytest.param(FixedPointInitializer(0.5), id="FixedPointInitializer"),
  pytest.param(NormalInitializer(), id="NormalInitializer"),
  pytest.param(NormalInitializer(mean=1.0, sd=0.5), id="NormalInitializer-settings"),
  pytest.param(UniformInitializer(), id="UniformInitializer"),
  pytest.param(UniformInitializer(low=-2.0, high=2.0), id="UniformInitializer-settings"),
  pytest.param(MaxLengthStopper(5), id="MaxLengthStopper"),
  pytest.param(StandardErrorStopper(0.5), id="StandardErrorStopper"),
  pytest.param(ConfidenceIntervalStopper([0.0]), id="ConfidenceIntervalStopper"),
  pytest.param(ConfidenceIntervalStopper([-1.0, 1.0], confidence=0.9), id="ConfidenceIntervalStopper-settings"),
  pytest.param(StopRule(MaxLengthStopper(5), StandardErrorStopper(0.5), min_items=2), id="StopRule"),
]


def _round_trip[T](obj: T) -> T:
  """Return a copy of ``obj`` made by pickling it, as a worker process receives it."""
  return pickle.loads(pickle.dumps(obj))


def _contexts(item_bank: ItemBank) -> list[StepContext]:
  """Return three sessions, empty, short and longer, with exposure rates and an exclusion, each with a new generator."""
  exposure_rates = np.linspace(0.0, 0.6, N_ITEMS)
  return [
    make_context(item_bank, rng=np.random.default_rng(1)),
    make_context(
      item_bank,
      [0, 5, 9],
      [True, False, True],
      theta=0.4,
      se=0.8,
      rng=np.random.default_rng(2),
      exposure_rates=exposure_rates,
      excluded_items=[2],
    ),
    make_context(
      item_bank,
      [12, 4, 20, 7, 25, 1, 16],
      [False, True, False, False, True, False, True],
      theta=-0.7,
      se=0.35,
      rng=np.random.default_rng(3),
      exposure_rates=exposure_rates,
    ),
  ]


def _outputs(strategy: object, item_bank: ItemBank) -> list[object]:
  """Return what ``strategy`` gives on the fixed contexts, or for initializers on fixed seeds."""
  match strategy:
    case BaseSelector():
      return [strategy.select(context) for context in _contexts(item_bank)]
    case BaseEstimator():
      return [strategy.estimate(context) for context in _contexts(item_bank)]
    case BaseStopper():
      return [strategy.stop(context) for context in _contexts(item_bank)]
    case BaseInitializer():
      return [strategy.initialize(item_bank=item_bank, rng=np.random.default_rng(seed)) for seed in range(3)]
    case _:
      msg = f"no outputs defined for {type(strategy).__name__}"
      raise TypeError(msg)


def _built_in_strategies() -> set[type]:
  """Return every concrete strategy class that a strategy package exports."""
  packages = [importlib.import_module(name) for name in STRATEGY_PACKAGES]
  return {
    obj
    for package in packages
    for name in package.__all__
    if inspect.isclass(obj := getattr(package, name))
    and issubclass(obj, STRATEGY_BASES)
    and not inspect.isabstract(obj)
  }


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_strategy_copy_behaves_like_the_original(strategy: object) -> None:
  """A strategy that fails to pickle, or whose copy loses part of its configuration, fails here.

  EAP's and MAP's priors, and MLWI's flat one, were closures, which pickle cannot send: an engine using them could not
  reach a worker process. The strategy runs before it is pickled, so whatever it caches from the bank goes along too.
  """
  item_bank = ItemBank.generate(N_ITEMS, rng=11)
  expected = _outputs(strategy, item_bank)

  copy = _round_trip(strategy)

  assert (repr(copy), _outputs(copy, item_bank)) == (repr(strategy), expected)


def test_every_built_in_strategy_has_a_pickle_case() -> None:
  """A strategy class exported without a case in STRATEGIES could stop pickling without any test noticing."""
  covered = {type(param.values[0]) for param in STRATEGIES}

  assert covered == _built_in_strategies()


def _run(engine: CatEngine) -> tuple[list[tuple[int, ...]], list[list[float]], list[float]]:
  """Return the items, the estimates and the final standard errors of a seeded run of three examinees."""
  result = simulate(engine, true_thetas=[-1.0, 0.0, 1.5], rng=5)
  return (
    [session.administered_items for session in result.sessions],
    [list(session.theta_history) for session in result.sessions],
    result.examinees()["se"].tolist(),
  )


def test_engine_copy_runs_like_the_original() -> None:
  """An engine with EAP's default prior failed ``pickle.dumps``, so it could not be sent to a worker process.

  MEI's look-ahead estimator and the engine's own estimator are both EAP with built-in priors, and the stopper a rule
  of two stoppers; the copied engine must give the same seeded run.
  """
  engine = CatEngine(
    ItemBank.generate(N_ITEMS, rng=11),
    NormalInitializer(),
    MEISelector(estimator=EAPEstimator()),
    EAPEstimator(log_prior=normal_log_prior(0.5, 2.0)),
    StopRule(MaxLengthStopper(6), StandardErrorStopper(0.4)),
  )

  assert _run(_round_trip(engine)) == _run(engine)


@pytest.mark.parametrize("duplicate", [_round_trip, copy.deepcopy], ids=["pickle", "deepcopy"])
def test_a_copied_result_is_as_immutable_as_the_original(duplicate) -> None:
  """NumPy pickles an array without its read-only flag, so a restored result's arrays became writeable.

  A pickled or deep-copied result must hold the same sessions, abilities and counts, still read-only. Fails if the
  copy restores the fields as they are, without making its arrays read-only, which lets its exposure counts be
  changed so that they disagree with its sessions.
  """
  result = simulate(
    CatEngine(
      ItemBank.generate(N_ITEMS, rng=3),
      FixedPointInitializer(0.0),
      MaxInfoSelector(),
      EAPEstimator(),
      MaxLengthStopper(4),
    ),
    true_thetas=[-1.0, 0.5],
    rng=2,
  )

  copied = duplicate(result)

  assert copied.sessions == result.sessions
  assert copied.true_thetas is not None
  assert (copied.true_thetas.tolist(), copied.exposure_counts.tolist(), copied.proposal_counts.tolist()) == (
    [-1.0, 0.5],
    result.exposure_counts.tolist(),
    result.proposal_counts.tolist(),
  )
  assert (copied.duration, copied.parts) == (result.duration, 1)
  arrays = (copied.true_thetas, copied.exposure_counts, copied.proposal_counts)
  assert not any(array.flags.writeable for array in arrays)
