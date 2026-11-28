"""Every computation the baseline records, as named cases grouped in families.

A case is a function of no arguments that computes one output from fixed inputs: fixed item banks,
abilities, response patterns, step contexts and seeds, all defined in this module. ``FAMILIES`` maps
each family to its cases, and each family is recorded in its own data file. Case names are written
out here, never derived from catsim's reprs, so a changed repr leaves the data as it is.

The families:

- ``estimation``: every estimator, with default and non-default settings, on fixed response patterns
  of every length up to the bank's, constant patterns included.
- ``selection``: every selector's choices in fixed contexts, with a seeded generator, with and
  without exposure rates and an ``r_max`` cap, and with excluded items; and the selection helpers.
  Each case records the selector's order of preference: its choice, then its choice with that item
  excluded, and so on, up to ``RANKING_LENGTH`` choices or until it has no item left.
- ``stopping``: every stopper and ``StopRule`` combination on a grid of fixed contexts.
- ``initialization``: every initializer, drawing from seeded generators.
- ``metrics``: every function of ``catsim.metrics`` on fixed arrays.
- ``whole_tests``: whole tests answered from fixed response rows, by configurations that draw no
  random number, so the random model of the run cannot change them; two of them cap exposure, which
  makes each session depend on the ones before it.

:func:`uncovered` lists any public strategy class, function or result member that no case
exercises, and the generator refuses to record while it lists one.

The estimators' evaluation counts are not recorded: they count calls that SciPy's searches,
root finders and quadrature make, which change with SciPy's version (QUADPACK's count did between
SciPy 1.16 and 1.18) while the estimates do not.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy
from irtlib import ItemBank
from irtlib.quadrature import (
  QuadratureGrid,
  normal_log_prior,
  uniform_log_prior,
)

from catsim import NoItemsAvailableError, SimulationResult
from catsim.estimation import (
  BaseEstimator,
  EAPEstimator,
  MAPEstimator,
  NumericalSearchEstimator,
  WarmLikelihoodEstimator,
)
from catsim.exposure import BaseExposureControl
from catsim.initialization import BaseInitializer, FixedPointInitializer, NormalInitializer, UniformInitializer
from catsim.selection import (
  AStratBBlockSelector,
  AStratSelector,
  BaseSelector,
  CapFallback,
  ClusterMethod,
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
  StratifiedSelector,
  The54321Selector,
  UrrySelector,
  below_cap,
  best_pool,
  pick_best,
  remaining,
)
from catsim.stopping import (
  BaseStopper,
  ConfidenceIntervalStopper,
  MaxLengthStopper,
  StandardErrorStopper,
  StopRule,
)
from tests.reference import adapters

if TYPE_CHECKING:
  from irtlib.typing import FloatArray, IntArray

Case = Callable[[], object]

# The fixed inputs.

# Twelve items with irregular parameters: no two difficulties are equal or sit symmetrically around
# an ability the cases use, so no selector's score ties on them.
_A = (1.21, 0.83, 1.67, 0.95, 1.38, 2.04, 0.71, 1.12, 1.53, 0.89, 1.86, 1.29)
_B = (-2.31, -1.47, -0.88, -0.34, 0.12, 0.57, 1.03, 1.71, 2.26, -0.61, 0.79, -1.12)
_C = (0.21, 0.05, 0.17, 0.0, 0.12, 0.26, 0.08, 0.19, 0.14, 0.23, 0.03, 0.1)
_D = (0.97, 1.0, 0.93, 0.99, 0.95, 1.0, 0.98, 0.91, 0.96, 1.0, 0.94, 0.99)

# The bank inputs exercise the item bank's reading of one to four columns and of a single item.
BANK_INPUTS: dict[str, FloatArray] = {
  "1pl": numpy.column_stack([_B]),
  "2pl": numpy.column_stack([_A, _B]),
  "3pl": numpy.column_stack([_A, _B, _C]),
  "4pl": numpy.column_stack([_A, _B, _C, _D]),
  "single_item": numpy.array([1.4, -0.3, 0.15, 0.97]),
}
BANKS = {name: ItemBank(items) for name, items in BANK_INPUTS.items()}
MODEL_BANKS = ("1pl", "2pl", "3pl", "4pl")

# A 1PL bank of repeated and symmetric difficulties, on which selectors break exact ties (ADR-0017).
TIE_BANK = ItemBank(numpy.array([[-1.0], [-0.5], [-0.5], [0.0], [0.0], [0.0], [0.5], [0.5], [1.0], [1.5]]))


def _whole_test_bank(n_items: int) -> ItemBank:
  """Return a 4PL bank whose parameters are fractional parts of multiples of irrational steps.

  Only multiplications, additions and remainders build it, which IEEE 754 rounds the same way on
  every platform, so the bank is identical everywhere and has no two equal parameters.
  """
  rows = []
  for j in range(n_items):
    u1, u2 = (j * 0.6180339887498949) % 1.0, (j * 0.4142135623730951 + 0.25) % 1.0
    u3, u4 = (j * 0.7320508075688772 + 0.5) % 1.0, (j * 0.2360679774997898 + 0.1) % 1.0
    rows.append([0.6 + 1.6 * u1, -3.0 + 6.0 * u2, 0.25 * u3, 0.9 + 0.1 * u4])
  return ItemBank(numpy.array(rows))


WHOLE_TEST_BANK = _whole_test_bank(30)

# The order the estimation cases administer the twelve items in, and the responses of their mixed pattern.
ITEM_ORDER = (5, 0, 9, 3, 11, 7, 1, 10, 2, 8, 4, 6)
MIXED = (True, False, False, True, True, False, True, False, True, True, False, True)
PATTERNS: dict[str, tuple[bool, ...]] = {"correct": (True,) * 12, "incorrect": (False,) * 12, "mixed": MIXED}

RANKING_LENGTH = 6
SELECTION_SEED = 20261125
WHOLE_TEST_SEED = 1

# The estimation family.

# A grid with uneven weights, so the weights take part in the posterior rather than cancelling.
_UNEVEN_GRID = QuadratureGrid(
  nodes=numpy.array([-4.0, -3.0, -2.2, -1.5, -0.9, -0.4, 0.0, 0.4, 0.9, 1.5, 2.2, 3.0, 4.0]),
  weights=numpy.array([0.2, 0.5, 0.8, 1.1, 1.3, 1.45, 1.5, 1.45, 1.3, 1.1, 0.8, 0.5, 0.2]),
)

ESTIMATORS: dict[str, Callable[[], BaseEstimator]] = {
  "NumericalSearchEstimator()": NumericalSearchEstimator,
  "NumericalSearchEstimator(bounds=(-3, 3))": lambda: NumericalSearchEstimator(bounds=(-3.0, 3.0)),
  "NumericalSearchEstimator(tol=1e-9)": lambda: NumericalSearchEstimator(tol=1e-9),
  "EAPEstimator()": EAPEstimator,
  "EAPEstimator(grid=uniform(33, -4, 4))": lambda: EAPEstimator(grid=QuadratureGrid.uniform(33, -4.0, 4.0)),
  "EAPEstimator(grid=uneven)": lambda: EAPEstimator(grid=_UNEVEN_GRID),
  "EAPEstimator(log_prior=normal(0.5, 1.5))": lambda: EAPEstimator(log_prior=normal_log_prior(0.5, 1.5)),
  "EAPEstimator(log_prior=uniform(-4, 4))": lambda: EAPEstimator(log_prior=uniform_log_prior(-4.0, 4.0)),
  "MAPEstimator()": MAPEstimator,
  "MAPEstimator(log_prior=uniform())": lambda: MAPEstimator(log_prior=uniform_log_prior()),
  "MAPEstimator(log_prior=uniform(-2, 2))": lambda: MAPEstimator(log_prior=uniform_log_prior(-2.0, 2.0)),
  "MAPEstimator(log_prior=normal(-0.5, 0.8))": lambda: MAPEstimator(log_prior=normal_log_prior(-0.5, 0.8)),
  "MAPEstimator(bounds=(-3, 3))": lambda: MAPEstimator(bounds=(-3.0, 3.0)),
  "WarmLikelihoodEstimator()": WarmLikelihoodEstimator,
  "WarmLikelihoodEstimator(bounds=(-2, 2))": lambda: WarmLikelihoodEstimator(bounds=(-2.0, 2.0)),
  "WarmLikelihoodEstimator(tol=1e-9)": lambda: WarmLikelihoodEstimator(tol=1e-9),
}


def _estimation_contexts() -> dict[str, dict[str, Any]]:
  """Return the sessions the estimators estimate for, by label: each length with each distinct pattern."""
  contexts: dict[str, dict[str, Any]] = {"L00": {"theta": 0.3}}
  for length in range(1, len(ITEM_ORDER) + 1):
    seen: set[tuple[bool, ...]] = set()
    for name, pattern in PATTERNS.items():
      if (responses := pattern[:length]) not in seen:
        seen.add(responses)
        contexts[f"L{length:02d}|{name}"] = {"administered_items": ITEM_ORDER[:length], "responses": responses}
  # Dodd's step starts from the previous estimate, which these move past the bank's extreme difficulties.
  for length, name, theta in ((3, "correct", 2.5), (3, "incorrect", -2.5), (5, "correct", 5.9), (5, "incorrect", -0.7)):
    contexts[f"L{length:02d}|{name}|theta={theta}"] = {
      "administered_items": ITEM_ORDER[:length],
      "responses": PATTERNS[name][:length],
      "theta": theta,
    }
  return contexts


ESTIMATION_CONTEXTS = _estimation_contexts()


def _estimates(make: Callable[[], BaseEstimator], bank: ItemBank) -> dict[str, list[object]]:
  """Return the estimate and standard error of one estimator on every estimation context."""
  estimator = make()
  outputs: dict[str, list[object]] = {}
  for label, settings in ESTIMATION_CONTEXTS.items():
    administered_items = settings.get("administered_items", ())
    se = 1.0 if administered_items else math.inf
    context = adapters.step_context(bank, rng=numpy.random.default_rng(0), se=se, **settings)
    estimate = estimator.estimate(context)
    outputs[label] = [estimate.theta, estimate.se]
  return outputs


def _estimation_cases() -> dict[str, Case]:
  return {
    f"{label}|{bank}": functools.partial(_estimates, make, BANKS[bank])
    for label, make in ESTIMATORS.items()
    for bank in MODEL_BANKS
  }


ESTIMATION_EXEMPT = {"BaseEstimator": "abstract"}

# The selection family.


def _clusters(bank: ItemBank) -> list[int]:
  """Return four cluster labels spread over the bank's items."""
  return [(index * 7) % 4 for index in range(bank.n_items)]


def _linear_items(bank: ItemBank) -> list[int]:
  """Return every third item of the bank, from the last down."""
  return list(range(bank.n_items - 1, -1, -3))


# Each selector is built for the bank it selects from, since cluster labels and linear lists depend on it.
SELECTORS: dict[str, Callable[[ItemBank], BaseSelector]] = {
  "MaxInfoSelector()": lambda _: MaxInfoSelector(),
  "MaxInfoSelector(r_max=0.3)": lambda _: MaxInfoSelector(r_max=0.3),
  "UrrySelector()": lambda _: UrrySelector(),
  "IntervalInfoSelector()": lambda _: IntervalInfoSelector(),
  "IntervalInfoSelector(interval=0.8)": lambda _: IntervalInfoSelector(interval=0.8),
  "IntervalInfoSelector(confidence=0.8)": lambda _: IntervalInfoSelector(confidence=0.8),
  "KLSelector()": lambda _: KLSelector(),
  "KLSelector(window=1.5)": lambda _: KLSelector(window=1.5),
  "KLSelector(r_max=0.3)": lambda _: KLSelector(r_max=0.3),
  "LinearSelector(every third, from the last)": lambda bank: LinearSelector(_linear_items(bank)),
  "MEISelector()": lambda _: MEISelector(),
  "MEISelector(estimator=EAPEstimator())": lambda _: MEISelector(estimator=EAPEstimator()),
  "MEISelector(r_max=0.3)": lambda _: MEISelector(r_max=0.3),
  "MLWISelector()": lambda _: MLWISelector(),
  "MLWISelector(n_nodes=21, bounds=(-4, 4))": lambda _: MLWISelector(n_nodes=21, bounds=(-4.0, 4.0)),
  "MLWISelector(r_max=0.3)": lambda _: MLWISelector(r_max=0.3),
  "ProgressiveSelector(test_size=8)": lambda _: ProgressiveSelector(test_size=8),
  "ProgressiveSelector(test_size=8, acceleration=1.5)": lambda _: ProgressiveSelector(test_size=8, acceleration=1.5),
  "ProgressiveSelector(test_size=8, r_max=0.3)": lambda _: ProgressiveSelector(test_size=8, r_max=0.3),
  "ProportionalSelector(test_size=8)": lambda _: ProportionalSelector(test_size=8),
  "ProportionalSelector(test_size=8, acceleration=-0.5)": lambda _: ProportionalSelector(
    test_size=8, acceleration=-0.5
  ),
  "ProportionalSelector(test_size=8, r_max=0.3)": lambda _: ProportionalSelector(test_size=8, r_max=0.3),
  "RandomSelector()": lambda _: RandomSelector(),
  "RandomesqueSelector(bin_size=4)": lambda _: RandomesqueSelector(bin_size=4),
  "The54321Selector()": lambda _: The54321Selector(),
  "The54321Selector(initial_bin_size=3)": lambda _: The54321Selector(initial_bin_size=3),
  "AStratSelector(test_size=8)": lambda _: AStratSelector(test_size=8),
  "AStratSelector(test_size=8, n_strata=4)": lambda _: AStratSelector(test_size=8, n_strata=4),
  "AStratBBlockSelector(test_size=8, n_strata=4)": lambda _: AStratBBlockSelector(test_size=8, n_strata=4),
  "MaxInfoStratSelector(test_size=8, n_strata=4)": lambda _: MaxInfoStratSelector(test_size=8, n_strata=4),
  "MaxInfoBBlockSelector(test_size=8, n_strata=4)": lambda _: MaxInfoBBlockSelector(test_size=8, n_strata=4),
  "ClusterSelector(item_info)": lambda bank: ClusterSelector(_clusters(bank)),
  "ClusterSelector(cluster_info)": lambda bank: ClusterSelector(_clusters(bank), method=ClusterMethod.CLUSTER_INFO),
  "ClusterSelector(weighted_info)": lambda bank: ClusterSelector(_clusters(bank), method=ClusterMethod.WEIGHTED_INFO),
  "ClusterSelector(r_max=0.3, passive)": lambda bank: ClusterSelector(_clusters(bank), r_max=0.3),
  "ClusterSelector(r_max=0.3, aggressive)": lambda bank: ClusterSelector(
    _clusters(bank), r_max=0.3, r_control=CapFallback.AGGRESSIVE
  ),
}
# The selectors whose settings above cap exposure, which the cases also give rates all at or above the cap.
CAPPED = {label for label in SELECTORS if "r_max" in label}

SELECTION_CONTEXTS: dict[str, dict[str, Any]] = {
  "start": {},
  "after_3": {"administered_items": (3, 8, 1), "responses": (True, False, True), "theta": 0.42, "se": 0.9},
  "after_6_excluded": {
    "administered_items": (5, 0, 9, 2, 11, 7),
    "responses": (True, True, False, True, False, False),
    "excluded_items": (4, 10),
    "theta": -1.1,
    "se": 0.45,
  },
  "high_after_1": {"administered_items": (2,), "responses": (True,), "theta": 2.7, "se": 1.3},
}
TIE_CONTEXTS: dict[str, dict[str, Any]] = {
  "start": {},
  "after_2": {"administered_items": (3, 8), "responses": (True, False), "se": 1.1},
}
EXPOSURE_RATES: dict[str, FloatArray | None] = {
  "no_rates": None,
  "rates": numpy.array([0.45, 0.1, 0.32, 0.0, 0.29, 0.6, 0.31, 0.05, 0.3, 0.12, 0.5, 0.2]),
  "rates_all_capped": numpy.array([0.55, 0.4, 0.62, 0.31, 0.47, 0.9, 0.35, 0.33, 0.71, 0.5, 0.38, 0.44]),
}


def _ranking(make: Callable[[ItemBank], BaseSelector], bank: ItemBank, settings: dict[str, Any]) -> list[object]:
  """Return a selector's successive choices as each chosen item is excluded, ending with the error that stopped it.

  One generator, seeded with ``SELECTION_SEED``, serves every choice. The list holds up to
  ``RANKING_LENGTH`` item indices, then the name of ``NoItemsAvailableError`` if the selector ran out.
  """
  selector = make(bank)
  context = adapters.step_context(bank, rng=numpy.random.default_rng(SELECTION_SEED), **settings)
  choices: list[object] = []
  for _ in range(RANKING_LENGTH):
    try:
      item = selector.select(context)
    except NoItemsAvailableError:
      choices.append(NoItemsAvailableError.__name__)
      break
    choices.append(item)
    context = adapters.excluding(context, item)
  return choices


def _selection_contexts(selector: str) -> dict[str, tuple[ItemBank, dict[str, Any]]]:
  """Return the banks and contexts a selector is asked in, by label."""
  contexts: dict[str, tuple[ItemBank, dict[str, Any]]] = {}
  rate_labels = list(EXPOSURE_RATES) if selector in CAPPED else ["no_rates", "rates"]
  for name, settings in SELECTION_CONTEXTS.items():
    for rates in rate_labels:
      contexts[f"4pl|{name}|{rates}"] = (BANKS["4pl"], {**settings, "exposure_rates": EXPOSURE_RATES[rates]})
  # KL integrates its divergence over quadrature nodes in node order, so an item mirrored around the
  # estimate adds the same terms in reverse order, and whether the two tie depends on how the sums
  # round (ADR-0017). That differs between platforms, so the tie bank leaves KL out.
  if not selector.startswith("KLSelector"):
    for name, settings in TIE_CONTEXTS.items():
      contexts[f"tie_bank|{name}"] = (TIE_BANK, settings)
  return contexts


def _strata(make: Callable[[ItemBank], BaseSelector], bank: ItemBank) -> list[object]:
  selector = make(bank)
  assert isinstance(selector, StratifiedSelector)
  return list(selector.strata(bank))


def _selection_cases() -> dict[str, Case]:
  cases: dict[str, Case] = {
    f"{label}|{context}": functools.partial(_ranking, make, bank, settings)
    for label, make in SELECTORS.items()
    for context, (bank, settings) in _selection_contexts(label).items()
  }
  for label, make in SELECTORS.items():
    if "Strat" in label:
      cases[f"{label}|strata|4pl"] = functools.partial(_strata, make, BANKS["4pl"])
      cases[f"{label}|strata|whole_test_bank"] = functools.partial(_strata, make, WHOLE_TEST_BANK)
  for theta in (-1.1, 0.0, 0.42, 2.7):
    labels = _clusters(BANKS["4pl"])
    cases[f"ClusterSelector.sum_cluster_infos|theta={theta}"] = functools.partial(
      ClusterSelector.sum_cluster_infos, theta, BANKS["4pl"], labels
    )
    cases[f"ClusterSelector.weighted_cluster_infos|theta={theta}"] = functools.partial(
      ClusterSelector.weighted_cluster_infos, theta, BANKS["4pl"], labels
    )
  cases |= _selection_helper_cases()
  return cases


def _pick_best_draws(candidates: IntArray, scores: FloatArray) -> list[int]:
  """Return eight successive choices of :func:`pick_best` from one seeded generator."""
  rng = numpy.random.default_rng(SELECTION_SEED)
  return [pick_best(candidates, scores, rng) for _ in range(8)]


def _selection_helper_cases() -> dict[str, Case]:
  cases: dict[str, Case] = {}
  for name, settings in SELECTION_CONTEXTS.items():
    cases[f"remaining|{name}"] = functools.partial(
      lambda settings: remaining(adapters.step_context(BANKS["4pl"], rng=numpy.random.default_rng(0), **settings)),
      settings,
    )
  candidates = numpy.array([0, 2, 3, 5, 6, 8, 9, 11])
  for name, rates in EXPOSURE_RATES.items():
    for r_max in (0.3, 0.5, 1.0):
      cases[f"below_cap|{name}|r_max={r_max}"] = functools.partial(below_cap, candidates, rates, r_max)
  tied = numpy.array([0.5, 0.9, 0.9, 0.1, 0.9, 0.2, 0.9, 0.3])
  unique = numpy.array([0.5, 0.9, 0.7, 0.1, 0.8, 0.2, 0.6, 0.3])
  for name, scores in (("tied", tied), ("unique", unique)):
    cases[f"pick_best|{name}"] = functools.partial(_pick_best_draws, candidates, scores)
    for size in (1, 2, 3, 6, 8):
      cases[f"best_pool|{name}|size={size}"] = functools.partial(best_pool, candidates, scores, size)
  return cases


SELECTION_EXEMPT = {
  "BaseSelector": "abstract",
  "StratifiedSelector": "abstract; its strata method is recorded for every stratified selector",
  "CapFallback": "an enum of settings, used by the ClusterSelector cases",
  "ClusterMethod": "an enum of settings, used by the ClusterSelector cases",
  "check_r_max": "an input check that returns its argument",
  "check_test_size": "an input check that returns its argument",
}

# The stopping family.

STOPPERS: dict[str, Callable[[], BaseStopper]] = {
  "MaxLengthStopper(1)": lambda: MaxLengthStopper(1),
  "MaxLengthStopper(5)": lambda: MaxLengthStopper(5),
  "StandardErrorStopper(0.3)": lambda: StandardErrorStopper(0.3),
  "StandardErrorStopper(0.5)": lambda: StandardErrorStopper(0.5),
  "ConfidenceIntervalStopper((-1, 0, 1))": lambda: ConfidenceIntervalStopper((-1.0, 0.0, 1.0)),
  "ConfidenceIntervalStopper((0.5,), confidence=0.9)": lambda: ConfidenceIntervalStopper((0.5,), confidence=0.9),
  "ConfidenceIntervalStopper((-2, -0.5, 0.5, 2), confidence=0.99)": lambda: ConfidenceIntervalStopper(
    (-2.0, -0.5, 0.5, 2.0), confidence=0.99
  ),
  "StopRule(SE(0.4), MaxLength(6))": lambda: StopRule(StandardErrorStopper(0.4), MaxLengthStopper(6)),
  "StopRule(SE(0.4), MaxLength(6), min_items=3)": lambda: StopRule(
    StandardErrorStopper(0.4), MaxLengthStopper(6), min_items=3
  ),
  "StopRule(CI((-1, 1)), SE(0.3))": lambda: StopRule(ConfidenceIntervalStopper((-1.0, 1.0)), StandardErrorStopper(0.3)),
  "StopRule(StopRule(SE(0.35), MaxLength(8)), CI((0,)), min_items=2)": lambda: StopRule(
    StopRule(StandardErrorStopper(0.35), MaxLengthStopper(8)), ConfidenceIntervalStopper((0.0,)), min_items=2
  ),
}


def _stopping_contexts() -> dict[str, dict[str, Any]]:
  """Return the sessions the stoppers judge, by label: lengths, estimates and standard errors on a grid."""
  responses = (True, False, True, True, False, False, True, False)
  contexts: dict[str, dict[str, Any]] = {}
  for length in (0, 1, 3, 5, 6, 8):
    for theta in (-1.5, -0.2, 0.05, 0.9, 2.2):
      # A standard error is infinite before the first response and finite after it.
      for se in (1.2, 0.5, 0.4, 0.31, 0.3, 0.29, 0.1) if length else (math.inf,):
        contexts[f"n={length}|theta={theta}|se={se}"] = {
          "administered_items": ITEM_ORDER[:length],
          "responses": responses[:length],
          "theta": theta,
          "se": se,
        }
  return contexts


STOPPING_CONTEXTS = _stopping_contexts()


def _stops(make: Callable[[], BaseStopper]) -> dict[str, object]:
  """Return a stopper's decision on every stopping context, and for a rule which of its stoppers hold."""
  stopper = make()
  outputs: dict[str, object] = {}
  for label, settings in STOPPING_CONTEXTS.items():
    context = adapters.step_context(BANKS["4pl"], rng=numpy.random.default_rng(0), **settings)
    decision = stopper.stop(context)
    if isinstance(stopper, StopRule):
      held = stopper.satisfied(context)
      outputs[label] = [
        decision,
        [index for index, part in enumerate(stopper.stoppers) if any(part is h for h in held)],
      ]
    else:
      outputs[label] = decision
  return outputs


def _stopping_cases() -> dict[str, Case]:
  return {label: functools.partial(_stops, make) for label, make in STOPPERS.items()}


STOPPING_EXEMPT = {"BaseStopper": "abstract"}

# The initialization family.

INITIALIZERS: dict[str, Callable[[], BaseInitializer]] = {
  "FixedPointInitializer(0.7)": lambda: FixedPointInitializer(0.7),
  "FixedPointInitializer(-1.3)": lambda: FixedPointInitializer(-1.3),
  "NormalInitializer()": NormalInitializer,
  "NormalInitializer(0.5, 2.0)": lambda: NormalInitializer(0.5, 2.0),
  "UniformInitializer()": UniformInitializer,
  "UniformInitializer(-2.0, 3.0)": lambda: UniformInitializer(-2.0, 3.0),
}


def _initial_values(make: Callable[[], BaseInitializer], seed: int) -> list[float]:
  """Return five successive initial estimates drawn from one generator seeded with ``seed``."""
  initializer, rng = make(), numpy.random.default_rng(seed)
  return [initializer.initialize(item_bank=BANKS["4pl"], rng=rng) for _ in range(5)]


def _initialization_cases() -> dict[str, Case]:
  return {
    f"{label}|seed={seed}": functools.partial(_initial_values, make, seed)
    for label, make in INITIALIZERS.items()
    for seed in (0, 1, 42)
  }


INITIALIZATION_EXEMPT = {"BaseInitializer": "abstract"}

# The metrics family.

_FIFTY = numpy.arange(50, dtype=float)
METRIC_ARRAYS: dict[str, tuple[FloatArray, FloatArray]] = {
  "five": (numpy.array([0.5, -1.2, 2.0, 0.0, -0.3]), numpy.array([0.7, -1.0, 1.6, 0.1, -0.35])),
  "identical": (numpy.array([0.5, -1.2, 2.0]), numpy.array([0.5, -1.2, 2.0])),
  "one": (numpy.array([1.25]), numpy.array([-0.75])),
  "fifty": ((_FIFTY * 0.6180339887498949) % 1.0 * 6.0 - 3.0, (_FIFTY * 0.4142135623730951) % 1.0 * 6.0 - 3.0),
}
OVERLAP_INPUTS: dict[str, tuple[FloatArray, int]] = {
  "four_items": (numpy.array([1.0, 0.5, 0.5, 0.0]), 2),
  "even": (numpy.full(8, 0.25), 2),
  "same_test": (numpy.ones(4), 4),
  "rates": (numpy.array([0.45, 0.1, 0.32, 0.0, 0.29, 0.6, 0.31, 0.05, 0.3, 0.12, 0.5, 0.2]), 3),
}


# Public functions of catsim.metrics that no case is named after, with the reason. Added after adda72dc by ADR-0016,
# with the Monte Carlo errors, and tested by tests/test_metrics.py.
METRICS_EXEMPT = {
  "mean": "the mean of a per-session statistic, added after adda72dc",
  "exposure_rates": "exposure rates from counts, added after adda72dc; the whole tests record the rates of their runs",
}


def _metrics_cases() -> dict[str, Case]:
  cases: dict[str, Case] = {}
  for name, (actual, predicted) in METRIC_ARRAYS.items():
    cases[f"bias|{name}"] = functools.partial(adapters.bias, actual, predicted)
    cases[f"mse|{name}"] = functools.partial(adapters.mse, actual, predicted)
    cases[f"rmse|{name}"] = functools.partial(adapters.rmse, actual, predicted)
  for name, (rates, test_size) in OVERLAP_INPUTS.items():
    cases[f"overlap_rate|{name}"] = functools.partial(adapters.overlap_rate, rates, test_size)
  return cases


# The whole_tests family.


def _recorded_responses(n_examinees: int, item_bank: ItemBank) -> tuple[FloatArray, list[float]]:
  """Return a response matrix with an ability per row, and those abilities.

  Examinee ``i`` of ability ``theta_i`` answers item ``j`` correctly when ``b_j - theta_i`` is below
  a noise term that cycles through eleven values. Only comparisons and exact arithmetic decide the
  answers, so the matrix is the same on every platform. The last two examinees answer every item
  correctly and every item incorrectly.
  """
  difficulty = item_bank.difficulty
  thetas = [-2.25 + 0.5 * i for i in range(n_examinees - 2)]
  rows = [
    [float(difficulty[j] - theta < (((i * 7 + j * 13) % 11) - 5) * 0.3) for j in range(item_bank.n_items)]
    for i, theta in enumerate(thetas)
  ]
  rows += [[1.0] * item_bank.n_items, [0.0] * item_bank.n_items]
  return numpy.array(rows), [*thetas, 3.5, -3.5]


RESPONSES, TRUE_THETAS = _recorded_responses(12, WHOLE_TEST_BANK)


def _with_gaps(responses: FloatArray) -> FloatArray:
  """Return the responses with gaps: every third item of the first six examinees, all but ten of the seventh's."""
  gapped = responses.copy()
  for i in range(6):
    gapped[i, [j for j in range(gapped.shape[1]) if (i + j) % 3 == 0]] = numpy.nan
  gapped[6, [j for j in range(gapped.shape[1]) if j % 3 != 0]] = numpy.nan
  return gapped


GAPPED_RESPONSES = _with_gaps(RESPONSES)


def _bank_clusters() -> list[int]:
  return _clusters(WHOLE_TEST_BANK)


# The strategies of each whole test, built fresh for every run. None of them draws a random number on
# a bank without ties, which the generator checks by running each test with two seeds. The thresholds
# of the StopRule tests end some sessions by standard error and others by length, and two sessions of
# the min_items test reach its standard error at the third item and are held to the fourth.
WHOLE_TESTS: dict[str, Callable[[], dict[str, Any]]] = {
  "MaxInfo|NumericalSearch|MaxLength(10)": lambda: {
    "selector": MaxInfoSelector(),
    "estimator": NumericalSearchEstimator(),
    "stopper": MaxLengthStopper(10),
  },
  "MaxInfo|EAP|StopRule(SE(0.49), MaxLength(15))": lambda: {
    "selector": MaxInfoSelector(),
    "estimator": EAPEstimator(),
    "stopper": StopRule(StandardErrorStopper(0.49), MaxLengthStopper(15)),
  },
  "Urry|MAP|MaxLength(8)": lambda: {
    "selector": UrrySelector(),
    "estimator": MAPEstimator(),
    "stopper": MaxLengthStopper(8),
  },
  "IntervalInfo|EAP(grid=uniform(33, -4, 4))|StopRule(CI((-1, 0, 1)), MaxLength(12))": lambda: {
    "selector": IntervalInfoSelector(),
    "estimator": EAPEstimator(grid=QuadratureGrid.uniform(33, -4.0, 4.0)),
    "stopper": StopRule(ConfidenceIntervalStopper((-1.0, 0.0, 1.0)), MaxLengthStopper(12)),
  },
  "KL|WLE|MaxLength(10)": lambda: {
    "selector": KLSelector(),
    "estimator": WarmLikelihoodEstimator(),
    "stopper": MaxLengthStopper(10),
  },
  "MEI(EAP)|EAP|MaxLength(6)": lambda: {
    "selector": MEISelector(estimator=EAPEstimator()),
    "estimator": EAPEstimator(),
    "stopper": MaxLengthStopper(6),
  },
  "MLWI|MAP(log_prior=uniform())|MaxLength(8)": lambda: {
    "selector": MLWISelector(),
    "estimator": MAPEstimator(log_prior=uniform_log_prior()),
    "stopper": MaxLengthStopper(8),
  },
  "Linear(10 items)|NumericalSearch|MaxLength(15)": lambda: {
    "selector": LinearSelector(_linear_items(WHOLE_TEST_BANK)),
    "estimator": NumericalSearchEstimator(),
    "stopper": MaxLengthStopper(15),
  },
  "AStrat(10, 5)|EAP|MaxLength(10)": lambda: {
    "selector": AStratSelector(test_size=10, n_strata=5),
    "estimator": EAPEstimator(),
    "stopper": MaxLengthStopper(10),
  },
  "AStratBBlock(10, 5)|WLE|MaxLength(10)": lambda: {
    "selector": AStratBBlockSelector(test_size=10, n_strata=5),
    "estimator": WarmLikelihoodEstimator(),
    "stopper": MaxLengthStopper(10),
  },
  "MaxInfoStrat(10, 5)|MAP|MaxLength(10)": lambda: {
    "selector": MaxInfoStratSelector(test_size=10, n_strata=5),
    "estimator": MAPEstimator(),
    "stopper": MaxLengthStopper(10),
  },
  "MaxInfoBBlock(10, 5)|NumericalSearch|MaxLength(10)": lambda: {
    "selector": MaxInfoBBlockSelector(test_size=10, n_strata=5),
    "estimator": NumericalSearchEstimator(),
    "stopper": MaxLengthStopper(10),
  },
  "Cluster(cluster_info)|EAP|MaxLength(10)": lambda: {
    "selector": ClusterSelector(_bank_clusters(), method=ClusterMethod.CLUSTER_INFO),
    "estimator": EAPEstimator(),
    "stopper": MaxLengthStopper(10),
  },
  "MaxInfo|EAP|StopRule(SE(0.65), MaxLength(7), min_items=4)|start=-0.8|true_thetas": lambda: {
    "initializer": FixedPointInitializer(-0.8),
    "selector": MaxInfoSelector(),
    "estimator": EAPEstimator(),
    "stopper": StopRule(StandardErrorStopper(0.65), MaxLengthStopper(7), min_items=4),
    "true_thetas": TRUE_THETAS,
  },
  "MaxInfo|EAP|MaxLength(8)|decline 6 items": lambda: {
    "selector": MaxInfoSelector(),
    "estimator": EAPEstimator(),
    "stopper": MaxLengthStopper(8),
    "exposure_control": adapters.DeclineListed((0, 3, 7, 12, 19, 25)),
  },
  "MaxInfo|NumericalSearch|MaxLength(12)|missing responses": lambda: {
    "selector": MaxInfoSelector(),
    "estimator": NumericalSearchEstimator(),
    "stopper": MaxLengthStopper(12),
    "responses": GAPPED_RESPONSES,
  },
  "MaxInfo(r_max=0.25)|EAP|MaxLength(6)|exposure feedback": lambda: {
    "selector": MaxInfoSelector(r_max=0.25),
    "estimator": EAPEstimator(),
    "stopper": MaxLengthStopper(6),
  },
  "Cluster(item_info, r_max=0.4)|MAP|MaxLength(5)|exposure feedback": lambda: {
    "selector": ClusterSelector(_bank_clusters(), r_max=0.4),
    "estimator": MAPEstimator(),
    "stopper": MaxLengthStopper(5),
  },
}


def whole_test(name: str, seed: int = WHOLE_TEST_SEED) -> dict[str, object]:
  """Return the sessions and exposure of a whole test, run on its recorded responses with ``seed``.

  Parameters
  ----------
  name : str
      A key of ``WHOLE_TESTS``.
  seed : int, optional
      Seed of the run's random draws, which the configurations must not use. Default is
      ``WHOLE_TEST_SEED``.

  Returns
  -------
  dict[str, object]
      The run's sessions, each with its items, responses, estimates, standard errors, excluded
      items and stop reason (and, under a ``StopRule``, the positions of its
      stoppers that hold at the end), and the run's exposure and proposal counts and rates, overlap
      rate and, with true abilities, bias, MSE and RMSE.
  """
  settings = WHOLE_TESTS[name]()
  stopper = settings["stopper"]
  assert isinstance(stopper, BaseStopper)
  run = adapters.run_recorded(
    adapters.build_engine(
      WHOLE_TEST_BANK,
      settings.get("initializer", FixedPointInitializer(0.0)),
      settings["selector"],
      settings["estimator"],
      stopper,
      exposure_control=settings.get("exposure_control"),
    ),
    settings.get("responses", RESPONSES),
    rng=seed,
    true_thetas=settings.get("true_thetas"),
  )
  sessions = []
  for record in run.sessions:
    session: dict[str, object] = {
      "administered_items": record.administered_items,
      "responses": record.responses,
      "thetas": record.thetas,
      "ses": record.ses,
      "excluded_items": record.excluded_items,
      "stop_reason": record.stop_reason,
    }
    if isinstance(stopper, StopRule):
      held = stopper.satisfied(record.final_context)
      session["satisfied"] = [index for index, part in enumerate(stopper.stoppers) if any(part is h for h in held)]
    sessions.append(session)
  return {
    "sessions": sessions,
    "exposure_counts": run.exposure_counts,
    "proposal_counts": run.proposal_counts,
    "exposure_rates": run.exposure_rates,
    "proposal_rates": run.proposal_rates,
    "overlap_rate": run.overlap_rate,
    "bias": run.bias,
    "mse": run.mse,
    "rmse": run.rmse,
  }


# Public members of SimulationResult that the whole tests do not record, with the reason. The whole tests record the
# members their outputs are named after: the sessions, the exposure and proposal counts and rates, the overlap rate and
# the accuracy metrics.
RESULT_EXEMPT = {
  "item_bank": "the bank the whole test runs on, an input",
  "true_thetas": "the true abilities the whole test is given, an input",
  "parts": "the number of parts, an input; every whole test runs in one part",
  "duration": "the wall-clock time of the run, which changes from run to run",
  # Added after adda72dc by ADR-0016 from the sessions the whole tests record, and tested by tests/test_simulation.py.
  "steps": "the table of the sessions' steps, added after adda72dc",
  "examinees": "the table of the sessions, added after adda72dc",
  "deciles": "the metrics by decile of the true ability, added after adda72dc",
  "ability_levels": "the metrics at each distinct true ability, added after adda72dc",
  "test_length": "the mean length of the sessions, added after adda72dc",
  "se": "the mean standard error of the sessions' final estimates, added after adda72dc",
}


def _whole_test_cases() -> dict[str, Case]:
  cases: dict[str, Case] = {name: functools.partial(whole_test, name) for name in WHOLE_TESTS}
  cases["inputs|responses"] = lambda: RESPONSES
  cases["inputs|gapped_responses"] = lambda: GAPPED_RESPONSES
  return cases


# The families, and what they cover.

FAMILIES: dict[str, dict[str, Case]] = {
  "estimation": _estimation_cases(),
  "selection": _selection_cases(),
  "stopping": _stopping_cases(),
  "initialization": _initialization_cases(),
  "metrics": _metrics_cases(),
  "whole_tests": _whole_test_cases(),
}


# The estimators whose estimate ends a numerical search, so that it is defined only to within the
# search's tolerance (see tests/reference/records.py).
SEARCHING_ESTIMATORS = (NumericalSearchEstimator, MAPEstimator, WarmLikelihoodEstimator)


@dataclass(frozen=True, slots=True)
class Comparison:
  """How a case's floats compare with their recorded values (see :mod:`tests.reference.records`).

  Parameters
  ----------
  unit : float
      1 for abilities, standard errors and the accuracy of estimates, which can cancel to rounding
      noise near 0; 0 for every other float, which compares relatively.
  searched : bool
      Whether the floats come from an estimator's numerical search, which is defined only to
      within its tolerance.
  """

  unit: float
  searched: bool


def comparison(family: str, name: str) -> Comparison:
  """Return how the floats of a case compare with their recorded values.

  Parameters
  ----------
  family : str
      A key of ``FAMILIES``.
  name : str
      A case of that family.

  Returns
  -------
  Comparison
      The case's unit, and whether its floats come from a numerical search.
  """
  head = name.split("|", 1)[0]
  if family == "estimation" and head in ESTIMATORS:
    return Comparison(unit=1.0, searched=isinstance(ESTIMATORS[head](), SEARCHING_ESTIMATORS))
  if family == "whole_tests" and name in WHOLE_TESTS:
    return Comparison(unit=1.0, searched=isinstance(WHOLE_TESTS[name]()["estimator"], SEARCHING_ESTIMATORS))
  on_scale = family == "metrics" and head in {"bias", "mse", "rmse"}
  return Comparison(unit=1.0 if on_scale else 0.0, searched=False)


def _named(family: str, name: str) -> bool:
  """Return whether a case of the family is named after ``name``: the name, then ``|``, ``(`` or ``.``."""
  return any(case == name or case.startswith((f"{name}|", f"{name}(", f"{name}.")) for case in FAMILIES[family])


def _concrete_subclasses(package: str, base: type) -> set[type]:
  """Return the concrete subclasses of ``base`` that a package's ``__all__`` exports."""
  module = importlib.import_module(package)
  exported = [getattr(module, name) for name in module.__all__]
  return {
    value for value in exported if inspect.isclass(value) and issubclass(value, base) and not inspect.isabstract(value)
  }


def _module_functions(module_name: str) -> set[str]:
  """Return the public functions a module defines."""
  module = importlib.import_module(module_name)
  return {
    name
    for name, value in vars(module).items()
    if not name.startswith("_") and inspect.isfunction(value) and value.__module__ == module_name
  }


# The strategy packages, each with its base class, the strategies the cases configure, the family
# whose case names cover its other exports, and the exports exempt from cases.
_STRATEGY_PACKAGES: dict[str, tuple[type, Callable[[], set[type]], str, dict[str, str]]] = {
  "catsim.selection": (
    BaseSelector,
    lambda: {type(make(BANKS["4pl"])) for make in SELECTORS.values()},
    "selection",
    SELECTION_EXEMPT,
  ),
  "catsim.estimation": (
    BaseEstimator,
    lambda: {type(make()) for make in ESTIMATORS.values()},
    "estimation",
    ESTIMATION_EXEMPT,
  ),
  "catsim.stopping": (BaseStopper, lambda: {type(make()) for make in STOPPERS.values()}, "stopping", STOPPING_EXEMPT),
  "catsim.initialization": (
    BaseInitializer,
    lambda: {type(make()) for make in INITIALIZERS.values()},
    "initialization",
    INITIALIZATION_EXEMPT,
  ),
  # catsim.exposure exports only its base class, so a concrete control it exports needs cases of its own.
  "catsim.exposure": (BaseExposureControl, set, "whole_tests", {"BaseExposureControl": "abstract"}),
}


def uncovered() -> list[str]:
  """Return every public strategy class, computational function and result member that no case exercises.

  Every concrete strategy class a strategy package exports must be the type of a strategy the cases
  configure. Every other name a strategy package exports, and every public function of
  ``catsim.metrics``, must name a case or be exempt with a reason. Every public member of
  ``SimulationResult`` must name an output of the whole tests, or be exempt with a reason.

  Returns
  -------
  list[str]
      The names, sorted; empty when the cases cover everything.
  """
  missing: set[str] = set()
  for package, (base, configured, family, exempt) in _STRATEGY_PACKAGES.items():
    strategies = configured()
    missing |= {f"{package}.{cls.__name__}" for cls in _concrete_subclasses(package, base) - strategies}
    names = {cls.__name__ for cls in strategies} | exempt.keys()
    exports = importlib.import_module(package).__all__
    missing |= {f"{package}.{name}" for name in exports if name not in names and not _named(family, name)}
  functions = _module_functions("catsim.metrics") - METRICS_EXEMPT.keys()
  missing |= {f"catsim.metrics.{name}" for name in functions if not _named("metrics", name)}
  recorded = set(whole_test(next(iter(WHOLE_TESTS)))) | RESULT_EXEMPT.keys()
  results = {name for name in dir(SimulationResult) if not name.startswith("_")}
  missing |= {f"catsim.SimulationResult.{name}" for name in results - recorded}
  return sorted(missing)
