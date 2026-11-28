"""Parity tests with catR, catsim's reference implementation (ADR-0020).

The expected values are catR's outputs on fixed cases, which ``tests/parity/catr/generate.R`` wrote, with the R and catR
versions, into the JSON files beside it; CONTRIBUTING.md says how to regenerate them. The tests need no R. Each case
records catR's settings, and the tests build the catsim strategy that matches them. Item indices in the files are
0-based, as in catsim. The cases come in ADR-0020's three levels: single computations (probabilities, estimators,
selectors, schedules and stopping rules), whole tests on recorded responses (``TestWholeTests``) and simulations of
the randomized selectors, compared within a conservative bound on their Monte Carlo error (``TestSimulations``).

A case where catsim departs from catR on purpose, with no setting that reproduces catR, is a strict expected failure
with the reason (``_DEPARTURES``): it must fail its assertion, so a departure that closes, or a case that stops
departing, fails the run.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest
from irtlib import ItemBank
from irtlib.quadrature import LogPrior, QuadratureGrid, normal_log_prior, uniform_log_prior
from irtlib.typing import FloatArray

from catsim import CatEngine, StepContext, simulate
from catsim.estimation import (
  BaseEstimator,
  EAPEstimator,
  MAPEstimator,
  NumericalSearchEstimator,
  WarmLikelihoodEstimator,
)
from catsim.initialization import FixedPointInitializer
from catsim.selection import (
  BaseSelector,
  KLSelector,
  MaxInfoSelector,
  MEISelector,
  MLWISelector,
  ProgressiveSelector,
  ProportionalSelector,
  RandomesqueSelector,
  The54321Selector,
  UrrySelector,
)
from catsim.selection.progressive import _position_weight  # noqa: PLC2701 - the selectors expose no weight
from catsim.stopping import BaseStopper, ConfidenceIntervalStopper, MaxLengthStopper, StandardErrorStopper, StopRule
from tests.banks import two_pl_bank
from tests.generators import DeterministicRNG

if TYPE_CHECKING:
  from collections.abc import Callable, Sequence

  from catsim import Session

Case = dict[str, Any]

_DATA = Path(__file__).parent / "parity" / "catr"


def _load(name: str) -> dict[str, Any]:
  with (_DATA / f"{name}.json").open(encoding="utf-8") as file:
    return json.load(file)


_ESTIMATION, _SELECTION, _STOPPING, _WHOLE_TESTS, _SIMULATIONS = (
  _load(name) for name in ("estimation", "selection", "stopping", "whole_tests", "simulations")
)

# Closed forms, quadratures on the same nodes with the same weights, and KL's quadratures, which both packages take to
# convergence: a relative 1e-10 (ADR-0020).
_CLOSED_FORM_RTOL = 1e-10
# catR's thetaEst solves its estimating equations with uniroot at its default tolerance, .Machine$double.eps ** 0.25,
# which is 2 ** -13; catsim's estimators search to their own tol, 1e-6. Each estimate may miss the root by its
# package's tolerance, so the two may differ by their sum.
_ESTIMATE_ATOL = 2.0**-13 + 1e-6

_ADR_18_DODD = (
  "ADR-0018: after an all-correct pattern catsim's Dodd step never moves the estimate down, so an estimate already "
  "above the bank's hardest item stays put; catR's constantPatt = 'var' moves it halfway back to that item"
)
_DODD_SE = (
  "after Dodd's step catsim reports the Fisher standard error at the new estimate; catR's semTheta reports an infinite "
  "one for any constantPatt other than BM, EAP and WL"
)
_CONSTANT_DEFAULT = (
  "catR's default for all-correct and all-incorrect patterns (constantPatt = NULL) returns the bound of its range; "
  "catsim's NumericalSearchEstimator always takes Dodd's step (ADR-0018), which catR's constantPatt = 'var' matches"
)
_ML_MINIMUM = (
  "the likelihood has a local minimum inside the range; catR's thetaEst solves the likelihood equation with uniroot, "
  "which converges to that minimum, while catsim maximizes the likelihood"
)
_WL_ROOTS = (
  "Warm's modified score has three roots in the range, two maxima of the weighted likelihood and a minimum; catR's "
  "uniroot converges to the lower maximum (-0.59), and catsim returns the higher one (0.66), as its technique page "
  "states"
)
_WL_ROOTS_WHOLE_TEST = (
  "after items 21 and 38, answered correctly and incorrectly, Warm's modified score has three roots in the range, "
  "-0.44, 0.63 and 0.84; catR's uniroot converges to 0.84, and catsim returns -0.44, the root of larger weighted "
  "likelihood, as in wl/three_wl_roots. MLWI gives these two items first whatever the estimate, and the later "
  "estimates agree again"
)
_MLWI_VALUES = (
  "catsim's MLWISelector weights its grid nodes equally, where catR's MWI integrates by the trapezoid rule, which "
  "halves the weight of the two end nodes; the scores differ by up to 2% before the first item, where the "
  "likelihood is 1 at those nodes, and no MLWISelector setting gives trapezoid weights"
)
_MEI_WHOLE_TEST = (
  "catsim's MEISelector scores the information of the administered test plus the candidate at each look-ahead "
  "estimate (van der Linden, 1998); catR's MEI scores the candidate's information alone, as the technique page "
  "states. The two agree only before the first item"
)
_KL_LIKELIHOOD = (
  "catR's nextItem KL weights the divergence by the likelihood of the responses over parInt; catsim's KLSelector "
  "integrates it unweighted over the window theta +- c / sqrt(n) (Chang and Ying, 1996), and no KLSelector setting "
  "weights it. The kl_window cases compare the divergence through catR's KL() with the likelihood set to 1"
)
_PAST_TEST_SIZE = (
  "ADR-0019: past the planned length the weight stays 1 and the exponent N; catR's sums keep growing, so its weight "
  "exceeds 1"
)
_ONE_ITEM_TEST = (
  "ADR-0019: a one-item test is at its last position from the start, with weight 1; catR gives weight 0 at the "
  "first position whatever the test length"
)

# Each test's cases that catsim departs from on purpose, by case id, with the reason: the case must fail its assertion.
_DEPARTURES: dict[str, dict[str, str]] = {
  "estimate": {
    "ml_dodd_past_hardest/three_correct": _ADR_18_DODD,
    "ml_constant_default/three_correct": _CONSTANT_DEFAULT,
    "ml_constant_default/three_incorrect": _CONSTANT_DEFAULT,
    "ml/two_stationary_points": _ML_MINIMUM,
    "ml/rising_at_bound": _ML_MINIMUM,
    "wl/three_wl_roots": _WL_ROOTS,
  },
  "standard_error": {
    "ml_dodd/one_correct": _DODD_SE,
    "ml_dodd/three_incorrect": _DODD_SE,
    "ml_dodd/three_correct": _DODD_SE,
    "ml_dodd_past_hardest/three_correct": _DODD_SE,
    "ml_constant_default/three_correct": _CONSTANT_DEFAULT,
    "ml_constant_default/three_incorrect": _CONSTANT_DEFAULT,
    "ml/two_stationary_points": _ML_MINIMUM,
    "ml/rising_at_bound": _ML_MINIMUM,
    "wl/three_wl_roots": _WL_ROOTS,
  },
  "selection": {"kl/two_items": _KL_LIKELIHOOD},
  "values": {"mei/two_items": _MEI_WHOLE_TEST},
  "trapezoid_values": {
    f"mlwi/{context}": _MLWI_VALUES for context in ("start", "two_items", "five_items", "all_correct")
  },
  "schedule": {
    **{
      f"{rule}/past_test_size_N5_AP1_position{n}": _PAST_TEST_SIZE
      for rule in ("progressive", "proportional")
      for n in (6, 7)
    },
    **{f"{rule}/one_item_test_N1_AP1_position1": _ONE_ITEM_TEST for rule in ("progressive", "proportional")},
  },
  **{
    test: {f"mlwi_wl/simulation/theta_{theta}": _WL_ROOTS_WHOLE_TEST for theta in ("-1.2", "0", "1.2")}
    for test in ("whole_test_estimates", "whole_test_standard_errors")
  },
}


# The data file whose cases each key of _DEPARTURES names.
_DEPARTURE_DATA = {
  "estimate": _ESTIMATION,
  "standard_error": _ESTIMATION,
  "selection": _SELECTION,
  "values": _SELECTION,
  "trapezoid_values": _SELECTION,
  "schedule": _SELECTION,
  "whole_test_estimates": _WHOLE_TESTS,
  "whole_test_standard_errors": _WHOLE_TESTS,
}


def _cases(data: dict[str, Any], test: str, prefixes: tuple[str, ...] = ("",)) -> list[Any]:
  """Return the cases whose id starts with one of ``prefixes``, each a strict expected failure if catsim departs there.

  ``test`` names the entry of ``_DEPARTURES`` that holds the test's departures.
  """
  departures = _DEPARTURES.get(test, {})
  return [
    pytest.param(
      case,
      id=case["id"],
      marks=[pytest.mark.xfail(strict=True, raises=AssertionError, reason=departures[case["id"]])]
      if case["id"] in departures
      else [],
    )
    for case in data["cases"]
    if case["id"].startswith(prefixes)
  ]


def _floats(values: list[Any]) -> FloatArray:
  """Return a list of JSON numbers, or of the strings generate.R writes for infinities, as a float array."""
  return np.array([float(value) for value in values], dtype=float)


class TestData:
  """The recorded data are one run of the pinned catR, and the departures name recorded cases."""

  def test_one_catr_run(self) -> None:
    """Files written by different R or catR versions would compare catsim with two references at once."""
    versions = {
      (data["catr_version"], data["r_version"])
      for data in (_ESTIMATION, _SELECTION, _STOPPING, _WHOLE_TESTS, _SIMULATIONS)
    }

    assert len(versions) == 1

  def test_departures_name_recorded_cases(self) -> None:
    """A departure whose case was renamed, dropped or put under another test would mark nothing and hide the change."""
    for test, departures in _DEPARTURES.items():
      recorded = {case["id"] for case in _DEPARTURE_DATA[test]["cases"]}

      assert set(departures) <= recorded, test


def _log_prior(prior: dict[str, Any]) -> LogPrior:
  """Return catsim's log prior for catR's priorDist and priorPar: the normal's mean and sd, or the uniform's bounds."""
  first, second = prior["parameters"]
  if prior["distribution"] == "norm":
    return normal_log_prior(first, second)
  return uniform_log_prior(first, second)


def _trapezoid_grid(low: float, high: float, nodes: int) -> QuadratureGrid:
  """Return catR's EAP grid, parInt = c(low, high, nodes), with the trapezoid weights of its integrate.catR."""
  weights = np.full(nodes, (high - low) / (nodes - 1))
  weights[[0, -1]] /= 2
  return QuadratureGrid(nodes=np.linspace(low, high, nodes), weights=weights)


def _estimator(case: Case) -> BaseEstimator:
  """Return the catsim estimator that matches catR's method and settings in the case."""
  method = case["method"]
  if method == "EAP":
    return EAPEstimator(grid=_trapezoid_grid(**case["grid"]), log_prior=_log_prior(case["prior"]))
  bounds = (float(case["range"][0]), float(case["range"][1]))
  if method == "BM":
    return MAPEstimator(log_prior=_log_prior(case["prior"]), bounds=bounds)
  if method == "WL":
    return WarmLikelihoodEstimator(bounds=bounds)
  return NumericalSearchEstimator(bounds=bounds)


def _estimation_context(case: Case) -> StepContext:
  """Return the context of the case's response pattern, with catR's current.th as the previous estimate."""
  return StepContext(
    item_bank=ItemBank(np.array(_ESTIMATION["bank"], dtype=float)),
    rng=np.random.default_rng(0),
    administered_items=tuple(case["items"]),
    responses=tuple(case["responses"]),
    theta=float(case["current_theta"]),
  )


class TestEstimators:
  """catR's thetaEst and semTheta, with ML, BM, EAP and WL, against the matching catsim estimator.

  ML is NumericalSearchEstimator, BM is MAPEstimator, EAP is EAPEstimator on catR's grid with its trapezoid weights,
  and WL is WarmLikelihoodEstimator, each on catR's range. EAP is a quadrature on the same nodes, so it is compared as a
  closed form; the others search, and are compared within the estimates' tolerance.
  """

  @pytest.mark.parametrize("case", _cases(_ESTIMATION, "estimate"))
  def test_estimate(self, case: Case) -> None:
    """A wrong estimating equation, prior, range, grid or Dodd step would move the estimate off catR's thetaEst."""
    estimate = _estimator(case).estimate(_estimation_context(case))

    if case["method"] == "EAP":
      assert estimate.theta == pytest.approx(case["theta"], rel=_CLOSED_FORM_RTOL, abs=0)
    else:
      assert estimate.theta == pytest.approx(case["theta"], rel=0, abs=_ESTIMATE_ATOL)

  @pytest.mark.parametrize("case", _cases(_ESTIMATION, "standard_error"))
  def test_standard_error(self, case: Case) -> None:
    """A standard error that left out the prior's curvature, or added Warm's correction, would differ from semTheta's.

    Away from EAP the two standard errors are taken at estimates that differ by up to the estimates' tolerance, so
    they may differ by that much times the standard error's slope, which generate.R records; the 1 added to the slope
    leaves room for MAPEstimator's central-difference curvature of the prior.
    """
    estimate = _estimator(case).estimate(_estimation_context(case))

    if case["method"] == "EAP":
      assert estimate.se == pytest.approx(float(case["se"]), rel=_CLOSED_FORM_RTOL, abs=0)
    else:
      assert estimate.se == pytest.approx(float(case["se"]), rel=0, abs=_ESTIMATE_ATOL * (abs(case["se_slope"]) + 1))


_SELECTION_BANK = ItemBank(np.array(_SELECTION["bank"], dtype=float))


def _selection_context(case: Case, rng: np.random.Generator | None = None) -> StepContext:
  """Return the case's context: its administered items and responses, and its ability estimate."""
  return StepContext(
    item_bank=_SELECTION_BANK,
    rng=np.random.default_rng(0) if rng is None else rng,
    administered_items=tuple(case["administered"]),
    responses=tuple(case["responses"]),
    theta=float(case["theta"]),
  )


# catR's settings for the criteria in generate.R: parInt = c(-4, 4, 33), and for MEI the Bayes modal estimate with a
# standard normal prior on range = c(-4, 4) and Fisher information.
_MLWI = MLWISelector(n_nodes=33, bounds=(-4.0, 4.0))
_MEI = MEISelector(estimator=MAPEstimator(bounds=(-4.0, 4.0)))
_SELECTORS: dict[str, BaseSelector] = {
  "mfi": MaxInfoSelector(),
  "bopt": UrrySelector(),
  "mlwi": _MLWI,
  "mei": _MEI,
  "kl": KLSelector(),
  "kl_window": KLSelector(),
}


# The randomized selectors whose bin, the items they draw from, generate.R records as catR's randomesque bin.
_POOL_SELECTORS: dict[str, Callable[[Case], BaseSelector]] = {
  "randomesque": lambda case: RandomesqueSelector(case["bin_size"]),
  "the54321": lambda _: The54321Selector(),
}


def _selector(case: Case) -> BaseSelector:
  return _SELECTORS[case["id"].split("/")[0]]


class TestSelectors:
  """catR's nextItem with MFI, bOpt, MLWI, MEI and KL, and the criterion values behind it, against catsim's selectors.

  A case's ranking is the order in which catR picks every candidate when each pick is made unavailable for the next,
  its choice first; catsim's is the same with each pick excluded. The bank's parameters are continuous, and generate.R
  checks that no two candidates nearly tie. The kl_window cases rank by catR's KL() over catsim's window. The
  randomesque and the54321 cases compare the bin a randomized selector draws its item from.
  """

  @pytest.mark.parametrize("case", _cases(_SELECTION, "selection", tuple(f"{name}/" for name in _SELECTORS)))
  def test_ranking(self, case: Case) -> None:
    """A selector maximizing another criterion, or reading the wrong estimate or responses, would pick another item.

    The first item of the ranking is the selector's choice in the context; a criterion that agreed with catR's on that
    item only would order the other candidates differently.
    """
    context = _selection_context(case)
    ranking: list[int] = []
    for _ in case["candidates"]:
      ranking.append(_selector(case).select(dataclasses.replace(context, excluded_items=tuple(ranking))))

    assert ranking == case["ranking"]

  @pytest.mark.parametrize("case", _cases(_SELECTION, "pool", tuple(f"{name}/" for name in _POOL_SELECTORS)))
  def test_randomesque_pool(self, case: Case) -> None:
    """A bin of the wrong size, or of other items than the most informative at the estimate, would draw elsewhere.

    The pool is catR's randomesque bin, the items nextItem draws from with MFI and randomesque = k, best first. catR has
    no 5-4-3-2-1 method; its bin at each position is catR's randomesque bin with k = max(1, 5 - administered items).
    """
    rng = DeterministicRNG([])
    _POOL_SELECTORS[case["id"].split("/")[0]](case).select(_selection_context(case, rng=rng))
    ((pool, probabilities),) = rng.choice_calls

    assert probabilities is None
    assert pool.tolist() == case["pool"]

  @pytest.mark.parametrize("case", _cases(_SELECTION, "values", ("mfi/",)))
  def test_information_values(self, case: Case) -> None:
    """Information that MaxInfoSelector reads from the bank off catR's Ii would rank items by another criterion."""
    context = _selection_context(case)

    np.testing.assert_allclose(
      context.item_bank.information(context.theta)[case["candidates"]], _floats(case["values"]), rtol=_CLOSED_FORM_RTOL
    )

  @pytest.mark.parametrize("case", _cases(_SELECTION, "values", ("mlwi/",)))
  def test_mlwi_values(self, case: Case) -> None:
    """Scores that weighted the likelihood or the information otherwise than catR's MWI would differ in proportion.

    The expected scores are catR's integrand, the likelihood times the information at each node of its grid, summed
    with catsim's equal weights. catsim's scores are divided by the likelihood's sum, so both are divided by their
    largest.
    """
    scores = _MLWI._score(_selection_context(case), np.array(case["candidates"], dtype=np.intp))  # noqa: SLF001 - select shows only the best score
    expected = _floats(case["node_sums"])

    np.testing.assert_allclose(scores / np.max(scores), expected / np.max(expected), rtol=_CLOSED_FORM_RTOL)

  @pytest.mark.parametrize("case", _cases(_SELECTION, "trapezoid_values", ("mlwi/",)))
  def test_mlwi_trapezoid_values(self, case: Case) -> None:
    """Scores in proportion to catR's own MWI, a trapezoid rule, would mean the departure closed and is stale."""
    scores = _MLWI._score(_selection_context(case), np.array(case["candidates"], dtype=np.intp))  # noqa: SLF001 - select shows only the best score
    expected = _floats(case["values"])

    np.testing.assert_allclose(scores / np.max(scores), expected / np.max(expected), rtol=_CLOSED_FORM_RTOL)

  @pytest.mark.parametrize("case", _cases(_SELECTION, "values", ("mei/",)))
  def test_mei_values(self, case: Case) -> None:
    """Expected information with the response probabilities swapped, or the wrong look-ahead, would miss catR's MEI.

    The values depend on the look-ahead estimates, so each may differ by the estimates' tolerance times the steepest
    slope of its candidate's information, which generate.R records from catR's dIi.
    """
    context = _selection_context(case)
    # select shows only the best score
    scores = [_MEI._expected_information(context, item) for item in case["candidates"]]  # noqa: SLF001

    np.testing.assert_array_less(
      np.abs(np.array(scores) - _floats(case["values"])), _ESTIMATE_ATOL * _floats(case["value_slopes"])
    )

  @pytest.mark.parametrize("case", _cases(_SELECTION, "values", ("kl_window/",)))
  def test_kl_values(self, case: Case) -> None:
    """A divergence taken the wrong way round, or integrated inaccurately over the window, would miss catR's KL().

    The window is catR's, recorded by generate.R; whether KLSelector narrows its own window as c / sqrt(n) shows in
    its ranking, above, where the kl_window/window_sensitive case picks another item under the windows of n - 1 and
    n + 1 items.
    """
    context = _selection_context(case)
    items = context.item_bank.items[case["candidates"]]
    scores = KLSelector._kl_global(context.theta, items, float(case["half_width"]))  # noqa: SLF001 - select shows only the largest

    np.testing.assert_allclose(scores, _floats(case["values"]), rtol=_CLOSED_FORM_RTOL)


class TestSchedules:
  """catR's progressive and proportional rules with rule = "length", thr = N and AP = acceleration (ADR-0019).

  catR draws the progressive rule's random component with R's generator; the stand-in generator hands catsim the same
  uniform draws, which generate.R recorded. catR's proportional selection probabilities are read from nextItem.
  """

  @pytest.mark.parametrize("case", _cases(_SELECTION, "schedule", ("progressive/",)))
  def test_progressive_choice(self, case: Case) -> None:
    """A schedule that weighted information too much or too little at a position would blend to another item.

    The weight is checked against catR's itself, since one random draw per item rarely makes the choice depend on its
    exact value. With the same draws, catR's score of every item is catsim's times the largest available information,
    so the two choose the same item.
    """
    position = len(case["administered"]) + 1
    selector = ProgressiveSelector(test_size=case["test_size"], acceleration=case["acceleration"])

    assert _position_weight(position, case["test_size"], case["acceleration"]) == pytest.approx(
      case["weight"], rel=_CLOSED_FORM_RTOL, abs=0
    )
    assert selector.select(_selection_context(case, rng=DeterministicRNG(case["uniforms"]))) == case["item"]

  @pytest.mark.parametrize("case", _cases(_SELECTION, "schedule", ("proportional/",)))
  def test_proportional_probabilities(self, case: Case) -> None:
    """An exponent off Barrada et al.'s schedule, N times its weight, would draw with other probabilities than catR."""
    rng = DeterministicRNG([])
    ProportionalSelector(test_size=case["test_size"], acceleration=case["acceleration"]).select(
      _selection_context(case, rng=rng)
    )
    ((candidates, probabilities),) = rng.choice_calls

    assert probabilities is not None
    assert candidates.tolist() == case["candidates"]
    np.testing.assert_allclose(probabilities, _floats(case["probabilities"]), rtol=_CLOSED_FORM_RTOL)


def _stopper(rule: str, threshold: float, alpha: float) -> BaseStopper:
  """Return catsim's stopper for one of catR's stopping rules."""
  if rule == "length":
    return MaxLengthStopper(int(threshold))
  if rule == "precision":
    return StandardErrorStopper(threshold)
  return ConfidenceIntervalStopper([threshold], confidence=1 - alpha)


class TestStoppers:
  """catR's checkStopRule with the length, precision and classification rules, against catsim's stoppers.

  catR stops as soon as any of its rules holds, as a StopRule of the matching stoppers does. Its classification rule
  takes one threshold, the ConfidenceIntervalStopper with a single bound; several classification rules stop once the
  interval clears any one threshold, a StopRule of single-bound stoppers.
  """

  @pytest.mark.parametrize("case", _cases(_STOPPING, "stopping"))
  def test_decision(self, case: Case) -> None:
    """A stopper with the wrong comparison, strict at a length or precision threshold, or with the wrong z would differ.

    No classification case sits exactly on its boundary: R's qnorm and SciPy's norm.ppf differ by a few units in the
    last place, so an interval that ends exactly on the threshold in one package does not in the other. The cases 1e-9
    either side of it check where the boundary is.
    """
    stoppers = [
      _stopper(rule, threshold, case["alpha"])
      for rule, threshold in zip(case["rules"], case["thresholds"], strict=True)
    ]
    n_items = case["n_items"]
    context = StepContext(
      item_bank=two_pl_bank([1.0] * 12, np.linspace(-2, 2, 12).tolist()),
      rng=np.random.default_rng(0),
      administered_items=tuple(range(n_items)),
      responses=(True,) * n_items,
      theta=float(case["theta"]),
      se=float(case["se"]),
    )

    assert [stopper.stop(context) for stopper in stoppers] == case["per_rule"]
    assert StopRule(*stoppers).stop(context) is case["decision"]


def _stop_rule(case: Case) -> StopRule:
  """Return catsim's stopping rule for catR's stop list: its rules, each with its threshold, stop at whichever holds."""
  return StopRule(
    *(_stopper(rule, threshold, alpha=0.05) for rule, threshold in zip(case["rules"], case["thresholds"], strict=True))
  )


_WHOLE_TEST_SELECTORS: dict[str, BaseSelector] = {"MFI": MaxInfoSelector(), "bOpt": UrrySelector(), "MLWI": _MLWI}


@functools.cache
def _whole_test_sessions(configuration: str, bank: str) -> tuple[Session, ...]:
  """Return catsim's sessions of a bank's recorded examinees under a configuration of whole_tests.R, in examinee order.

  The selector chooses the first item at ability 0, as catR does with no early item; the selection rule, the estimator
  and the stopping rules are those matching catR's settings in the configuration's cases.
  """
  case = next(case for case in _WHOLE_TESTS["cases"] if case["id"].startswith(f"{configuration}/{bank}/"))
  engine = CatEngine(
    ItemBank(np.array(_WHOLE_TESTS["banks"][bank], dtype=float)),
    FixedPointInitializer(0.0),
    _WHOLE_TEST_SELECTORS[case["criterion"]],
    _estimator(case),
    _stop_rule(case),
  )
  # No item ties and no strategy draws, so the seed changes nothing.
  return tuple(simulate(engine, responses=np.array(_WHOLE_TESTS["examinees"][bank], dtype=bool), rng=0).sessions)


def _whole_test_session(case: Case) -> Session:
  configuration, bank, _ = case["id"].split("/")
  return _whole_test_sessions(configuration, bank)[case["examinee"]]


class TestWholeTests:
  """catR's randomCAT on recorded responses (level 2), against catsim's simulate on the same response patterns.

  Each case is one recorded examinee's test with MFI, bOpt or MLWI selection, the ML, BM, EAP or WL estimator of
  TestEstimators and catR's precision and length rules. catR starts with no early item at ability 0, as catsim's
  FixedPointInitializer(0) does; generate.R checks that catR's default start, one early item that startItems chooses at
  ability 0 by the test's own criterion, gives the same MFI and bOpt tests. It also checks that every choice and
  stopping decision survives a change of the estimate by the estimates' tolerance, so the cases do not hinge on the
  last digits.
  """

  @pytest.mark.parametrize("case", _cases(_WHOLE_TESTS, "whole_test_items"))
  def test_items(self, case: Case) -> None:
    """A selector, estimator or stopper that departed from catR's at any step would give another item or test length."""
    assert list(_whole_test_session(case).administered_items) == case["items"]

  @pytest.mark.parametrize("case", _cases(_WHOLE_TESTS, "whole_test_estimates"))
  def test_estimates(self, case: Case) -> None:
    """An estimate that read the responses of another step, or took Dodd's step from another estimate, would drift.

    The estimates are those after every response, the starting ability first, compared as in TestEstimators.
    """
    thetas = np.array([estimate.theta for estimate in _whole_test_session(case).estimates])
    expected = np.array(case["thetas"], dtype=float)

    assert thetas.shape == expected.shape
    if case["method"] == "EAP":
      np.testing.assert_allclose(thetas, expected, rtol=_CLOSED_FORM_RTOL, atol=0)
    else:
      np.testing.assert_allclose(thetas, expected, rtol=0, atol=_ESTIMATE_ATOL)

  @pytest.mark.parametrize("case", _cases(_WHOLE_TESTS, "whole_test_standard_errors"))
  def test_standard_errors(self, case: Case) -> None:
    """A standard error that stopped a precision rule at another item would come from the wrong items or estimate.

    They are compared as in TestEstimators, except where ML's pattern is still constant: there catR reports an infinite
    standard error and catsim the Fisher one at Dodd's estimate, the departure TestEstimators records for
    ml_dodd/one_correct. The test checks that catR's is infinite exactly there, which is all it checks for the
    all_correct examinee under ML, whose pattern stays constant.
    """
    standard_errors = np.array([estimate.se for estimate in _whole_test_session(case).estimates[1:]])
    expected = _floats(case["ses"])
    row = _WHOLE_TESTS["examinees"][case["bank"]]
    pattern = [row[case["examinee"]][item] for item in case["items"]]
    constant = np.array([len(set(pattern[: k + 1])) == 1 for k in range(len(pattern))])
    finite = np.isfinite(expected)

    assert (~finite).tolist() == (constant & (case["method"] == "ML")).tolist()
    assert standard_errors.shape == expected.shape
    if case["method"] == "EAP":
      np.testing.assert_allclose(standard_errors, expected, rtol=_CLOSED_FORM_RTOL, atol=0)
    else:
      tolerance = _ESTIMATE_ATOL * (np.abs(_floats(case["se_slopes"])) + 1)
      np.testing.assert_array_less(np.abs(standard_errors - expected)[finite], tolerance[finite])


# A difference between catsim's statistic and catR's fails when it exceeds this many standard errors of the
# difference, sqrt(se_catsim ** 2 + se_catR ** 2). Each simulation case makes 54 comparisons, four statistics and 50
# exposure rates, and over the four cases Bonferroni's bound puts the chance that two agreeing implementations fail any
# of them below 0.15% (216 times the normal distribution's two-sided tail beyond 4.5, 6.8e-6).
_SIMULATION_Z = 4.5
_SIMULATION_SEED = 20261125


# catsim's randomized selector for each of catR's rules in simulations.R: randomesque MFI, progressive and proportional.
_SIMULATION_SELECTORS: dict[str, Callable[[Case], BaseSelector]] = {
  "MFI": lambda case: RandomesqueSelector(case["randomesque"]),
  "progressive": lambda case: ProgressiveSelector(case["test_size"], acceleration=case["acceleration"]),
  "proportional": lambda case: ProportionalSelector(case["test_size"], acceleration=case["acceleration"]),
}


def _simulation_summary(sessions: Sequence[Session], true_thetas: FloatArray, n_items: int) -> dict[str, Any]:
  """Return the statistics simulations.R records for catR, with their standard errors over examinees, for the sessions.

  The bias and the mean test length come with the standard deviation over the square root of n, the RMSE with the
  delta-method standard error sd(squared errors) / (2 RMSE sqrt(n)) (Morris, White and Crowther, 2019), and catR's
  overlap rate, the sum of squared exposure rates over the mean test length, with the jackknife standard error over the
  examinees.
  """
  n = len(sessions)
  errors = np.array([session.estimates[-1].theta for session in sessions]) - true_thetas
  administered = np.zeros((n, n_items))
  for row, session in zip(administered, sessions, strict=True):
    row[list(session.administered_items)] = 1
  counts = administered.sum(axis=0)
  lengths = administered.sum(axis=1)
  rates = counts / n
  left_out = (counts - administered) / (n - 1)
  overlaps = (left_out**2).sum(axis=1) / left_out.sum(axis=1)
  rmse = math.sqrt(np.mean(errors**2))
  return {
    "bias": float(np.mean(errors)),
    "bias_se": float(np.std(errors, ddof=1) / math.sqrt(n)),
    "rmse": rmse,
    "rmse_se": float(np.std(errors**2, ddof=1) / (2 * rmse * math.sqrt(n))),
    "mean_length": float(np.mean(lengths)),
    "mean_length_se": float(np.std(lengths, ddof=1) / math.sqrt(n)),
    "overlap": float(np.sum(rates**2) / np.sum(rates)),
    "overlap_se": math.sqrt((n - 1) / n * np.sum((overlaps - np.mean(overlaps)) ** 2)),
    "exposure_counts": counts,
  }


@functools.cache
def _catsim_simulation(case_id: str) -> dict[str, Any]:
  """Return the statistics of catsim's run of a simulations.R case, on its bank and true abilities."""
  case = next(case for case in _SIMULATIONS["cases"] if case["id"] == case_id)
  bank = ItemBank(np.array(_SIMULATIONS["bank"], dtype=float))
  true_thetas = np.array(_SIMULATIONS["true_thetas"], dtype=float)
  estimator = EAPEstimator(grid=_trapezoid_grid(**_SIMULATIONS["grid"]), log_prior=_log_prior(_SIMULATIONS["prior"]))
  selector = _SIMULATION_SELECTORS[case["criterion"]](case)
  engine = CatEngine(bank, FixedPointInitializer(0.0), selector, estimator, _stop_rule(case))
  result = simulate(engine, true_thetas=true_thetas, rng=_SIMULATION_SEED)
  return _simulation_summary(result.sessions, true_thetas, bank.n_items)


def _assert_agree(case: Case, statistic: str) -> None:
  """Assert that catsim's statistic differs from catR's by at most _SIMULATION_Z standard errors of the difference."""
  catsim = _catsim_simulation(case["id"])
  difference = catsim[statistic] - case[statistic]
  error = math.hypot(catsim[f"{statistic}_se"], case[f"{statistic}_se"])

  assert abs(difference) <= _SIMULATION_Z * error, (difference, error)


class TestSimulations:
  """catR's simulateRespondents with the randomized selection rules (level 3), against catsim's simulate.

  Both packages run the same true abilities, the quantiles of the standard normal distribution, on the same bank, with
  catR's start at ability 0, EAP estimation on its grid and the same stopping rules, each with its own random numbers:
  catsim's from a fixed seed, catR's as simulations.R describes. A statistic agrees when the two differ by at most
  _SIMULATION_Z standard errors of their difference.

  The standard errors are taken over the examinees, as if their abilities were drawn at random, so they count the
  spread of the statistics between abilities as noise. Both packages use the same fixed abilities, whose spread would
  cancel in a paired difference, so the bound is conservative: for two catsim runs of randomesque_precision on
  different seeds, the mean length's standard error of the difference is 0.131, against 0.094 for the paired
  differences of each examinee.
  """

  @pytest.mark.parametrize("case", _cases(_SIMULATIONS, "simulation_bias"))
  def test_bias(self, case: Case) -> None:
    """Final estimates pulled toward one end of the scale, as by a prior or start off catR's, would shift the bias."""
    _assert_agree(case, "bias")

  @pytest.mark.parametrize("case", _cases(_SIMULATIONS, "simulation_rmse"))
  def test_rmse(self, case: Case) -> None:
    """A selector that gave less informative items than catR's rule would leave the estimates further from the truth."""
    _assert_agree(case, "rmse")

  @pytest.mark.parametrize("case", _cases(_SIMULATIONS, "simulation_length"))
  def test_mean_length(self, case: Case) -> None:
    """A precision rule with another threshold, or tests that gained information at another pace, would end elsewhere.

    The fixed-length cases have no Monte Carlo error: every test has the planned length in both packages.
    """
    _assert_agree(case, "mean_length")

  @pytest.mark.parametrize("case", _cases(_SIMULATIONS, "simulation_overlap"))
  def test_overlap(self, case: Case) -> None:
    """A bin of another size, or a schedule weighting information more or less than catR's, would change the overlap."""
    _assert_agree(case, "overlap")

  @pytest.mark.parametrize("case", _cases(_SIMULATIONS, "simulation_exposure"))
  def test_exposure_rates(self, case: Case) -> None:
    """A selector that drew from other items than catR's rule, such as a bin at another estimate, would expose them.

    Each item's exposure rate is compared by the two-proportion z test, its standard error from the rate pooled over
    the two runs, which have the same number of examinees.
    """
    n = len(_SIMULATIONS["true_thetas"])
    catsim = _catsim_simulation(case["id"])["exposure_counts"]
    catr = np.array(case["exposure_counts"], dtype=float)
    pooled = (catsim + catr) / (2 * n)
    error = np.sqrt(pooled * (1 - pooled) * 2 / n)

    assert np.all(np.abs(catsim - catr) / n <= _SIMULATION_Z * error)
