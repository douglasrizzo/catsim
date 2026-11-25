"""Parity tests with catR, catsim's reference implementation (ADR-0020), level 1: single computations.

The expected values are catR's outputs on fixed cases, which ``tests/parity/catr/generate.R`` wrote, with the R and catR
versions, into the JSON files beside it; CONTRIBUTING.md says how to regenerate them. The tests need no R. Each case
records catR's settings, and the tests build the catsim strategy that matches them. Item indices in the files are
0-based, as in catsim.

A case where catsim departs from catR on purpose, with no setting that reproduces catR, is a strict expected failure
with the reason (``_DEPARTURES``): it must fail its assertion, so a departure that closes, or a case that stops
departing, fails the run.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from catsim import ItemBank, StepContext, irt
from catsim.estimation import (
  BaseEstimator,
  EAPEstimator,
  LogPrior,
  MAPEstimator,
  NumericalSearchEstimator,
  QuadratureGrid,
  WarmLikelihoodEstimator,
  normal_log_prior,
  uniform_log_prior,
)
from catsim.selection import (
  BaseSelector,
  KLSelector,
  MaxInfoSelector,
  MEISelector,
  MLWISelector,
  ProgressiveSelector,
  ProportionalSelector,
  UrrySelector,
)
from catsim.selection.progressive import _position_weight  # noqa: PLC2701 - the selectors expose no weight
from catsim.stopping import BaseStopper, ConfidenceIntervalStopper, MaxLengthStopper, StandardErrorStopper, StopRule
from catsim.typing import FloatArray
from tests.banks import two_pl_bank
from tests.generators import DeterministicRNG

Case = dict[str, Any]

_DATA = Path(__file__).parent / "parity" / "catr"


def _load(name: str) -> dict[str, Any]:
  with (_DATA / f"{name}.json").open(encoding="utf-8") as file:
    return json.load(file)


_IRT, _ESTIMATION, _SELECTION, _STOPPING = (_load(name) for name in ("irt", "estimation", "selection", "stopping"))

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
}


# The data file whose cases each key of _DEPARTURES names.
_DEPARTURE_DATA = {
  "estimate": _ESTIMATION,
  "standard_error": _ESTIMATION,
  "selection": _SELECTION,
  "values": _SELECTION,
  "trapezoid_values": _SELECTION,
  "schedule": _SELECTION,
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
    versions = {(data["catr_version"], data["r_version"]) for data in (_IRT, _ESTIMATION, _SELECTION, _STOPPING)}

    assert len(versions) == 1

  def test_departures_name_recorded_cases(self) -> None:
    """A departure whose case was renamed, dropped or put under another test would mark nothing and hide the change."""
    for test, departures in _DEPARTURES.items():
      recorded = {case["id"] for case in _DEPARTURE_DATA[test]["cases"]}

      assert set(departures) <= recorded, test


class TestItemResponseFunctions:
  """catR's Pi and Ii, on 1PL to 4PL items, against catsim.irt."""

  @pytest.mark.parametrize("case", _cases(_IRT, "probability"))
  def test_probability(self, case: Case) -> None:
    """A wrong logistic, or a guessing or slipping parameter entering it the wrong way, would move icc off catR's Pi."""
    np.testing.assert_allclose(
      irt.icc(_floats(case["thetas"]), np.array(case["items"], dtype=float)),
      np.array(case["probability"], dtype=float),
      rtol=_CLOSED_FORM_RTOL,
    )

  @pytest.mark.parametrize("case", _cases(_IRT, "information"))
  def test_information(self, case: Case) -> None:
    """An information formula that drops c or d, or squares the wrong term, would move off catR's Ii."""
    np.testing.assert_allclose(
      irt.information(_floats(case["thetas"]), np.array(case["items"], dtype=float)),
      np.array(case["information"], dtype=float),
      rtol=_CLOSED_FORM_RTOL,
    )


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


def _selector(case: Case) -> BaseSelector:
  return _SELECTORS[case["id"].split("/")[0]]


class TestSelectors:
  """catR's nextItem with MFI, bOpt, MLWI, MEI and KL, and the criterion values behind it, against catsim's selectors.

  A case's ranking is the order in which catR picks every candidate when each pick is made unavailable for the next,
  its choice first; catsim's is the same with each pick excluded. The bank's parameters are continuous, and generate.R
  checks that no two candidates nearly tie. The kl_window cases rank by catR's KL() over catsim's window.
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
