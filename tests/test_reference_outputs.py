"""Tests that catsim computes what it computed at adda72dc, the baseline of the rescoping.

Each case of ``tests/reference/cases.py`` is recomputed with the current code and compared with its
output recorded in ``tests/reference/data/``: exactly for integers, Booleans and strings, and to
1e-12 for floats, relative to their magnitude or their unit; the outputs of an estimator's search
compare to its tolerance away from the environment that recorded them (see
``tests/reference/records.py``). The recorded data changes only where an ADR changes a result; see
``tests/reference/__init__.py``.

The generator, not a test, checks that every public strategy and computation of adda72dc has a
case: public API added later is not part of the baseline.
"""

from __future__ import annotations

import functools
import math
import re

import pytest

from tests.reference import cases, records
from tests.reference.generate import environment, family_text, stale_files

CASES = [(family, name) for family, family_cases in cases.FAMILIES.items() for name in family_cases]


@functools.cache
def _recorded(family: str) -> dict[str, object]:
  """Return a family's recorded outputs, by case name."""
  return records.load(records.family_file(family))["cases"]


@functools.cache
def _in_recording_environment() -> bool:
  """Return whether this platform and these NumPy and SciPy versions are those that recorded the data."""
  return records.load(records.PROVENANCE_FILE)["environment"] == environment()


@functools.cache
def _computed(family: str, name: str) -> object:
  """Return a case's output as the current code computes it, computed once per test run."""
  return records.normalize(cases.FAMILIES[family][name]())


@pytest.mark.parametrize(("family", "name"), CASES, ids=[f"{family}:{name}" for family, name in CASES])
def test_case_matches_its_recorded_output(family: str, name: str) -> None:
  """A computation that changed since adda72dc fails its case.

  A different probability, estimate, standard error, chosen item, stopping decision or exposure
  count, a lost output or a changed type differs from the recorded data.
  """
  recorded = _recorded(family)
  assert name in recorded, f"{family} has no recorded output for {name!r}; see tests/reference/__init__.py"
  how = cases.comparison(family, name)
  absolute = records.SEARCH_TOLERANCE if how.searched and not _in_recording_environment() else 0.0
  found = records.differences(recorded[name], _computed(family, name), unit=how.unit, absolute=absolute)
  assert not found, f"{len(found)} differences, the first ones:\n" + "\n".join(found[:20])


@pytest.mark.parametrize("family", list(cases.FAMILIES))
def test_data_file_holds_exactly_the_cases(family: str) -> None:
  """A recorded output whose case was deleted from cases.py would no longer be checked by anything."""
  assert sorted(_recorded(family)) == sorted(cases.FAMILIES[family])


def test_data_directory_holds_one_file_per_family() -> None:
  """A family without a data file, or a data file left behind by a deleted family, would go unchecked."""
  assert all(records.family_file(family).is_file() for family in cases.FAMILIES)
  assert stale_files() == []


def test_data_was_recorded_from_committed_source() -> None:
  """Data recorded from source with uncommitted changes would not be the outputs of the commit it names."""
  provenance = records.load(records.PROVENANCE_FILE)
  assert re.fullmatch(r"[0-9a-f]{40}", provenance["catsim_commit"])
  assert provenance["catsim_source_modified"] is False


def test_regenerating_in_the_recorded_environment_reproduces_the_data() -> None:
  """Data edited by hand, or left stale after its cases changed, differs from what the generator writes.

  Only the platform and the NumPy and SciPy versions that recorded the data reproduce it bit for
  bit, so elsewhere the test is skipped and the tolerance of the case tests applies.
  """
  if not _in_recording_environment():
    recorded_environment = records.load(records.PROVENANCE_FILE)["environment"]
    pytest.skip(f"the data was recorded on {recorded_environment}, and this is {environment()}")
  for family, family_cases in cases.FAMILIES.items():
    expected = family_text({name: _computed(family, name) for name in family_cases})
    assert records.family_file(family).read_text(encoding="utf-8") == expected, family


@pytest.mark.parametrize(
  ("expected", "actual"),
  [
    (1.0, 1.0 + 4e-12),
    (-250.0, -250.0 * (1 + 2e-12)),
    (7.6e-22, 0.0),
    (7.6e-22, 7.6e-22 * (1 + 2e-12)),
    (-4e-16, 4e-16),
    (1, 2),
    (1, 1.0),
    (True, 1),
    (math.inf, -math.inf),
    (math.nan, 0.0),
    ("stopper", "no_items"),
    (None, 0.0),
    ([1, 2], [1, 2, 3]),
    ({"a": 1}, {"b": 1}),
    ([[0.5, 1.0]], [[0.5, 1.1]]),
  ],
)
def test_comparison_reports_a_difference(expected: object, actual: object) -> None:
  """A comparison that let a float past 1e-12, a tail value lost to 0, an int for a float or a missing entry through
  would pass every case."""
  assert records.differences(expected, actual)


@pytest.mark.parametrize(("expected", "actual"), [(0.0, 2e-12), (0.3, 0.3 + 2e-12), (2.5, 2.5 * (1 + 2e-12))])
def test_comparison_on_the_ability_scale_reports_a_difference(expected: float, actual: float) -> None:
  """A unit that loosened abilities past 1e-12 of the scale would let a changed estimate pass."""
  assert records.differences(expected, actual, unit=1.0)


@pytest.mark.parametrize(
  ("expected", "actual"),
  [
    (1.0, 1.0 + 5e-13),
    (-250.0, -250.0 * (1 + 5e-13)),
    (7.6e-22, 7.6e-22 * (1 + 5e-13)),
    (0.0, 0.0),
    (math.inf, math.inf),
    (math.nan, math.nan),
    ([1, True, "a", None, {"x": 0.25}], [1, True, "a", None, {"x": 0.25}]),
  ],
)
def test_comparison_accepts_rounding(expected: object, actual: object) -> None:
  """A comparison stricter than 1e-12, or that told NaN from NaN, would fail on the last bits of another platform."""
  assert records.differences(expected, actual) == []


@pytest.mark.parametrize(("expected", "actual"), [(2.0e-17, -8.4e-19), (0.3, 0.3 + 5e-13), (-4.5e-16, 3e-13)])
def test_comparison_on_the_ability_scale_accepts_rounding_near_zero(expected: float, actual: float) -> None:
  """A comparison that judged an estimate of 0 relatively would fail on the rounding noise a symmetric prior leaves."""
  assert records.differences(expected, actual, unit=1.0) == []


def test_data_reads_back_the_floats_it_was_written_from() -> None:
  """A data file that rounded floats, or wrote infinities and NaN it cannot read, would record other values."""
  values = [0.1, 1 / 3, -2.0e-300, 5e-324, 1.7976931348623157e308, math.inf, -math.inf]
  text = records.dumps({"cases": {"values": values}})
  read = records.load_text(text)["cases"]["values"]
  assert read == values
  assert math.isnan(records.load_text(records.dumps([math.nan]))[0])


@pytest.mark.parametrize(("expected", "actual"), [(0.5, 0.5 + 9e-6), (-2.0, -2.0 - 9e-6)])
def test_comparison_of_a_search_accepts_its_tolerance(expected: float, actual: float) -> None:
  """A comparison that held a search's estimate to 1e-12 off the recording platform would fail on Linux and Windows."""
  assert records.differences(expected, actual, unit=1.0, absolute=records.SEARCH_TOLERANCE) == []


@pytest.mark.parametrize(("expected", "actual"), [(0.5, 0.5 + 2e-5), (2, 1)])
def test_comparison_of_a_search_reports_more_than_its_tolerance(expected: float, actual: float) -> None:
  """A search tolerance that let a changed estimate or a changed item through would hide a lost computation."""
  assert records.differences(expected, actual, unit=1.0, absolute=records.SEARCH_TOLERANCE)


def test_searches_are_told_apart_from_closed_forms() -> None:
  """Marking EAP or a closed form as a search would loosen it elsewhere; missing a search would fail elsewhere."""
  assert cases.comparison("estimation", "MAPEstimator()|4pl") == cases.Comparison(unit=1.0, searched=True)
  assert cases.comparison("estimation", "EAPEstimator()|4pl") == cases.Comparison(unit=1.0, searched=False)
  assert cases.comparison("whole_tests", "KL|WLE|MaxLength(10)").searched
  assert not cases.comparison("whole_tests", "MEI(EAP)|EAP|MaxLength(6)").searched
  assert cases.comparison("selection", "ClusterSelector.sum_cluster_infos|theta=0.0") == cases.Comparison(
    unit=0.0, searched=False
  )
  assert cases.comparison("metrics", "rmse|five").unit == 1.0
