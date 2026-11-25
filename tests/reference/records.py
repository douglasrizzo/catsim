"""Write, read and compare the reference data files.

A data file holds one family of cases as JSON, in the layout of the ``pretty-format-json``
pre-commit hook (two-space indent, sorted keys), so the hook leaves it as written. Floats are
written by ``repr``, which reads back as the same float; NaN and the infinities as ``NaN``,
``Infinity`` and ``-Infinity``, which Python's ``json`` reads back. JSON distinguishes integers
from floats, so the comparison knows which values must match exactly.

Floats agree when they differ by at most ``RELATIVE_TOLERANCE`` of the larger magnitude, or of the
quantity's ``unit`` when that is larger. Probabilities, information and densities have no unit:
their smallest values, such as a probability of 1e-22 far from an item's difficulty, are as exact
as their largest, so a value lost to 0 fails. Abilities, standard errors and the accuracy of
estimates have the ability scale's unit, 1: they can cancel to rounding noise, such as the
posterior mean of a symmetric prior, which is 0 up to about 1e-17.
"""

from __future__ import annotations

import enum
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy

DATA_DIR = Path(__file__).parent / "data"
PROVENANCE_FILE = DATA_DIR / "provenance.json"

RELATIVE_TOLERANCE = 1e-12


def normalize(value: object) -> Any:
  """Return ``value`` as JSON's native types: dict, list, str, int, float, bool and None.

  NumPy arrays and scalars become Python lists and scalars, tuples become lists, enum members
  become their values, and a mapping with keys other than strings becomes a list of
  ``[key, value]`` pairs in key order, since JSON keys are strings.

  Raises
  ------
  TypeError
      If ``value`` holds anything else.
  """
  if isinstance(value, enum.Enum):
    return normalize(value.value)
  if isinstance(value, numpy.ndarray | numpy.generic):
    return normalize(value.tolist())
  if value is None or isinstance(value, bool | int | float | str):
    return value
  if isinstance(value, Mapping):
    if all(isinstance(key, str) for key in value):
      return {key: normalize(item) for key, item in value.items()}
    return [[normalize(key), normalize(item)] for key, item in sorted(value.items())]
  if isinstance(value, Sequence):
    return [normalize(item) for item in value]
  msg = f"cannot record a {type(value).__name__} as reference data: {value!r}"
  raise TypeError(msg)


def dumps(data: object) -> str:
  """Return ``data`` as the text of a data file, in the pre-commit hook's layout."""
  return json.dumps(normalize(data), indent=2, sort_keys=True, allow_nan=True) + "\n"


def family_file(family: str) -> Path:
  """Return the path of the data file of a family of cases."""
  return DATA_DIR / f"{family}.json"


def load_text(text: str) -> Any:
  """Return the data that the text of a data file holds."""
  return json.loads(text)


def load(path: Path) -> Any:
  """Return the contents of a data file."""
  return load_text(path.read_text(encoding="utf-8"))


def _floats_agree(expected: float, actual: float, unit: float) -> bool:
  """Return whether two floats agree within the tolerance; non-finite values must be identical."""
  if not (math.isfinite(expected) and math.isfinite(actual)):
    return (math.isnan(expected) and math.isnan(actual)) or expected == actual
  return abs(expected - actual) <= RELATIVE_TOLERANCE * max(abs(expected), abs(actual), unit)


def _container_differences(expected: object, actual: object, path: str, unit: float) -> list[str] | None:
  """Return the differences between two dicts or two lists of the same type, or None for other values."""
  where = path or "<root>"
  if isinstance(expected, dict) and isinstance(actual, dict):
    if expected.keys() != actual.keys():
      missing, extra = sorted(expected.keys() - actual.keys()), sorted(actual.keys() - expected.keys())
      return [f"{where}: keys differ, missing {missing}, unexpected {extra}"]
    return [line for key in expected for line in differences(expected[key], actual[key], f"{path}.{key}", unit=unit)]
  if isinstance(expected, list) and isinstance(actual, list):
    if len(expected) != len(actual):
      return [f"{where}: expected {len(expected)} entries, got {len(actual)}: {actual!r}"]
    pairs = enumerate(zip(expected, actual, strict=True))
    return [line for index, pair in pairs for line in differences(*pair, f"{path}[{index}]", unit=unit)]
  return None


def differences(expected: object, actual: object, path: str = "", *, unit: float = 0.0) -> list[str]:
  """Return where ``actual`` departs from ``expected``, one line per difference.

  Both are JSON's native types. Integers, Booleans, strings and None must be equal, and of the
  same type; floats must agree within :data:`RELATIVE_TOLERANCE` of the larger of their magnitudes
  and ``unit``; lists must have the same length and dicts the same keys. ``unit`` is 0 for
  quantities compared relatively and 1 for quantities on the ability scale.
  """
  where = path or "<root>"
  if isinstance(expected, float) and isinstance(actual, float):
    agree = _floats_agree(expected, actual, unit)
  elif type(expected) is not type(actual):
    return [f"{where}: expected {expected!r} ({type(expected).__name__}), got {actual!r} ({type(actual).__name__})"]
  elif (nested := _container_differences(expected, actual, path, unit)) is not None:
    return nested
  else:
    agree = expected == actual
  return [] if agree else [f"{where}: expected {expected!r}, got {actual!r}"]
