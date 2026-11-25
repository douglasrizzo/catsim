"""Record the reference outputs of the current code: ``uv run python -m tests.reference.generate``.

The generator computes every case of :mod:`tests.reference.cases` and writes one data file per
family to ``tests/reference/data/``, and ``provenance.json`` beside them: the commit of the catsim
source it ran on, whether that source had uncommitted changes, and the versions of catsim, Python,
NumPy and SciPy and the platform, and removes any data file of a family that no longer exists.
With ``--check`` it writes nothing and reports the family files that the current code would change,
comparing them byte for byte, and the stale ones. It never compares ``provenance.json``, which
changes with every commit.

It refuses to record while a public strategy class or computational function has no case, or
while a whole test gives different outputs for two seeds, which would mean that its configuration
draws random numbers and that ADR-0013's random model could change it.

The data records catsim at adda72dc, and changes only where an ADR changes a result: the pull
request that regenerates it names the ADR and the outputs that change.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import scipy

import catsim
from tests.reference import cases, records

ABOUT = (
  "Reference outputs of catsim, recorded by tests/reference/generate.py from the source that "
  "provenance.json names. They change only where an ADR changes a result, and the pull request that "
  "changes them names the ADR and the outputs. Floats compare to 1e-12 of their magnitude or unit, and the "
  "estimates of a numerical search, away from the recording environment, to its tolerance; everything else "
  "exactly (see tests/reference/records.py)."
)
OTHER_SEED = 20261125


def family_text(outputs: dict[str, object]) -> str:
  """Return the text of a family's data file, given its outputs by case name."""
  return records.dumps({"about": ABOUT, "cases": outputs})


def compute(family: str) -> dict[str, object]:
  """Return every output of a family, by case name, as JSON's native types."""
  return {name: records.normalize(case()) for name, case in cases.FAMILIES[family].items()}


def environment() -> dict[str, str]:
  """Return what the outputs' last bits may depend on: the platform and the NumPy and SciPy versions."""
  return {
    "machine": platform.machine(),
    "numpy": numpy.__version__,
    "scipy": scipy.__version__,
    "system": platform.system(),
  }


def _git(*args: str) -> str:
  """Return the output of a git command run in the repository that holds the imported catsim."""
  source = Path(catsim.__file__).resolve().parent
  return subprocess.run(["git", *args], cwd=source, capture_output=True, text=True, check=True).stdout.strip()


def provenance() -> dict[str, object]:
  """Return where and how the outputs were computed."""
  return {
    "catsim_commit": _git("rev-parse", "HEAD"),
    "catsim_source_modified": bool(_git("status", "--porcelain", "--", ".")),
    "catsim_version": catsim.__version__,
    "environment": environment(),
    "python": platform.python_version(),
  }


def seed_dependent_whole_tests() -> list[str]:
  """Return the whole tests whose outputs change with the run's seed."""
  return [
    name
    for name in cases.WHOLE_TESTS
    if records.differences(
      records.normalize(cases.whole_test(name)), records.normalize(cases.whole_test(name, seed=OTHER_SEED))
    )
  ]


def stale_files() -> list[Path]:
  """Return the data files that belong to no family of the cases."""
  expected = {records.PROVENANCE_FILE, *(records.family_file(family) for family in cases.FAMILIES)}
  return sorted(path for path in records.DATA_DIR.glob("*.json") if path not in expected)


def _report(heading: str, lines: list[str]) -> None:
  """Write a heading and indented lines to standard error."""
  sys.stderr.write("".join([heading, "\n", *(f"  {line}\n" for line in lines)]))


def main(argv: list[str] | None = None) -> int:
  """Record the reference outputs, or with ``--check`` report which files the current code would change.

  Parameters
  ----------
  argv : list[str] or None, optional
      Command-line arguments. Default is None, which reads ``sys.argv``.

  Returns
  -------
  int
      0 on success; 1 if the cases miss a public name, a whole test depends on its seed, or, with
      ``--check``, a family file would change or is stale.
  """
  parser = argparse.ArgumentParser(
    prog="python -m tests.reference.generate", description="Record the reference outputs of the current code."
  )
  parser.add_argument("--check", action="store_true", help="compare with the data files instead of writing them")
  args = parser.parse_args(argv)

  if missing := cases.uncovered():
    _report("No reference case covers:", missing)
    return 1
  if seed_dependent := seed_dependent_whole_tests():
    _report("These whole tests depend on the seed, so they draw random numbers:", seed_dependent)
    return 1

  records.DATA_DIR.mkdir(exist_ok=True)
  changed = [path.name for path in stale_files()]
  for family in cases.FAMILIES:
    path = records.family_file(family)
    text = family_text(compute(family))
    if args.check:
      if not path.exists() or path.read_text(encoding="utf-8") != text:
        changed.append(path.name)
    else:
      path.write_text(text, encoding="utf-8")
  if args.check:
    if changed:
      _report("The current code would change or remove:", changed)
      return 1
    _report("The data files match the current code byte for byte.", [])
    return 0
  for path in stale_files():
    path.unlink()
  records.PROVENANCE_FILE.write_text(records.dumps(provenance()), encoding="utf-8")
  _report(f"Wrote {len(cases.FAMILIES)} data files and provenance.json to {records.DATA_DIR}.", [])
  return 0


if __name__ == "__main__":
  sys.exit(main())
