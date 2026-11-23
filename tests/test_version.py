import importlib
import importlib.metadata
import tomllib
from pathlib import Path

import pytest
from packaging.version import Version

import catsim


def test_version_matches_pyproject() -> None:
  """Fail when ``catsim.__version__`` differs from the version in ``pyproject.toml``.

  Tag 0.19.0 shipped ``__version__ = "0.18.0"``: the string was hard-coded in ``__init__.py`` and
  the release bumped only ``pyproject.toml``.
  """
  with (Path(__file__).parents[1] / "pyproject.toml").open("rb") as f:
    expected = tomllib.load(f)["project"]["version"]
  assert Version(catsim.__version__) == Version(expected)


def test_import_without_package_metadata_gives_a_placeholder_version(monkeypatch: pytest.MonkeyPatch) -> None:
  """Fail when importing catsim from a source tree that was never installed raises instead of importing.

  ``importlib.metadata`` finds no metadata for a ``PYTHONPATH=src`` import or a vendored copy.
  """

  def no_metadata(name: str) -> str:
    raise importlib.metadata.PackageNotFoundError(name)

  monkeypatch.setattr(importlib.metadata, "version", no_metadata)
  try:
    importlib.reload(catsim)
    assert catsim.__version__ == "0+unknown"
  finally:
    monkeypatch.undo()
    importlib.reload(catsim)
