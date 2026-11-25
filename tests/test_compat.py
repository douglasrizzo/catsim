"""Tests for catsim.compat, the one module that imports typing-extensions (ADR-0021)."""

import subprocess
import sys

import pytest

# Imports every catsim module with typing_extensions unimportable, as on an installation that lacks it.
IMPORT_EVERY_MODULE_WITHOUT_TYPING_EXTENSIONS = """
import importlib, pkgutil, sys
sys.modules["typing_extensions"] = None
import catsim
for module in pkgutil.walk_packages(catsim.__path__, "catsim."):
  importlib.import_module(module.name)
"""


@pytest.mark.skipif(sys.version_info < (3, 12), reason="catsim depends on typing-extensions below Python 3.12")
def test_every_module_imports_without_typing_extensions() -> None:
  """A module that imports typing_extensions on Python 3.12 or later fails there for users who do not have it.

  catsim requires typing-extensions only below 3.12, but the development tools install it on every version, so the
  rest of the suite would not notice; the subprocess makes it unimportable.
  """
  result = subprocess.run(
    [sys.executable, "-c", IMPORT_EVERY_MODULE_WITHOUT_TYPING_EXTENSIONS], capture_output=True, text=True, check=False
  )

  assert result.returncode == 0, result.stderr
