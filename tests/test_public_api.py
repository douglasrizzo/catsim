"""Tests for catsim's public import paths and the names its public signatures use (ADR-0023)."""

import ast
import dataclasses
import importlib
import importlib.util
import inspect
import pkgutil
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

import catsim

TESTS = Path(__file__).parent
PACKAGES = [
  "catsim",
  "catsim.selection",
  "catsim.estimation",
  "catsim.initialization",
  "catsim.stopping",
  "catsim.exposure",
]
# Top-level modules that are import paths of their own; catsim's other modules are internal.
PUBLIC_MODULES = ["catsim.metrics", "catsim.plot"]
# Names that ADR-0023's list retired, each for the one its concept keeps; item_id is one of its rejected alternatives.
# The rescoping retired the last four: a step result's selected_item_id (item), the metrics' actual and predicted
# (true_thetas and thetas) and the engine's state (session).
RETIRED_NAMES = {
  "administered_item_ids",
  "est_theta",
  "item_id",
  "item_indices",
  "item_type",
  "itemtype",
  "response_vector",
  "response_vectors",
  "seed",
  "updated_theta",
  "selected_item_id",
  "actual",
  "predicted",
  "state",
}


# The mutable runtime that ADR-0006 replaced with immutable sessions and an engine of three steps, and the runner and
# response providers that simulate replaced (ADR-0014).
RETIRED_EXPORTS = {
  "CatSessionSnapshot",
  "CatSessionState",
  "CatStepResult",
  "ExposureTracker",
  "ResponseProvider",
  "RunContext",
  "SimulatedResponseProvider",
  "SimulationRunner",
}
# The item response theory that catsim takes from irtlib (ADR-0024): no catsim package exports these names, and no
# catsim module of these names exists.
IRTLIB_NAMES = {"IrtModel", "ItemBank", "irt", "typing"}
IRTLIB_MODULES = {"catsim.irt", "catsim.item_bank", "catsim.typing"}
# The members that the classes catsim keeps lost: the engine's methods that changed a session in place (ADR-0006); the
# session's identifier and true ability, which the caller keeps beside it (ADR-0006), and its flags and counts, which
# stop_reason and its fields already give (ADR-0010); and the result's lists and aggregates of the sessions, which its
# tables and its metric methods replace (ADR-0016).
RETIRED_MEMBERS = {
  catsim.CatEngine: {"apply_response", "run_session", "select_next", "should_stop", "start_session", "step"},
  catsim.Session: {"administered_count", "is_stopped", "session_id", "snapshot", "true_theta"},
  catsim.SimulationResult: {
    "administered_items",
    "evaluations",
    "final_ses",
    "final_thetas",
    "mean_evaluations",
    "responses",
    "test_lengths",
    "theta_histories",
  },
}


def _internal_modules(package: ModuleType) -> Iterator[ModuleType]:
  """Yield the modules of ``package`` that are not import paths of their own."""
  for info in pkgutil.iter_modules(package.__path__):
    name = f"{package.__name__}.{info.name}"
    if not info.ispkg and name not in PUBLIC_MODULES:
      yield importlib.import_module(name)


def _top_level_definitions(module: ModuleType) -> set[str]:
  """Return the public names a module's own source defines: classes, functions and assigned aliases."""
  names: set[str] = set()
  for node in ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8")).body:
    if isinstance(node, ast.ClassDef | ast.FunctionDef):
      names.add(node.name)
    elif isinstance(node, ast.Assign):
      names.update(target.id for target in node.targets if isinstance(target, ast.Name))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      names.add(node.target.id)
  return {name for name in names if not name.startswith("_")}


def _public_objects() -> Iterator[tuple[str, object]]:
  """Yield every name exported by a package's ``__all__`` or defined by a public module, with its object."""
  for name in PACKAGES:
    package = importlib.import_module(name)
    for attribute in package.__all__:
      yield f"{name}.{attribute}", getattr(package, attribute)
  for name in PUBLIC_MODULES:
    module = importlib.import_module(name)
    for attribute, value in vars(module).items():
      if not attribute.startswith("_") and getattr(value, "__module__", None) == name:
        yield f"{name}.{attribute}", value


def _defining_module(obj: object) -> str | None:
  """Return the module that defines a class or function, or None for any other object, such as a type alias."""
  return obj.__module__ if inspect.isclass(obj) or inspect.isfunction(obj) else None


def _catsim_classes(cls: type) -> list[type]:
  """Return the classes of ``cls``'s method resolution order that catsim defines."""
  return [klass for klass in cls.__mro__ if klass.__module__.startswith("catsim")]


def _callables(label: str, obj: object) -> Iterator[tuple[str, Callable[..., object]]]:
  """Yield a public function, or the constructor and public methods that catsim defines on a class."""
  if not inspect.isclass(obj):
    if inspect.isfunction(obj):
      yield label, obj
    return
  for klass in _catsim_classes(obj):
    for name, value in vars(klass).items():
      if (name.startswith("_") and name != "__init__") or isinstance(value, property):
        continue
      method = getattr(obj, name)
      if inspect.isfunction(method) or inspect.ismethod(method):
        yield f"{label}.{name}", method


def _attributes(cls: type) -> set[str]:
  """Return the public attribute names of a class: its fields, properties and methods."""
  names = {field.name for field in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else set()
  for klass in _catsim_classes(cls):
    names.update(vars(klass))
    names.update(inspect.get_annotations(klass))
  return {name for name in names if not name.startswith("_")}


@pytest.mark.parametrize("package_name", PACKAGES)
def test_every_exported_name_is_defined_inside_its_package(package_name: str) -> None:
  """An ``__all__`` entry that names nothing breaks ``import *``; one from another package gives a second path.

  A class or function exported by a subpackage must come from that subpackage's own modules, and one exported by
  ``catsim`` from catsim, so that each object has one public import path.
  """
  package = importlib.import_module(package_name)

  missing = {name for name in package.__all__ if not hasattr(package, name)}
  foreign = {
    name
    for name in set(package.__all__) - missing
    if (module := _defining_module(getattr(package, name))) is not None and not module.startswith(f"{package_name}.")
  }

  assert (missing, foreign) == (set(), set())


@pytest.mark.parametrize("package_name", PACKAGES)
def test_every_public_definition_of_an_internal_module_is_exported(package_name: str) -> None:
  """A public class, function or alias missing from its package's ``__all__`` is reachable only through a module file.

  LogPrior annotated the Bayesian estimators' signatures while only ``catsim.estimation.bayesian`` exported it, and the
  selector helpers that custom selectors use were importable only from ``catsim.selection.base``.
  """
  package = importlib.import_module(package_name)

  unexported = {
    f"{module.__name__}.{name}"
    for module in _internal_modules(package)
    for name in _top_level_definitions(module) - set(package.__all__)
  }

  assert unexported == set()


def test_the_tests_import_each_public_name_from_its_public_path() -> None:
  """A test that imports MaxInfoSelector from ``catsim.selection.information`` breaks when that module is renamed.

  ADR-0023 lets catsim split, merge or rename its internal modules, so code outside catsim imports each name from its
  public path. A private name, which only its own tests reach, may come from its module.
  """
  public_paths = {*PACKAGES, *PUBLIC_MODULES}
  offenders = {
    f"{path.relative_to(TESTS)}: from {node.module} import {alias.name}"
    for path in TESTS.rglob("*.py")
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
    if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("catsim.")
    for alias in node.names
    if node.module not in public_paths and not alias.name.startswith("_")
  }

  assert offenders == set()


def test_no_public_signature_takes_a_retired_name() -> None:
  """A public function, constructor or method with a parameter of a retired name, such as est_theta or seed, fails here.

  ADR-0023 gives each concept one name; ``posterior`` took ``response_vector`` and ``ItemBank.test_information`` took
  ``item_indices`` where the rest of the API says ``responses`` and ``indices``. Only parameter names are checked, not
  the functions' own names.
  """
  offenders = {
    f"{qualified}({parameter})"
    for label, obj in _public_objects()
    for qualified, function in _callables(label, obj)
    for parameter in inspect.signature(function).parameters
    if parameter in RETIRED_NAMES
  }

  assert offenders == set()


def test_a_parameter_named_items_is_an_item_matrix() -> None:
  """A public parameter named ``items`` that takes item indices, as ``LinearSelector(items)`` did, fails here.

  ADR-0023 gives ``items`` to an item parameter matrix, one row per item, and ``indices`` to a list of item indices, so
  an ``items`` annotation never holds integers.
  """
  offenders = {
    f"{qualified}(items: {parameter.annotation})"
    for label, obj in _public_objects()
    for qualified, function in _callables(label, obj)
    if (parameter := inspect.signature(function).parameters.get("items")) is not None
    and re.search(r"\bint\b|IntArray", str(parameter.annotation))
  }

  assert offenders == set()


def test_no_public_attribute_uses_a_retired_name() -> None:
  """A session, step result or simulation result attribute such as administered_item_ids or updated_theta fails here.

  Every exported class is checked, so a field the result types gain later is held to the same table.
  """
  offenders = {
    f"{label}.{name}"
    for label, obj in _public_objects()
    if inspect.isclass(obj)
    for name in _attributes(obj) & RETIRED_NAMES
  }

  assert offenders == set()


def test_the_mutable_runtime_is_gone() -> None:
  """A retired session type, run context, exposure tracker, runner or response provider, or an engine method that
  changed a session in place, left reachable gives a second way to run a test, with the state that outlived a run
  (ADR-0006) or the responses drawn as items are given (ADR-0014). A session that held its true ability again, or a
  result's final_thetas beside its examinees table, is a second record of one fact.
  """
  exported = RETIRED_EXPORTS & {name for name in dir(catsim) if not name.startswith("_")}
  members = {f"{cls.__name__}.{name}" for cls, names in RETIRED_MEMBERS.items() for name in names if hasattr(cls, name)}

  assert (exported, members) == (set(), set())


def test_names_that_irtlib_holds_are_not_exported_by_catsim() -> None:
  """A re-export such as ``catsim.ItemBank``, or a module such as ``catsim.irt``, gives an irtlib object a second import
  path, which ADR-0024 and ADR-0023 rule out."""
  exported = {
    f"{package}.{name}"
    for package in PACKAGES
    for name in IRTLIB_NAMES
    if hasattr(importlib.import_module(package), name)
  }
  modules = {name for name in IRTLIB_MODULES if importlib.util.find_spec(name) is not None}

  assert (exported, modules) == (set(), set())
