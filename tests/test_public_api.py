"""Tests for catsim's public import paths and the names its public signatures use (ADR-0014)."""

import ast
import dataclasses
import importlib
import inspect
import pkgutil
from collections.abc import Callable, Iterator
from pathlib import Path
from types import ModuleType

import pytest

PACKAGES = [
  "catsim",
  "catsim.selection",
  "catsim.estimation",
  "catsim.initialization",
  "catsim.stopping",
  "catsim.exposure",
]
# Top-level modules that are import paths of their own; catsim's other modules are internal.
PUBLIC_MODULES = ["catsim.irt", "catsim.metrics", "catsim.plot", "catsim.typing"]
# Names that ADR-0014's table retired, each for the one its concept keeps.
RETIRED_NAMES = {
  "administered_item_ids",
  "est_theta",
  "item_indices",
  "item_type",
  "itemtype",
  "response_vector",
  "response_vectors",
  "seed",
  "updated_theta",
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


def test_no_public_signature_takes_a_retired_name() -> None:
  """A public function, constructor or method with a parameter of a retired name, such as est_theta or seed, fails here.

  ADR-0014 gives each concept one name; ``posterior`` took ``response_vector`` and ``ItemBank.test_information`` took
  ``item_indices`` where the rest of the API says ``responses`` and ``indices``. Only parameter names are checked, not
  the functions' own names: ``irt.validate_response_vector`` keeps its name.
  """
  offenders = {
    f"{qualified}({parameter})"
    for label, obj in _public_objects()
    for qualified, function in _callables(label, obj)
    for parameter in inspect.signature(function).parameters
    if parameter in RETIRED_NAMES
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
