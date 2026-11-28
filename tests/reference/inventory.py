"""Write catsim's capability inventory: every public name of its public import paths, with its signature.

``uv run python -m tests.reference.inventory > inventory.md`` writes it as Markdown: for every
package's ``__all__`` and every public module's own definitions, each class with its constructor,
bases and public members, each function with its signature, and each constant and type alias with
its value; then the public names of internal modules that no public path exports. Run on two
commits, the two outputs diff into what was added, removed or changed.
"""

from __future__ import annotations

import ast
import dataclasses
import enum
import importlib
import inspect
import pkgutil
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import catsim

PACKAGES = (
  "catsim",
  "catsim.selection",
  "catsim.estimation",
  "catsim.initialization",
  "catsim.stopping",
  "catsim.exposure",
)
MODULES = ("catsim.metrics", "catsim.plot")


# Spellings that differ only in how a module imports a name, rewritten to one form so that two
# inventories diff only where a signature changed.
_SPELLINGS = (
  (re.compile(r"NDArray\[numpy\.floating\[typing\.Any\]\]"), "FloatArray"),
  (re.compile(r"NDArray\[numpy\.integer\[typing\.Any\]\]"), "IntArray"),
  (re.compile(r"\bnumpy\.random\._generator\.Generator\b"), "numpy.random.Generator"),
  (re.compile(r"\bmatplotlib\.axes\._axes\.Axes\b"), "Axes"),
  (re.compile(r"\bcollections\.abc\.|\bnumpy\.typing\.|\bnpt\."), ""),
  (re.compile(r"\bcatsim(?:\.\w+)*\.(?=[A-Z])"), ""),
)


def _annotation(annotation: object) -> str:
  """Return an annotation as source text, without the quotes of a postponed one and in one spelling."""
  text = annotation if isinstance(annotation, str) else inspect.formatannotation(annotation)
  for pattern, replacement in _SPELLINGS:
    text = pattern.sub(replacement, text)
  return text


def signature(obj: Callable[..., object], *, drop_first: bool = False) -> str:
  """Return a callable's signature as source text, ``(params) -> return``.

  Parameters
  ----------
  obj : Callable
      A function, method or class.
  drop_first : bool, optional
      Whether to leave out the first parameter, ``self`` or ``cls``. Default is False.

  Returns
  -------
  str
      The signature, or ``(...)`` when Python cannot tell it.
  """
  try:
    sig = inspect.signature(obj)
  except (TypeError, ValueError):
    return "(...)"
  params = list(sig.parameters.values())[1:] if drop_first else list(sig.parameters.values())
  parts = []
  for param in params:
    text = param.name
    if param.kind is param.VAR_POSITIONAL:
      text = f"*{text}"
    elif param.kind is param.VAR_KEYWORD:
      text = f"**{text}"
    if param.annotation is not param.empty:
      text += f": {_annotation(param.annotation)}"
    if param.default is not param.empty:
      text += f" = {_annotation(repr(param.default))}"
    parts.append(text)
  if any(param.kind is param.KEYWORD_ONLY for param in params) and not any(
    param.kind is param.VAR_POSITIONAL for param in params
  ):
    first = next(index for index, param in enumerate(params) if param.kind is param.KEYWORD_ONLY)
    parts.insert(first, "*")
  returns = "" if sig.return_annotation is sig.empty else f" -> {_annotation(sig.return_annotation)}"
  return f"({', '.join(parts)}){returns}"


def _members(cls: type) -> list[str]:
  """Return one line per public member a class defines itself: fields, members, properties and methods."""
  lines = []
  fields = {field.name: field for field in dataclasses.fields(cls)} if dataclasses.is_dataclass(cls) else {}
  for name, field in fields.items():
    if not name.startswith("_") and name in cls.__dict__.get("__dataclass_fields__", {}):
      lines.append(f"field `{name}: {_annotation(field.type)}`")
  for name, value in vars(cls).items():
    if name.startswith("_") or name in fields:
      continue
    if isinstance(cls, enum.EnumMeta) and isinstance(value, cls):
      lines.append(f"member `{name} = {value.value!r}`")
    elif isinstance(value, property):
      returns = inspect.signature(value.fget).return_annotation if value.fget else inspect.Signature.empty
      suffix = "" if returns is inspect.Signature.empty else f" -> {_annotation(returns)}"
      lines.append(f"property `{name}{suffix}`")
    elif isinstance(value, staticmethod):
      lines.append(f"staticmethod `{name}{signature(value.__func__)}`")
    elif isinstance(value, classmethod):
      lines.append(f"classmethod `{name}{signature(value.__func__, drop_first=True)}`")
    elif inspect.isfunction(value):
      lines.append(f"method `{name}{signature(value, drop_first=True)}`")
    else:
      lines.append(f"attribute `{name} = {value!r}`")
  return lines


def describe(name: str, value: object) -> list[str]:
  """Return the inventory lines of one public name: a heading line, then one line per member of a class."""
  if inspect.isclass(value):
    bases = ", ".join(base.__name__ for base in value.__bases__ if base is not object)
    kind = "enum" if isinstance(value, enum.EnumMeta) else "class"
    constructor = "" if isinstance(value, enum.EnumMeta) else signature(value)
    head = f"- {kind} `{name}{constructor}`" + (f", bases {bases}" if bases else "")
    return [head, *(f"  - {line}" for line in _members(value))]
  if inspect.isfunction(value):
    return [f"- function `{name}{signature(value)}`"]
  if isinstance(value, ModuleType):
    return [f"- module `{name}`"]
  return [f"- value `{name} = {value!r}`"]


def _module_definitions(module: ModuleType) -> list[str]:
  """Return the public names a module's own source defines, in source order: classes, functions and assignments."""
  names: list[str] = []
  for node in ast.parse(Path(str(module.__file__)).read_text(encoding="utf-8")).body:
    if isinstance(node, ast.ClassDef | ast.FunctionDef):
      names.append(node.name)
    elif isinstance(node, ast.Assign):
      names.extend(target.id for target in node.targets if isinstance(target, ast.Name))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      names.append(node.target.id)
  return [name for name in dict.fromkeys(names) if not name.startswith("_")]


def _public_paths() -> dict[str, list[tuple[str, object]]]:
  """Return each public import path's names, with their objects."""
  paths: dict[str, list[tuple[str, object]]] = {}
  for path in PACKAGES:
    package = importlib.import_module(path)
    paths[path] = [(name, getattr(package, name)) for name in package.__all__]
  for path in MODULES:
    module = importlib.import_module(path)
    paths[path] = [(name, getattr(module, name)) for name in _module_definitions(module)]
  return paths


def _unexported(paths: dict[str, list[tuple[str, object]]]) -> list[tuple[str, object]]:
  """Return the public definitions of internal modules that no public path exports."""
  exported = {id(value) for names in paths.values() for _, value in names}
  found = []
  for info in pkgutil.walk_packages(catsim.__path__, prefix="catsim."):
    if info.name in PACKAGES or info.name in MODULES or info.ispkg:
      continue
    module = importlib.import_module(info.name)
    found += [
      (f"{info.name}.{name}", getattr(module, name))
      for name in _module_definitions(module)
      if id(getattr(module, name)) not in exported
    ]
  return found


def inventory() -> str:
  """Return the inventory as Markdown, with its entry count: one per public name and one per class member."""
  paths = _public_paths()
  sections = [
    (path, [line for name, value in names for line in describe(f"{path}.{name}", value)])
    for path, names in paths.items()
  ]
  unexported = [line for name, value in _unexported(paths) for line in describe(name, value)]
  sections.append(("Public definitions of internal modules that no public path exports", unexported))
  count = sum(len(lines) for _, lines in sections)
  body = "\n\n".join(f"## {title}\n\n" + ("\n".join(lines) if lines else "None.") for title, lines in sections)
  return (
    f"# catsim's capability inventory\n\ncatsim {catsim.__version__}, {count} entries: one per public name of a public "
    "import path, one per public member of a class, and one per public definition of an internal module that no "
    f"public path exports. Written by `tests/reference/inventory.py`.\n\n{body}\n"
  )


if __name__ == "__main__":
  sys.stdout.write(inventory())
