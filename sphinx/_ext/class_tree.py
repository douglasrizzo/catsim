"""Add a ``:direction:`` option to sphinxcontrib-mermaid's ``autoclasstree`` directive.

``autoclasstree`` always draws a class tree from top to bottom, so a base class with many
direct subclasses spreads them across one row that is too wide for the page. The option adds
Mermaid's ``direction`` statement to the generated class diagram.
"""

from typing import ClassVar

from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util.typing import OptionSpec
from sphinxcontrib.mermaid import MermaidClassDiagram

DIRECTIONS = ("tb", "bt", "lr", "rl")


def _direction(argument: str) -> str:
  return directives.choice(argument, DIRECTIONS).upper()


class ClassTree(MermaidClassDiagram):
  """``autoclasstree`` that also takes ``:direction:`` (``TB``, ``BT``, ``LR`` or ``RL``)."""

  option_spec: ClassVar[OptionSpec] = {**MermaidClassDiagram.option_spec, "direction": _direction}

  def get_mm_code(self) -> str:
    """Return the generated class diagram, with the requested direction if any."""
    code = super().get_mm_code()
    direction = self.options.get("direction")
    if not code or direction is None:
      return code
    # Mermaid accepts a class diagram's direction statement anywhere in its body.
    return f"{code}\n  direction {direction}"


def setup(app: Sphinx) -> dict[str, bool]:
  """Register ``autoclasstree`` in place of sphinxcontrib-mermaid's version."""
  app.setup_extension("sphinxcontrib.mermaid")
  app.add_directive("autoclasstree", ClassTree, override=True)
  return {"parallel_read_safe": True, "parallel_write_safe": True}
