"""Check the backlog cards and rewrite the index in ``backlog/README.md``.

Run ``uv run --no-project python backlog/_index.py``, as the pre-commit hook does. A card is a
Markdown file under ``backlog/`` that starts with frontmatter. Any other Markdown file there, except
the README and files whose name starts with ``_``, is listed as background. The script also checks
that every relative link in the folder resolves. It exits with status 1 when a check fails or when
it rewrote the index, so the hook fails until the change is staged.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

BACKLOG = Path(__file__).resolve().parent
START = "<!-- backlog-index:start -->"
END = "<!-- backlog-index:end -->"

# Code is stripped before links are read, so `table[i](x)` in a sketch is not taken for a link.
_FENCED_CODE = re.compile(r"^(`{3,}|~{3,}).*?^\1", flags=re.DOTALL | re.MULTILINE)
_INLINE_CODE = re.compile(r"`+[^`\n]*`+")
_LINK = re.compile(r"\]\(([^()\s]+)\)")


class Kind(StrEnum):
  """What a card asks for."""

  TECHNIQUE = "technique"
  FEATURE = "feature"
  FIX = "fix"
  DOCS = "docs"
  DECISION = "decision"
  RESEARCH = "research"
  MAINTENANCE = "maintenance"


class Status(StrEnum):
  """Whether a card can be picked up."""

  READY = "ready"
  NEEDS_RESEARCH = "needs-research"
  NEEDS_DECISION = "needs-decision"
  BLOCKED = "blocked"
  PARKED = "parked"


class Effort(StrEnum):
  """A card's size."""

  XS = "XS"
  S = "S"
  M = "M"
  L = "L"
  XL = "XL"


PRIORITIES = ("1", "2", "3", "4")
REQUIRED = ("title", "kind", "status", "priority", "effort")
LISTS = ("blocked_by", "raised_by")


@dataclass(frozen=True)
class Card:
  """One card's frontmatter, and its path relative to ``backlog/`` without the ``.md`` suffix."""

  path: str
  title: str
  kind: Kind
  status: Status
  priority: int
  effort: Effort
  blocked_by: tuple[str, ...]
  raised_by: tuple[str, ...]

  @property
  def area(self) -> str:
    """Return the folder the card is in, relative to ``backlog/``."""
    return self.path.rpartition("/")[0] or "."

  @property
  def link(self) -> str:
    """Return a Markdown link to the card, relative to the README."""
    return f"[{_cell(self.title)}]({self.path}.md)"


class CardError(ValueError):
  """A card whose frontmatter cannot be read, or a link that does not resolve."""


def _cell(text: str) -> str:
  """Return ``text`` safe for a Markdown table cell."""
  return text.replace("|", r"\|")


def _fields(text: str) -> dict[str, str] | None:
  """Return the frontmatter's fields as raw strings, or None when the text has no frontmatter."""
  lines = text.removeprefix("﻿").splitlines()
  if not lines or not lines[0].startswith("---"):
    return None
  if lines[0] != "---":
    msg = f"the first line is {lines[0]!r}; frontmatter opens with '---' alone"
    raise CardError(msg)
  try:
    end = lines.index("---", 1)
  except ValueError:
    msg = "the frontmatter has no closing '---' line"
    raise CardError(msg) from None
  fields: dict[str, str] = {}
  for number, line in enumerate(lines[1:end], start=2):
    if not line.strip():
      continue
    key, separator, value = line.partition(":")
    key = key.strip()
    if not separator:
      msg = f"frontmatter line {number}, {line!r}, is not 'key: value'"
      raise CardError(msg)
    if key in fields:
      msg = f"frontmatter line {number} repeats {key}"
      raise CardError(msg)
    fields[key] = value.strip()
  return fields


def _list(name: str, value: str) -> tuple[str, ...]:
  """Return the items of an inline list such as ``[a, b]``."""
  if not (value.startswith("[") and value.endswith("]")):
    msg = f"{name} is {value!r}; it must be an inline list such as [a, b]"
    raise CardError(msg)
  return tuple(item.strip() for item in value[1:-1].split(",") if item.strip())


def _choice[Choice: StrEnum](name: str, value: str, choices: type[Choice]) -> Choice:
  """Return ``value`` as a member of ``choices``."""
  try:
    return choices(value)
  except ValueError:
    allowed = ", ".join(member.value for member in choices)
    msg = f"{name} is {value!r}; it must be one of {allowed}"
    raise CardError(msg) from None


def card(path: str, fields: dict[str, str]) -> Card:
  """Return the card at ``path`` from its frontmatter fields, checking every value."""
  missing = [name for name in REQUIRED if name not in fields]
  if missing:
    msg = f"missing {', '.join(missing)}"
    raise CardError(msg)
  unknown = sorted(set(fields) - set(REQUIRED) - set(LISTS))
  if unknown:
    msg = f"unknown field {', '.join(unknown)}"
    raise CardError(msg)
  if not fields["title"]:
    msg = "title is empty"
    raise CardError(msg)
  if fields["priority"] not in PRIORITIES:
    msg = f"priority is {fields['priority']!r}; it must be one of {', '.join(PRIORITIES)}"
    raise CardError(msg)
  status = _choice("status", fields["status"], Status)
  blocked_by = _list("blocked_by", fields.get("blocked_by", "[]"))
  if path in blocked_by:
    msg = "blocked_by names the card itself"
    raise CardError(msg)
  if status is Status.READY and blocked_by:
    msg = f"status is ready, but blocked_by names {', '.join(blocked_by)}; make it blocked or empty the list"
    raise CardError(msg)
  return Card(
    path=path,
    title=fields["title"],
    kind=_choice("kind", fields["kind"], Kind),
    status=status,
    priority=int(fields["priority"]),
    effort=_choice("effort", fields["effort"], Effort),
    blocked_by=blocked_by,
    raised_by=_list("raised_by", fields.get("raised_by", "[]")),
  )


def broken_links(file: Path, text: str) -> list[str]:
  """Return the relative link targets in ``text`` that name no file or folder next to ``file``."""
  prose = _INLINE_CODE.sub("", _FENCED_CODE.sub("", text))
  targets = (match.group(1).partition("#")[0] for match in _LINK.finditer(prose))
  return [
    target
    for target in targets
    if target and ":" not in target and not (file.parent / target).exists()  # ":" marks a URL scheme
  ]


def _without_index(text: str) -> str:
  """Return the README's text without its generated index, the lines between the markers."""
  start, end = text.find(START), text.find(END)
  return text if start < 0 or end < start else text[:start] + text[end + len(END) :]


def _heading(text: str, fallback: str) -> str:
  """Return the text's first level-1 heading, or ``fallback`` when it has none."""
  for line in text.splitlines():
    if line.startswith("# "):
      return line[2:].strip()
  return fallback


def read(backlog: Path) -> tuple[list[Card], list[tuple[str, str]], list[str]]:
  """Return the cards, the background documents as (path, heading) pairs, and the errors found."""
  cards: list[Card] = []
  background: list[tuple[str, str]] = []
  errors: list[str] = []
  for file in sorted(backlog.rglob("*.md")):
    relative = file.relative_to(backlog).as_posix()
    text = file.read_text(encoding="utf-8")
    # The README's index is checked by rebuilding it: a stale index links the cards a commit deleted.
    written = _without_index(text) if file == backlog / "README.md" else text
    errors.extend(f"backlog/{relative}: the link {target} does not resolve" for target in broken_links(file, written))
    if file == backlog / "README.md" or file.name.startswith("_"):
      continue
    try:
      fields = _fields(text)
      if fields is None:
        background.append((relative, _heading(text, relative)))
      else:
        cards.append(card(relative.removesuffix(".md"), fields))
    except CardError as error:
      errors.append(f"backlog/{relative}: {error}")
  paths = {each.path for each in cards}
  errors.extend(
    f"backlog/{each.path}.md: blocked_by names {blocker}, which is not a card"
    for each in cards
    for blocker in each.blocked_by
    if blocker not in paths
  )
  return cards, background, errors


def _table(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
  """Return the lines of a Markdown table."""
  return [
    "| " + " | ".join(header) + " |",
    "|" + "|".join("---" for _ in header) + "|",
    *("| " + " | ".join(row) + " |" for row in rows),
  ]


def render(cards: list[Card], background: list[tuple[str, str]]) -> str:
  """Return the index: the cards waiting for a decision, the cards by priority, the parked cards and background."""
  ordered = sorted(cards, key=lambda each: (each.priority, each.area, each.title))
  counts = ", ".join(f"{sum(each.status is status for each in cards)} {status}" for status in Status)
  lines = [START, "", f"{len(cards)} cards: {counts}.", ""]

  decisions = [each for each in ordered if each.status is Status.NEEDS_DECISION]
  if decisions:
    lines += ["## Waiting for the maintainer", ""]
    lines += _table(("Card", "Area", "Priority"), [(each.link, each.area, str(each.priority)) for each in decisions])
    lines.append("")

  for priority in map(int, PRIORITIES):
    rows = [each for each in ordered if each.priority == priority and each.status is not Status.PARKED]
    if not rows:
      continue
    lines += [f"## Priority {priority}", ""]
    lines += _table(
      ("Card", "Area", "Kind", "Status", "Effort", "Blocked by", "Raised by"),
      [
        (
          each.link,
          each.area,
          each.kind.value,
          each.status.value,
          each.effort.value,
          ", ".join(f"[{path.rpartition('/')[2]}]({path}.md)" for path in each.blocked_by),
          _cell("; ".join(each.raised_by)),
        )
        for each in rows
      ],
    )
    lines.append("")

  parked = [each for each in ordered if each.status is Status.PARKED]
  if parked:
    lines += ["## Parked: outside the scope an ADR sets", ""]
    lines += _table(("Card", "Area", "Effort"), [(each.link, each.area, each.effort.value) for each in parked])
    lines.append("")

  if background:
    lines += ["## Background", ""]
    lines += [f"- [{_cell(heading)}]({path})" for path, heading in background]
    lines.append("")

  lines.append(END)
  return "\n".join(lines)


def main(backlog: Path = BACKLOG) -> int:
  """Check the cards and rewrite the README's index; return 1 when a check fails or the index changed."""
  readme = backlog / "README.md"
  cards, background, errors = read(backlog)
  text = readme.read_text(encoding="utf-8")
  start, end = text.find(START), text.find(END)
  if start < 0 or end < start:
    errors.append(f"backlog/README.md: the index markers {START} and {END} are missing or out of order")
  if errors:
    sys.stderr.write("".join(f"{error}\n" for error in errors))
    return 1
  updated = text[:start] + render(cards, background) + text[end + len(END) :]
  if updated == text:
    return 0
  readme.write_text(updated, encoding="utf-8")
  sys.stderr.write("backlog/README.md: the index was out of date and has been rewritten\n")
  return 1


if __name__ == "__main__":
  sys.exit(main())
