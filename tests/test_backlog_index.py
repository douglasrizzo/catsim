"""Tests for backlog/_index.py: the checks on the cards' frontmatter and links, and the index it writes."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "backlog" / "_index.py"


def _load() -> ModuleType:
  spec = importlib.util.spec_from_file_location("backlog_index", _SCRIPT)
  assert spec is not None
  assert spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  sys.modules[spec.name] = module  # dataclasses resolve the module's annotations through sys.modules
  spec.loader.exec_module(module)
  return module


index = _load()

README = "# Backlog\n\nIntro.\n\n<!-- backlog-index:start -->\n<!-- backlog-index:end -->\n\nAfter.\n"
VALID = {
  "title": "Do the thing",
  "kind": "feature",
  "status": "ready",
  "priority": "2",
  "effort": "S",
  "blocked_by": "[]",
  "raised_by": "[a review]",
}


def _card(fields: dict[str, str], body: str = "# Do the thing\n") -> str:
  return "---\n" + "".join(f"{key}: {value}\n" for key, value in fields.items()) + "---\n\n" + body


@pytest.fixture
def backlog(tmp_path: Path) -> Path:
  """Return an empty backlog folder with a README that has the index markers and an analysis/ area."""
  (tmp_path / "README.md").write_text(README)
  (tmp_path / "analysis").mkdir()
  return tmp_path


def test_the_index_lists_a_card_once_and_a_second_run_changes_nothing(backlog: Path) -> None:
  """A card appears as one row of its priority's table, and rerunning on an up-to-date README exits 0.

  Fails if the row's columns are out of order, if the text outside the markers changes, or if a
  second run rewrites the index again, which would make the hook fail on every commit.
  """
  (backlog / "analysis" / "thing.md").write_text(_card(VALID))

  assert index.main(backlog) == 1
  text = (backlog / "README.md").read_text()
  assert "| [Do the thing](analysis/thing.md) | analysis | feature | ready | S |  | a review |" in text
  assert text.count("analysis/thing.md") == 1
  assert text.startswith("# Backlog\n\nIntro.\n\n<!-- backlog-index:start -->")
  assert text.endswith("<!-- backlog-index:end -->\n\nAfter.\n")
  assert index.main(backlog) == 0


def test_a_deleted_card_leaves_the_index(backlog: Path) -> None:
  """Deleting a card, as the PR that does its work must, removes its row on the next run.

  Fails if the index keeps rows for cards that no longer exist, or if the stale row's link to the
  deleted card is reported as broken, which would stop the run before it rewrites the index.
  """
  (backlog / "analysis" / "thing.md").write_text(_card(VALID))
  (backlog / "analysis" / "other.md").write_text(_card({**VALID, "title": "Other"}))
  index.main(backlog)

  (backlog / "analysis" / "thing.md").unlink()

  assert index.main(backlog) == 1
  text = (backlog / "README.md").read_text()
  assert "analysis/thing.md" not in text
  assert "[Other](analysis/other.md)" in text


@pytest.mark.parametrize(
  ("text", "message"),
  [
    (_card({**VALID, "kind": "wish"}), "kind is 'wish'"),
    (_card({**VALID, "priority": "7"}), "priority is '7'"),
    (_card({**VALID, "owner": "me"}), "unknown field owner"),
    (_card({key: value for key, value in VALID.items() if key != "effort"}), "missing effort"),
    (_card({**VALID, "blocked_by": "analysis/other"}), "must be an inline list"),
    (_card({**VALID, "blocked_by": "[analysis/thing]", "status": "blocked"}), "names the card itself"),
    (_card({**VALID, "blocked_by": "[analysis/other]"}), "status is ready, but blocked_by names analysis/other"),
    ("---\ntitle: Do the thing\nstatus: ready\nstatus: blocked\n---\n", "repeats status"),
    ("---\ntitle: Do the thing\n", "no closing '---' line"),
    ("--- \ntitle: Do the thing\n---\n", "opens with '---' alone"),
  ],
)
def test_an_invalid_card_fails_and_leaves_the_readme_alone(
  backlog: Path, capsys: pytest.CaptureFixture[str], text: str, message: str
) -> None:
  """Each kind of invalid frontmatter is reported with the card's path, and the README is not rewritten.

  Fails if a bad value is accepted, or if a card whose opening line is not exactly '---' is quietly
  listed as background instead of reported.
  """
  (backlog / "analysis" / "other.md").write_text(_card({**VALID, "title": "Other"}))
  (backlog / "analysis" / "thing.md").write_text(text)

  assert index.main(backlog) == 1
  errors = capsys.readouterr().err.splitlines()
  assert len(errors) == 1
  assert errors[0].startswith("backlog/analysis/thing.md: ")
  assert message in errors[0]
  assert (backlog / "README.md").read_text() == README


def test_a_blocker_that_is_not_a_card_is_reported(backlog: Path, capsys: pytest.CaptureFixture[str]) -> None:
  """A blocked_by entry must name an existing card.

  Fails if a card stays blocked by a card that was deleted or renamed.
  """
  (backlog / "analysis" / "thing.md").write_text(_card({**VALID, "status": "blocked", "blocked_by": "[gone/card]"}))

  assert index.main(backlog) == 1
  assert "blocked_by names gone/card, which is not a card" in capsys.readouterr().err


def test_a_broken_relative_link_is_reported_but_code_and_urls_are_not(
  backlog: Path, capsys: pytest.CaptureFixture[str]
) -> None:
  """A relative link to a missing file fails; links inside code, URLs and anchors to existing files pass.

  Fails if a card moved to another folder can keep a dead link, or if `table[i](x)` in a sketch is
  read as a link.
  """
  body = (
    "# Do the thing\n\n[gone](../content/gone.md), [ok](other.md#section), [web](https://example.org/a.md),\n"
    "`f[i](x)` and\n\n```python\nvalues[k](theta)\n```\n"
  )
  (backlog / "analysis" / "other.md").write_text(_card({**VALID, "title": "Other"}))
  (backlog / "analysis" / "thing.md").write_text(_card(VALID, body))

  assert index.main(backlog) == 1
  errors = capsys.readouterr().err.splitlines()
  assert errors == ["backlog/analysis/thing.md: the link ../content/gone.md does not resolve"]


def test_a_file_without_frontmatter_is_listed_as_background(backlog: Path) -> None:
  """A Markdown file with no frontmatter is background, listed by its first heading, not a card.

  Fails if a literature review or a folder's README is rejected as a card, or counted as one.
  """
  (backlog / "literature-review.md").write_text("# Literature review\n\nNotes.\n")
  (backlog / "analysis" / "thing.md").write_text(_card(VALID))

  assert index.main(backlog) == 1
  text = (backlog / "README.md").read_text()
  assert "- [Literature review](literature-review.md)" in text
  assert "1 cards:" in text
