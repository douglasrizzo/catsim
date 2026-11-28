---
title: Link irtlib's names in the documentation once irtlib publishes its own
kind: docs
status: blocked
priority: 3
effort: S
blocked_by: [maintenance/build-on-irtlib]
raised_by: [maintainer on 2026-11-27]
---

# Link irtlib's names in the documentation once irtlib publishes its own

## Problem

catsim's documentation names irtlib's objects, such as `irtlib.ItemBank` and `irtlib.irt.icc`, as
literals with no link, because irtlib has no documentation site to link to. A reader has to find
each object in irtlib's source. Blocked also by irtlib's `maintenance/docs-site` card.

## What catsim will have

After `maintenance/build-on-irtlib`: irtlib's names written as literals, without a `:py:` role;
`nitpick_ignore_regex` in `sphinx/conf.py` matching `irtlib\..*`, so annotations rendered from
signatures build; and one link to irtlib's README from the API index and the installation section.

## Proposal

- Add irtlib to `intersphinx_mapping` in `sphinx/conf.py`, and drop the `irtlib\..*` entry of
  `nitpick_ignore_regex`.
- Turn the literals into cross-references, and point the README links at irtlib's site.
- The maintainer then reviews catsim's documentation against irtlib's, so that each topic has one
  home (`sphinx/AGENTS.md`).

## Tests

- `uv run sphinx-build -W` builds with no unresolved reference to irtlib.
