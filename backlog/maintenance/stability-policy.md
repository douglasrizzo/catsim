---
title: A stability and versioning policy?
kind: decision
status: needs-decision
priority: 4
effort: S
blocked_by: []
raised_by: [rescoping requirements]
---

# A stability and versioning policy?

catsim states no stability policy, by the maintainer's choice. This card holds the proposal to look
at when the maintainer wants one.

## The proposal

- Semantic versioning over the documented API, meaning what the API reference lists. A breaking
  change then needs a major version.
- Seeded results change in a minor version only to fix a bug, and the release notes say so.
- The random model changes only in a major version. That model is which draw comes from which
  stream (ADR-0013).
- Across platforms, results agree up to floating-point rounding.

## Context

- catR states no policy. Its API has been stable across its 3.x releases.
- catsim already guarantees one thing (ADR-0013): the same seed on the same catsim version gives the
  same results.
- Questions to settle together with this one:
  - which Python, NumPy and SciPy versions catsim supports, and for how long (for example the
    Scientific Python SPEC 0);
  - whether a deprecation warns for a release before a removal.

  AGENTS.md has no deprecation shims for now.
