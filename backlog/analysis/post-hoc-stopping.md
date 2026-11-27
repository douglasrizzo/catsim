---
title: Evaluate stopping rules post hoc on a maximum-length run
kind: feature
status: blocked
priority: 3
effort: M
blocked_by: [analysis/derived-results]
raised_by: [Babcock & Weiss (2012) walkthrough]
---

# Evaluate stopping rules post hoc on a maximum-length run

Found by the Babcock & Weiss (2012) replication of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

Without exposure feedback, the test a variable-length rule gives is a prefix of the maximum-length test
on the same seed: selection does not depend on the stopper. So a stopping-rule study can run the
maximum length once and apply each rule to the prefixes, instead of running every rule.

## Evidence

The agent applied every variable-length rule post hoc to the Fixed-50 sessions and reproduced catsim's
own runs exactly in all 44 cells (lengths and θ̂). The post-hoc pass took 8 s for the 44 cells, against
about 315 s for running every rule (the walkthrough's `crosscheck_posthoc.py` (kept locally)).

## Proposed shape (for the implementer to settle)

`result.stopped(engine_or_rule)` (name to settle): for each session, the shortest prefix at which the
rule stops, as a truncated result (derived-results's `truncated` generalised to a per-session length).

## Validity conditions (document them in the docstring as the call contract)

- No exposure feedback across sessions (no `r_max` cap, no exposure control that reads exposure rates),
  or the prefixes are not the tests the rule would have given.
- The full run's maximum length is at least the rule's maximum.
- The rule's stoppers draw no randomness, or their draws differ from a real run (`context.rng` of a
  rebuilt context is not the session's stream; check ADR-0013).
- A selector whose choice depends on the planned test length (`AStratSelector(test_size=...)`) gives
  different prefixes for different lengths: excluded.

Decide whether catsim checks these (it can see `r_max` on known selectors, not on custom ones) or only
documents them. Performance is not a requirement (AGENTS.md); the value is the exact correspondence,
which also makes rules comparable session by session.

## Tests

- On a seeded run without a cap, the post-hoc result equals running the rule directly: lengths and
  estimates, session by session (the agent's cross-check, as the reference).
