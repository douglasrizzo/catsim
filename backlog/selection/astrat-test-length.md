---
title: Check a-stratified selection's test size against the stopper
kind: feature
status: ready
priority: 3
effort: S
blocked_by: []
raised_by: [Chang & Ying (1999) walkthrough]
---

# Check a-stratified selection's test size against the stopper

Found by the Chang & Ying (1999) replication of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

- `CatEngine(..., AStratSelector(test_size=40), ..., MaxLengthStopper(41))` builds without complaint
  and raises on the first examinee's 41st item. The message is clear, but the mismatch is already
  visible in the configuration.
- With a variable-length rule, `StopRule(StandardErrorStopper(.3), MaxLengthStopper(40))`, it runs with
  no warning. Tests that stop early never reach the high-a strata, which defeats the method.
- The `AStratSelector` docstring says the stopper must end the test by `test_size`, but not that the
  method assumes fixed-length tests. The technique page
  (`sphinx/api/techniques/selection/stratified-selector.md`) covers only the first point.

## What to do

- Decide whether `CatEngine` can check this. It would need the stopper's maximum length, which only
  `MaxLengthStopper` (possibly inside a `StopRule`) knows. A check that sees through `StopRule` is
  feasible. A general one is not, because custom stoppers are opaque. Prefer a check of known types
  only if it stays small. Otherwise document.
- Say on the technique page and in the docstrings of the stratified selectors (the call contract) that
  the strata assume a fixed test length of `test_size`.

## Tests

- If the check is added: building the engine with a mismatched `MaxLengthStopper` raises, and the
  matching one does not.
