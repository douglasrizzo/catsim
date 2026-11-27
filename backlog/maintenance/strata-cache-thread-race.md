---
title: Make the stratified selectors' strata cache one attribute
kind: fix
status: ready
priority: 3
effort: XS
blocked_by: []
raised_by: [ADR audit (#225)]
---

# Make the stratified selectors' strata cache one attribute

## Problem

`StratifiedSelector.strata(item_bank)` caches its strata in two attributes, `_strata` and
`_strata_item_bank`, and checks then sets them in two steps:

```python
if self._strata is None or self._strata_item_bank is not item_bank:
  self._strata = self._build_strata(item_bank)
  self._strata_item_bank = item_bank
```

Two threads might share one selector across two banks, which an application embedding catsim may do.
Thread A can then set `_strata` for bank A, thread B can set both attributes for bank B, and A can set
`_strata_item_bank` back to bank A. The cache then pairs bank B's strata with bank A, and every later
call with bank A selects from bank B's indices. The result is silently wrong items, not an error.

catsim promises no thread safety (concurrency is the application's, ADR-0001), but a cache of
derived data (ADR-0008) should never pair data with the wrong source.

## Fix

Keep the pair in one attribute, assigned at once, such as
`self._strata_cache: tuple[ItemBank, list[IntArray]] | None`. A reader then sees an old pair or a new
pair, never a mixed one. The strata may still be built twice, which is harmless.

## Tests

- A test that interleaves the two banks' calls deterministically cannot reach the race through the
  public API. Instead, test the property the fix guarantees: after `strata(bank_b)`, then
  `strata(bank_a)`, each call returns the strata built from its own bank. Compare them with strata
  computed by hand from a sort of each bank's discriminations.
- If no test can fail on the old code without threads, say so in the PR instead of adding a test
  that cannot fail (the global testing rules).
