---
title: thOpt selection (catR `criterion = "thOpt"`)
kind: technique
status: needs-research
priority: 3
effort: S
blocked_by: []
raised_by: [catR comparison]
---

# thOpt selection (catR `criterion = "thOpt"`)

**Family:** selection. **Confidence the gap is real:** high.

> **Needs detailing before work starts.** catR's options here were recalled from memory, not read in
> its source. Confirm them in catR's `nextItem.R` and `startItems.R`, and expand the card, before
> designing anything.

## What is known

- catR's `thOpt` criterion selects the item whose ability of maximum information is closest to the
  current estimate; for the 3PL and 4PL that ability differs from the difficulty `b`. Magis (2013)
  is the likely source (the 4PL information function); catR also uses `thOpt` as a starting-item
  rule (`startSelect`). Verify both in catR's `nextItem.R` and `startItems.R`.
- catsim has `UrrySelector` (catR's `bOpt`: difficulty closest to the estimate) and computes each
  item's ability of maximum information already: `ItemBank.max_info_thetas` (`irt.max_info_theta`),
  used by the stratified selectors. A `thOpt` selector would reuse it.

## Open questions

- Whether catR's `thOpt` uses the closed-form maximum of the 4PL information or a numerical search,
  and whether catsim's `max_info_theta` agrees with it (a parity case).
- Ties and the exposure cap, consistently with ADR-0011 (random tie-breaking) and the other
  selectors.
