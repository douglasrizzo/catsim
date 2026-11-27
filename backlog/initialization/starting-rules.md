---
title: Starting rules (catR `randomCAT`'s `start` list)
kind: technique
status: needs-research
priority: 3
effort: M
blocked_by: []
raised_by: [catR comparison]
---

# Starting rules (catR `randomCAT`'s `start` list)

**Family:** initialization. **Confidence the gap is real:** medium: catsim starts differently, and
part of catR's options may have catsim equivalents.

> **Needs detailing before work starts.** catR's options here were recalled from memory, not read in
> its source. Confirm them in catR's `startItems.R` and the manual, and expand the card, before
> designing anything.

## What is known

- catR starts a test with one or more items chosen before the first estimate: the `start` list of
  `randomCAT` and the function `startItems` take fixed items (`fixItems`), a number of starting
  items (`nrItems`) at given abilities (`theta`), and a rule to choose them (`startSelect`: `bOpt`,
  `thOpt`, `MFI`, `progressive`, `proportional`), with randomesque and a seed. Verify the exact
  options in catR's `startItems.R` and the manual.
- catsim's initializers (`FixedPointInitializer`, `NormalInitializer`, `UniformInitializer`) give
  one starting ability; the selector then picks the first item at it, and every later item at the
  estimate after each response.
- So catR can give several items before estimating, or fixed items, which catsim cannot express
  except through a custom selector.

## Open questions

- Which of catR's starting options catsim cannot already express, and whether they belong to the
  initializer (a starting ability), a selector (the first items) or a new strategy type.
- How catR estimates after several starting items with constant responses, which overlaps with
  [constant-patterns-and-hybrid](../estimation/constant-patterns-and-hybrid.md).
