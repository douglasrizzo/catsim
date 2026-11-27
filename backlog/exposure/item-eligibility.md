---
title: Item-eligibility exposure control (catR `Mrmax = "IE"`, unverified)
kind: technique
status: needs-research
priority: 3
effort: M
blocked_by: []
raised_by: [catR comparison]
---

# Item-eligibility exposure control (catR `Mrmax = "IE"`, unverified)

**Family:** selection (exposure control). **Confidence the gap is real:** low: recalled from memory,
not read in catR's documentation.

## What is known

- The card's author recalled, without checking, that catR's `randomCAT` and `simulateRespondents` take `rmax` with a
  method `Mrmax`: `"restricted"` (items at the maximum rate are not selected, which catsim's
  `r_max` does, see ADR-0012) and `"IE"`, the item-eligibility method of van der Linden & Veldkamp
  (2004), where each item is eligible with a probability updated from its observed exposure.
  **Verify that `Mrmax` and `"IE"` exist** before anything else; if not, delete this card.
- catsim has `BaseExposureControl` (an accept-or-decline hook on the selector's proposal) and a spec
  for Sympson-Hetter ([sympson-hetter](sympson-hetter.md)), whose implementation notes say
  why its calibration is hard. Item eligibility is often presented as a way around that
  calibration.

## Open questions

- Whether item eligibility fits `BaseExposureControl` as it is, under the contract where
  exposure rates come from the caller (ADR-0006, ADR-0012).
- How it differs from Sympson-Hetter in what a simulation must report (eligibility probabilities as
  well as exposure rates).
