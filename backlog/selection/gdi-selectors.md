---
title: GDI and GDIP selection (catR `criterion = "GDI"`, `"GDIP"`)
kind: technique
status: needs-research
priority: 3
effort: M
blocked_by: []
raised_by: [catR comparison]
---

# GDI and GDIP selection (catR `criterion = "GDI"`, `"GDIP"`)

**Family:** selection. **Confidence the gap is real:** medium: the scope is unclear.

## What is known

- catR 3 lists `GDI` and `GDIP` among `nextItem`'s criteria (Magis & Barrada 2017). The global
  discrimination index comes from Kaplan, de la Torre & Barrada (2015), proposed for cognitive
  diagnosis CAT; catR is believed to apply it to IRT banks, with `GDIP` its posterior-weighted
  version. Verify in catR's `GDI.R` and `nextItem.R`, and in Magis & Barrada (2017).
- [cd-cat](../scope/cd-cat.md) files GDI and GDIP under CD-CAT, which needs a diagnostic model
  catsim does not have and which ADR-0001's scope excludes. If catR's version works on unidimensional
  IRT banks, it is in scope; if it needs a diagnostic model, close this card and leave it to the
  CD-CAT discussion.

## Open questions

- Which of the two catR implements, and its exact definition on a 4PL bank.
- Whether `GDIP` needs a posterior over a grid like the planned MPWI and KLP selectors
  (this folder), so they could share one implementation of the posterior.
