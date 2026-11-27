---
title: Content balancing (catR `cbControl`)
kind: technique
status: needs-research
priority: 3
effort: L
blocked_by: []
raised_by: [catR comparison]
---

# Content balancing (catR `cbControl`)

**Family:** selection (a constraint layered on a selector). **Confidence the gap is real:** high.

## What is known

- catR's `nextItem`, `randomCAT` and `simulateRespondents` take `cbControl` (the content groups'
  names and target proportions) and `cbGroup` (each item's group). Magis & Raîche (2012) and
  Magis & Barrada (2017) describe it; the method is believed to be Kingsbury & Zara's (1989):
  pick the content group whose administered proportion falls furthest below its target, then
  select within that group by the chosen criterion. Verify in catR's `nextItem.R`.
- catsim has no per-item metadata besides the IRT parameters. Items have string names (ADR-0004);
  per-item inputs a technique needs, such as content groups, are proposed as arrays aligned with
  the bank's rows (see [item-bank-metadata-schema](item-bank-metadata-schema.md)).
- Related discussions: [weighted-deviations](weighted-deviations.md) (a different content-balancing
  method, Stocking & Swanson 1993), [item-bank-metadata-schema](item-bank-metadata-schema.md) (where metadata
  lives), [sympson-hetter-with-content](../exposure/sympson-hetter-with-content.md).

## Open questions

- Kingsbury & Zara's rule exactly as catR implements it, including ties between groups and groups
  with no items left.
- Whether content balancing is a selector wrapper, a filter on the candidates before any selector,
  or an option of every selector, under the rescoped strategy contract.
- How it composes with exposure caps (`r_max`) and with randomesque selection.
