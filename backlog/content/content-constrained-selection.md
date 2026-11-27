---
title: Content-constrained selection (content balancing, WDM, Sympson-Hetter with content, shadow tests)
kind: technique
status: needs-research
priority: 3
effort: L
blocked_by: []
raised_by: [catR comparison, strategies literature review (April 2026)]
---

# Content-constrained selection

> **Needs detailing before work starts.** This card merges four earlier cards. An agent wrote them
> in April 2026 from an unverified literature survey, against the API from before the rescoping. Their
> code sketches no longer apply, and the sources below have not been checked. Read the sources, choose
> the methods to support, and write the technique sections of [`_template.md`](../_template.md) for
> each before implementing.

## Problem

Operational CATs must meet a test specification: a share of items per content area, items that must
not appear together (enemies), and budgets such as word count. catsim selects on information or
difficulty alone, so a researcher can't simulate a content-constrained test. catR (`cbControl`),
mirtCAT and RSCAT all offer some form of content constraint.

## Candidate methods

From the simplest to the most complete:

1. **Constrained CAT (CCAT, content balancing).** Kingsbury & Zara (1989): choose the content area
   whose administered share falls furthest below its target, then select within it by the usual
   criterion. catR's `cbControl`/`cbGroup` is believed to implement this; check catR's `nextItem.R`.
2. **Weighted deviations method (WDM).** Stocking & Swanson (1993): each step picks the item that
   maximizes information minus a weighted sum of the deviations from every constraint's bounds.
   Greedy, with no look-ahead.
3. **Sympson–Hetter with content.** Kingsbury & Zara (1989) and Stocking & Lewis (1998): exposure
   control at the content-area level on top of each item's own, calibrated by alternating passes.
   It depends on [sympson-hetter](../exposure/sympson-hetter.md).
4. **Shadow tests (STA).** van der Linden & Reese (1998): before each item, solve an integer program
   for a full-length test that contains the items given so far, meets every constraint and has
   maximum information at the estimate. Then give its best item not yet administered. Every item
   given then belongs to some feasible test. It needs an ILP solver, which ADR-0021 (NumPy and SciPy
   only) rules out as a dependency unless an optional extra and an ADR say otherwise. SciPy's
   `milp` may be enough; check it before considering an extra.

## What catsim has now

- Per-item inputs of a technique, such as content groups, are arrays aligned with the bank's rows,
  passed to the technique (ADR-0003, ADR-0004), as `ClusterSelector` takes its cluster labels. The
  item bank carries no metadata of its own.
- `BaseExposureControl` accepts or declines the selector's proposal (ADR-0011).
- A selector reads the session through `StepContext` (ADR-0007), so the counts administered per
  content area can be computed from `context.administered_items` and the group array. No session
  state is kept.

## Open questions

- Which methods to support, and in what order. CCAT is the catR parity baseline; shadow tests are
  the reference for operational tests.
- Is content balancing a wrapper around any selector, a filter on the candidate items, or a
  selector of its own?
- How it combines with `r_max` caps, randomesque selection and Sympson–Hetter. Two mechanisms can
  over-constrain each other.
- Constraint specification: soft target shares (CCAT), hard lower and upper bounds (WDM, STA), and
  enemy sets. Decide one input format that all the methods share.
- What a simulation must report: the share administered per content area, and constraint violations
  per session.

## References (unverified)

- Kingsbury, G. G., & Zara, A. R. (1989). Procedures for selecting items for computerized adaptive
  tests. *Applied Measurement in Education*, 2(4), 359–375.
- Stocking, M. L., & Swanson, L. (1993). A method for severely constrained item selection in adaptive
  testing. *Applied Psychological Measurement*, 17(3), 277–292.
- Stocking, M. L., & Lewis, C. (1998). Controlling item exposure conditional on ability in
  computerized adaptive testing. *Journal of Educational and Behavioral Statistics*, 23(1), 57–75.
- van der Linden, W. J., & Reese, L. M. (1998). A model for optimal constrained adaptive testing.
  *Applied Psychological Measurement*, 22(3), 259–270.
- van der Linden, W. J. (2005). *Linear models for optimal test design*. Springer.
- Veldkamp, B. P. (2022). Review of the shadow-test approach to adaptive testing. *Behaviormetrika*,
  49, 169–190.
- Magis, D., & Barrada, J. R. (2017). Computerized adaptive testing with R: Recent updates of the
  package catR. *Journal of Statistical Software*, Code Snippets, 76(1).
