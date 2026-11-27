---
title: Decide whether to reconsider the study layer that ADR-0016 ruled out
kind: decision
status: needs-decision
priority: 2
effort: S
blocked_by: []
raised_by: [Wang & Vispoel (1998) walkthrough, Chang & Ying (1999) walkthrough]
---

# Decide whether to reconsider the study layer that ADR-0016 ruled out

Raised by the Wang & Vispoel (1998) and Chang & Ying (1999) replications of the research walkthrough ([discovery](../discovery/README.md)). This card is a decision for the maintainer, not
implementation work.

## The trigger ADR-0016 set

ADR-0016 rejects "A study layer of conditions, replications and aggregation: one design among many a
study may need, which the study's own loop expresses directly; to reconsider if research use shows the
loop is where the work is."

## What the research use showed

- Wang & Vispoel: the runner was about 60 lines and the analysis about 200; most of it reshapes results
  across pool × estimator × length × θ into long tables and figures.
- Chang & Ying: pooling replications, averaging across them with SDs, and paired common-random-number
  differences (`metrics.mean(se_a - se_b)` from two `examinees()` tables) were where most of the code
  went.
- Babcock & Weiss: loops over banks × rules × populations, plus a `marginal_row()` helper.

Two of three agents named the loop as the main cost; the third wrote one without complaint.

## Options

1. **Keep ADR-0016 as it is.** Cards derived-results (pooling, subsets, truncation) and run-summary (a one-row summary)
   remove most of the reshaping without a study layer. Add a guide section showing the loop.
2. **Small helpers, no layer.** Besides derived-results and run-summary: a function that concatenates labelled results'
   tables into one long table (`{label: result}` → columns plus a `label` column), and a paired
   difference with its MC error between two results on the same seed and examinees
   (`running-simulations.md` already computes one by hand).
3. **A study layer.** Conditions × replications, run and aggregated by catsim. ADR-0016 lists why not.

Recommendation: decide after derived-results and run-summary land, then choose between 1 and 2 by rewriting one of the three
studies' scripts (kept locally by the maintainer) with them and counting what remains.

## If the decision changes ADR-0016

Update ADR-0016's rejected-alternatives entry, or add an ADR, in the same PR as the helpers.
