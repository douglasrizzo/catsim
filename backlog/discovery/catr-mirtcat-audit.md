---
title: Audit catsim against catR and mirtCAT
kind: research
status: ready
priority: 2
effort: M
blocked_by: []
raised_by: []
---

# Audit catsim against catR and mirtCAT

## Goal

Compare catsim with the two reference R packages, covering what each one computes, the numbers on
shared cases, and what a researcher can report with each. Turn the gaps that matter for research
into cards.

## Scope

- **catR.** The parity tests already compare catsim with catR 3.17 on fixed cases, whole tests and
  simulations (ADR-0020, `CONTRIBUTING.md`). The audit checks the rest:
  - catR features with no catsim counterpart; the `needs-research` technique cards came from memory
    and need verifying;
  - options catsim lacks, such as `constantPatt`, `D` and the item-eligibility `Mrmax`;
  - outputs of `simulateRespondents` that catsim's results do not give.
- **mirtCAT.** Nothing compares catsim with it yet. Check:
  - its design-object options (selection criteria, exposure and content constraints, stopping
    rules);
  - its preCAT stage;
  - its `generate_pattern` and `mirtCAT(..., design = ...)` outputs.

  Note which ones rest on polytomous or multidimensional models, which ADR-0002 excludes
  ([scope](../scope/)).

## How

- Read each package's manual and source. Run shared cases in R where catsim has the method, and
  record disagreements.
- Write findings into the existing cards where they fit, such as
  [content-balancing](../content/content-balancing.md) and
  [starting-rules](../initialization/starting-rules.md).
- Write new cards for the rest, with `raised_by: [catR and mirtCAT audit]`.

## Done when

Every `needs-research` card that cites catR is confirmed, corrected or deleted, and mirtCAT's
methods within ADR-0002's scope each have a card or a note that catsim already covers them.
