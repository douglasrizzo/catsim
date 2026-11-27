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

## Leads to check

> **Needs detailing.** The coordinator recalled these from memory of catR and never checked them.
> Confirm each one in catR's source and manual. Turn a real gap into a card with its sources, and
> drop a wrong claim.

- **Jeffreys prior.** `priorDist = "Jeffreys"` in catR's Bayesian estimators. It depends on the items
  given (the square root of the test information), so a fixed log-prior callable cannot express it.
  For the 1PL and 2PL it equals catsim's WLE; under the 3PL the two differ.
- **Robust estimation.** catR's `thetaEst` may have a `"ROB"` method, possibly Schuster & Yuan
  (2011).
- **Item-eligibility exposure control.** `Mrmax = "IE"` in `randomCAT` and `simulateRespondents`,
  van der Linden & Veldkamp (2004). Each item is eligible with a probability updated from its
  observed exposure, often presented as a way around Sympson–Hetter's calibration. Check that it
  fits `BaseExposureControl` with exposure rates from the caller (ADR-0006, ADR-0012).
- **GDI and GDIP selection.** `criterion = "GDI"`/`"GDIP"` in `nextItem`, from Kaplan, de la Torre &
  Barrada (2015), proposed for cognitive diagnosis. The gap is in scope only if catR applies it to
  unidimensional IRT banks; otherwise it belongs to [cd-cat](../scope/cd-cat.md). GDIP may share a
  posterior grid with the planned MPWI and KLP selectors.

## How

- Read each package's manual and source. Run shared cases in R where catsim has the method, and
  record disagreements.
- Write findings into the existing cards where they fit, such as
  [content-constrained-selection](../content/content-constrained-selection.md) and
  [starting-rules](../initialization/starting-rules.md).
- Write new cards for the rest, with `raised_by: [catR and mirtCAT audit]`.

## Done when

Every `needs-research` card that cites catR is confirmed, corrected or deleted, and mirtCAT's
methods within ADR-0002's scope each have a card or a note that catsim already covers them.
