---
title: Choose the WLE's standard error (Warm's corrected SE, the sandwich SE)
kind: technique
status: needs-research
priority: 3
effort: S
blocked_by: []
raised_by: [Wang & Vispoel (1998) walkthrough]
---

# Choose the WLE's standard error (Warm's corrected SE, the sandwich SE)

## Problem

`WarmLikelihoodEstimator` reports only the plain Fisher SE, the inverse square root of the test
information at the estimate. The WLE technique page discusses Warm's corrected SE and the sandwich
SE, but neither can be chosen. A study that compares SE formulas, as Wang & Wang (2001) did, cannot
be run.

## Proposal (to settle in research)

- A `se=` option on `WarmLikelihoodEstimator`, with a `StrEnum` of the formulas (AGENTS.md: closed
  option sets are `StrEnum`s).
- Check each formula against its source: Warm (1989); Magis (2016) for the corrected SE. The page
  cites Magis's Equation 14. Say where catsim departs from the source.

## Tests

- Each formula on a hand-worked short test, against a value computed from the source's equation.
- The default stays the Fisher SE, so seeded results don't change unless the option is set.
