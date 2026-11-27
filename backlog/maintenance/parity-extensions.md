---
title: Tighter and wider catR parity tests
kind: maintenance
status: ready
priority: 3
effort: M
blocked_by: []
raised_by: [catR parity whole tests (#223), technique pages (#226)]
---

# Tighter and wider catR parity tests

The parity tests compare catsim with catR 3.17 at three levels (ADR-0020, `CONTRIBUTING.md`). Two
kinds of extension are open.

## Tighter level-3 bounds (paired differences)

Level 3 compares simulations within Monte Carlo error. Both packages use the same fixed true
abilities, so the spread between abilities counts as noise, and the bounds are conservative. The
mean-length standard error in `randomesque_precision` is about 40% too wide: 0.131 against 0.094
paired.

Recording catR's per-examinee errors, lengths and administered matrix in `simulations.json`, and
testing paired differences, would tighten the bounds by about 30%. A mean-length shift of 0.42 items
would then be caught, where 0.59 passes today. Today the bound is documented as conservative.

## Cases that could exist and don't

- the exposure cap (`r_max`);
- `RandomSelector` and the 5-4-3-2-1 rule at level 3;
- catR's ML default for constant patterns, which
  `MAPEstimator(uniform_log_prior(-4, 4), bounds=(-4, 4))` reproduces. A case with that setting
  could replace an expected failure.

## How

Extend `tests/parity/catr/` (its R scripts, and `generate.R` to regenerate), following
`CONTRIBUTING.md`'s parity section. Regenerate the fixtures and check that the existing data stays
byte for byte.
