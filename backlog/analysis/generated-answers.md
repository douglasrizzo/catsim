---
title: A public function that draws a generated examinee's answers
kind: feature
status: needs-research
priority: 3
effort: S
blocked_by: []
raised_by: [notebook rewrite (#226)]
---

# A public function that draws a generated examinee's answers

## Problem

`simulate(engine, true_thetas=...)` gives each generated examinee answers to the whole bank, drawn
from the examinee's own stream (ADR-0013, ADR-0014). No public function draws those answers outside
a run. Some tasks need them:

- build a response matrix once and run every engine on it as recorded examinees;
- check a custom analysis against the answers a run used;
- teach what a generated examinee is, as the notebooks do.

The notebook rewrite of #226 met this gap and wrote the draws by hand. Hand-written draws do not
match `simulate`'s, so the two cannot be compared answer for answer.

## Proposal (to settle in research)

- A function, for example `catsim.simulation.answers(item_bank, true_thetas, rng)`, returning a
  boolean matrix of examinees × items. Row k equals what `simulate` draws for examinee k on the same
  `rng`. Name it by ADR-0023's naming table.
- Decide whether `simulate` itself calls it, so there is one implementation of the draw.

## Tests

- `simulate(engine, responses=answers(bank, thetas, rng=s), ...)` gives the same sessions as
  `simulate(engine, true_thetas=thetas, rng=s)` when the engine draws nothing else. The expected
  value comes from the existing run, not from the new function.
- The share of correct answers at θ matches `irt.icc` within Monte Carlo error over many draws.
