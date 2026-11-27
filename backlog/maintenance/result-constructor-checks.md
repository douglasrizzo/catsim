---
title: Simplify SimulationResult's constructor checks and other small duplication?
kind: decision
status: needs-decision
priority: 3
effort: S
blocked_by: []
raised_by: [ADR audit (#225)]
---

# Simplify SimulationResult's constructor checks and other small duplication?

The ADR audit (#225) listed simplification candidates and left them to the maintainer.

## Candidates

1. **Keep only three of `SimulationResult`'s constructor checks.** That saves about 90 source lines
   and 60 test lines. Its `__post_init__` checks the bank, the duration, every session, the proposal
   counts, the true abilities and `parts`. A result built by `simulate` is correct by construction.
   One built by hand, from sessions an application ran, is an explicit input, and ADR-0009 asks for
   it to be checked where it enters. Decide which checks guard that path.
2. **Merge the six "integer, not bool" checks** into one helper. The duration and `parts` checks
   above are two of them.
3. **Merge the two true-ability checks** in `simulation.py`: `_true_thetas` for `simulate`, and
   `_result_true_thetas` for the result.
4. **Reuse `checks._check_in_bank`** where `simulation.py` checks item indices by hand.
5. **Drop `_PartResult.exposure_counts`.** The result recounts exposure from the sessions.
6. **Drop the vulture entries** `WEIGHTED_INFO` and `writeable` in `pyproject.toml`, if they are no
   longer needed.

## Recommendation

Take 2 to 6: they remove duplication without changing behaviour. For 1, keep the checks that guard
hand-built results.
