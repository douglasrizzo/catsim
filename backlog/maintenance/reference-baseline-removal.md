---
title: When to delete the rescoping's reference baseline?
kind: decision
status: needs-decision
priority: 3
effort: XS
blocked_by: []
raised_by: [reference baseline (#218)]
---

# When to delete the rescoping's reference baseline?

## What it is

`tests/reference/` holds:

- reference outputs recorded from dev at adda72dc, before the rescoping changed the code;
- the generator, the adapter and the inventory script.

`tests/test_reference_outputs.py` checks them. The outputs cover every deterministic computation:
IRT functions, estimates on fixed responses, each selector's choice in a fixed context, the
stoppers, the initializers and the metrics. The baseline proved that the rescoping lost no
computation, independently of the tests it rewrote.

It compares catsim only with itself. The unit tests and the catR parity tests carry correctness on
their own. Every PR that changes a recorded result must regenerate the data and name the ADR that
changes it, which adds friction to each such change.

## The question

When does it go?

## Options

1. **Delete it now.** The rescoping's refactors are merged.
2. **Delete it after the analysis and estimation cards of priority 1 and 2 land.** Some of them
   touch results, and the baseline catches an accidental change there.
3. **Keep it.**

Recommendation: option 2.

Deleting it removes `tests/reference/`, `tests/test_reference_outputs.py`, and any mention of them
in `CONTRIBUTING.md`, `AGENTS.md` and `pyproject.toml`.
