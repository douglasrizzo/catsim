---
title: Save and load simulation results, and the cost of unpickling them
kind: decision
status: needs-decision
priority: 3
effort: M
blocked_by: []
raised_by: [Wang & Vispoel (1998) walkthrough, Babcock & Weiss (2012) walkthrough, Chang & Ying (1999) walkthrough]
---

# Save and load simulation results, and the cost of unpickling them

Found by all three replications of the research walkthrough ([the research walkthrough](../discovery/README.md)).

## Problem

A study runs once and analyses many times, so researchers save results. catsim has no save or load;
pickle works, but it is large and slow to read back, because `SimulationResult.__reduce__` rebuilds the
result through its constructor and every `Session` is rebuilt and checked.

## Evidence

- Wang & Vispoel: a pickled 13,000-session result is 15 MB and takes 0.9 s to unpickle; their plotting
  script took 74 s, mostly unpickling.
- Chang & Ying: 7.6 MB per 5,000 sessions; unpickling 157k sessions took 24 s, mostly in the
  frozen-dataclass `__setstate__`.
- Babcock & Weiss: 128 results pickled to 133 MB.

Performance is not a requirement (AGENTS.md); this card is about whether a researcher can keep a study's
raw results, not about speed for its own sake.

## Options (for the implementer and maintainer)

1. **Document the tables as the saved form.** `steps()` and `examinees()` are dicts of NumPy columns:
   `numpy.savez` writes them, and a constructor from `steps()` (derived-results-style) rebuilds a result. Smallest
   change; the bank must be saved too (`ItemBank` has its own matrix and names).
2. **`SimulationResult.save(path)` / `load(path)`** in a NumPy format. Adds a file format catsim must keep
   stable; weigh against ADR-0001's scope (catsim does not store live tests, but a study's results are
   not live tests).
3. **Make unpickling cheaper** while keeping the checks (ADR-0009 checks inputs where they enter catsim;
   a pickled result re-entering is an input). Measure first.

Report the measurements before and after any change, as AGENTS.md asks for performance changes.

## Tests

- A round trip (save then load, or tables then rebuild) gives equal tables and metrics.
