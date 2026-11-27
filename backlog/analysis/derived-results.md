---
title: Subset, truncate and pool results, and a by-length table
kind: feature
status: ready
priority: 1
effort: M
blocked_by: []
raised_by: [Wang & Vispoel (1998) walkthrough, Chang & Ying (1999) walkthrough]
---

# Subset, truncate and pool results, and a by-length table

Found by the Wang & Vispoel (1998) and Chang & Ying (1999) replications of the research walkthrough ([discovery](../discovery/README.md)).

## Problem

A study often analyses parts of a run, or several runs together, with the same tables and metrics:

1. **Truncation.** Bias, SD and RMSE by θ after 5, 10, 20 and 30 items of the same test. Metrics exist
   only at the final length. `plot.metric_by_length` computes per-length metrics internally
   (`_estimates_by_length`) but exposes no table, and cannot be grouped by ability level.
2. **Subsets.** The metrics of one θ point, or one condition, as a result of its own, for
   `plot.metric_by_length` or `overlap_rate`.
3. **Pooling.** Replications of one engine combined into one result: concatenated sessions, summed
   `proposal_counts`, concatenated `true_thetas`.

Both agents rebuilt `SimulationResult` from edited sessions to get these. The constructor is
keyword-only (`kw_only=True`), which `help()` does not make visible, so the first attempt raised a
`TypeError`.

## Evidence

Wang & Vispoel's workaround (the walkthrough's `catsim_native.py` (kept locally)):

```python
sessions = tuple(Session(administered_items=s.administered_items[:n], responses=s.responses[:n],
                 estimates=s.estimates[:n+1], stop_reason=StopReason.STOPPER) for s in r.sessions)
counts = numpy.bincount(numpy.concatenate([s.administered_items for s in sessions]), minlength=M)
SimulationResult(item_bank=r.item_bank, sessions=sessions, proposal_counts=counts,
                 duration=r.duration, true_thetas=r.true_thetas).ability_levels()
```

Chang & Ying pooled 5 replications × 5,000 examinees the same way
(the walkthrough's `report.py` (kept locally)).

## Proposed shape (for the implementer to settle)

- `result.subset(rows)`: a result of the sessions at `rows` (indices or a boolean mask), with their
  true abilities.
- `result.truncated(n_items)`: each session cut after its first `n_items` items, sessions already
  shorter kept whole.
- `SimulationResult.concat(results)` (or a module function): one result from results over the same
  item bank.
- A by-length table, one row per test length (optionally crossed with ability level), backing
  `plot.metric_by_length`, so the plot and the table share one implementation.

## Open questions

- **Proposal counts.** A truncated or subset result's `proposal_counts` cannot be recovered exactly:
  `Session.excluded_items` records which items the exposure control declined, not when. Options: the
  administrations only (exact without an exposure control), or the declines as well when the session
  records their positions. Decide and document; say in the docstring when the counts are approximate.
- **`duration` and `parts`** of a derived result: the source's, the sum, or NaN; say which.
- **Stop reason** of a truncated session: `STOPPER` is what the workaround used; a truncation is not a
  stopper's decision, so check ADR-0010 and ADR-0016 before choosing.
- Does this need an ADR? ADR-0016 shapes `SimulationResult`; adding derived results probably extends
  it rather than contradicting it. Update ADR-0016 if the result's invariants change.

## Tests

- `truncated(n)` metrics equal metrics computed by hand from `steps()` at position `n - 1`.
- `subset(mask)` of one θ point equals the matching `ability_levels()` row.
- `concat` of two seeded runs equals one run over both populations when there is no exposure feedback
  (same entropy per examinee? check ADR-0013: streams are per examinee position, so the second run's
  streams differ; build the expected value from the two runs' sessions, not from a third run).
- The by-length table against the values `plot.metric_by_length` draws.
