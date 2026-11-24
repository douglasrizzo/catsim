# ADR-0003: Keep Runtime State Outside Domain Objects

## Status

Accepted 2026-11-24

## Context

Runtime state was entangled with long-lived objects: the item bank kept a column of exposure
rates beside its parameters, so a bank reused across runs carried one run's exposure into the
next.

The exposure rates a selector checks its `r_max` cap against were the tracker's counts divided by
the planned size of the run, which the caller passed in. That caused two problems:

- Without the planned size, the engine passed no rates at all, so an `r_max` cap did nothing and
  nothing said so. That is the natural setup for a live deployment, which does not know in advance
  how many examinees it will test.
- With it, early examinees saw every rate near zero and the cap bore only on late ones, so the
  precision of an examinee's test depended on their place in the run.

catR's `simulateRespondents` divides by the examinees tested so far, with no restriction for the
first respondent and none at all when `rmax` is 1.

## Decision

Calibrated item data is domain data; runtime state lives in dedicated execution objects:

- `ItemBank` holds the item parameters and the values derived from them, in read-only arrays.
- `CatSessionState` holds one examinee's runtime state, the items it excluded included (ADR-0016).
- `ExposureTracker` and `SimulationResult` hold run-level exposure information.

`ExposureTracker` owns the denominator of the exposure rates, which follow catR:

- The tracker counts the sessions started as well as each item's administrations and proposals
  (administrations plus the exposure control's declines, which never enter the rates). The engine
  updates it.
- A session is selected against each item's administrations divided by the number of other
  sessions started so far. With sessions run one after another, as `SimulationRunner` runs them,
  that is catR's rate: the share of the examinees tested before the current one.
- The rates are computed when a session asks, not fixed when it starts, so one tracker serves
  sessions that run side by side without keeping anything per session.
- An `r_max` of 1 applies no cap, as in catR: an item that every earlier examinee received has a
  rate of exactly 1, which a strict `rate < r_max` test would pass over.
- `SimulationResult.exposure_rates` divides the run's counts by all its sessions: the exposure at
  the end of the run, not the rates any session saw.

## Consequences

- An item bank can be reused across sessions and simulations, and its cached values cannot go
  stale.
- A context with a tracker is all exposure control needs, so a live deployment gets it without
  knowing how many examinees it will test; without a tracker, selectors apply no cap.
- A session that starts and is abandoned still counts in the denominator, which lowers every later
  rate a little: the tracker cannot tell it from one still running.
- The tracker's counts are not part of any session snapshot: saving a tracker across restarts is
  the application's job ({doc}`/live-sessions`).

## Alternatives Considered

- Dividing by the planned size of the run: it leaves a live deployment without a cap and makes the
  cap bear on examinees by their place in the run.
- Counting the session being tested in the denominator: it divides the administrations of an item
  the session can still receive by one examinee more than could have received it, which lowers the
  rates most for the early sessions (the second session would see the first's items at 1/2, not 1).
- Fixing each session's rates when it starts: the tracker would keep per-session state.
- Keeping exposure on the item bank: one run's exposure would leak into the next.
