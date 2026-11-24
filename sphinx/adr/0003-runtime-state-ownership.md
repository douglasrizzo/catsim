# ADR-0003: Keep Runtime State Outside Domain Objects

## Status

Accepted

## Context

Exposure tracking and per-session execution details were previously entangled with long-lived
objects in ways that made repeated runs and object reuse harder to reason about.

The exposure rates a selector checks its `r_max` cap against were the tracker's counts divided by a
number of sessions the caller passed in `RunContext.total_sessions`, the planned size of the run.
That caused two problems:

- Without `total_sessions`, the engine passed no rates at all, so an `r_max` cap did nothing and
  nothing said so. That is the natural setup for a live deployment, which does not know in advance
  how many examinees it will test.
- With it, early examinees saw every rate near zero and the cap bore only on late ones, so the
  precision of an examinee's test depended on their place in the run: the later the examinee, the
  less informative the items left below the cap.

catR's `simulateRespondents` divides by the examinees tested so far instead: after respondent `i`
it sets the rates to the counts divided by `i`, and respondent `i + 1` is tested against them, with
no restriction for the first respondent and none at all when `rmax` is 1.

## Decision

Treat calibrated item data as domain data and keep runtime state in dedicated execution models:

- `ItemBank` holds item parameters and related derived data
- `CatSessionState` holds per-examinee runtime state
- `ExposureTracker` (in `catsim.state`) and `SimulationResult` (in `catsim.simulation`) hold
  run-level exposure information

`ExposureTracker` owns the denominator of the exposure rates, which follow catR:

- The tracker counts sessions as well as administrations. `CatEngine.start_session` calls
  `tracker.start_session()` when the run context has a tracker, after the initializer has
  succeeded, and `apply_response` records each administered item. `RunContext` has no
  `total_sessions`, and `ExposureTracker.rates()` takes no argument.
- A session is selected against each item's administrations divided by the number of other
  sessions started so far. With sessions run one after another, as `SimulationRunner` runs them,
  that is catR's rate: the share of the examinees tested before the current one. The session being
  tested is not in the denominator. Counting it would divide the administrations of an item it can
  still receive by one examinee more than could have received it, which lowers the rates most for
  the early sessions: the second session would see every item of the first at 1/2 instead of 1. Its
  own administrations stay in the counts, but only for items it has already administered, which it
  is never offered again.
- The rates are computed when a session asks, not fixed when it starts, so a single tracker serves
  sessions that run side by side without keeping anything per session: each sees the others'
  administrations as they happen, over the other sessions started so far.
- The first session sees every rate at 0: no examinee has been tested before it.
- An `r_max` of 1 applies no cap, as in catR. An item that every earlier examinee received has a
  rate of exactly 1, and the selectors' strict `rate < r_max` test would otherwise pass it over.
- `SimulationResult.exposure_rates` divides the run's counts by all its sessions: the exposure at
  the end of the run, not the rates any session saw.

## Consequences

- Reusing an item bank across sessions or simulations is safer.
- `ItemBank` keeps its own copy of the parameters and makes it read-only, together with the column
  views its properties return, so the values derived and cached from them (`max_info_thetas`,
  `model`, a stratified selector's strata) cannot go stale.
- Exposure becomes a property of a run, not of the calibrated bank itself.
- Run metrics have a clearer ownership model and are easier to validate.
- `ItemBank.items` has exactly four columns, `[a, b, c, d]`. The fifth exposure-rate column that
  earlier versions kept for compatibility is gone, together with `ItemBank.exposure_rates`,
  `update_exposure_rate`, `reset_exposure_rates` and `reset`. An item matrix with more than four
  columns raises `ValueError` instead of having its extra column read as exposure rates.
- A context with a tracker is all exposure control needs, so a live deployment gets it without
  knowing how many examinees it will test. `ExposureTracker` is exported from `catsim` for that
  use. A context without a tracker gives selectors no rates, and they apply no cap.
- The cap holds from the second examinee on: that examinee sees every item of the first at a rate
  of 1. A capped run's results differ from earlier versions; runs with `r_max = 1` do not change.
- In a live deployment, a session that starts and is abandoned before its first item still counts
  in the denominator, which lowers every later rate a little; so does starting a session twice for
  one examinee. The tracker cannot tell such a session from one still running, so the count stays.
- The tracker's counts are not part of any session snapshot. A deployment that restarts with a new
  tracker applies no cap until new sessions have run. To keep the cap across restarts, the
  application saves `tracker.counts` and `tracker.n_sessions` (e.g. as a list and an int in JSON)
  and rebuilds the tracker with `ExposureTracker(n_items, counts=..., n_sessions=...)`, which checks
  that there is one non-negative integer count per item and keeps its own copy of them. Saving is
  the application's job: it decides when (after every administration, on shutdown) and where.
- `ExposureTracker.rates()` gives the rates a session in progress is selected against, which leave
  that session out; the run's own exposure is `SimulationResult.exposure_rates`.
