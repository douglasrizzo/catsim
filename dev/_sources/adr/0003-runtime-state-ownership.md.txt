# ADR-0003: Keep Runtime State Outside Domain Objects

## Status

Accepted

## Context

Exposure tracking and per-session execution details were previously entangled with long-lived
objects in ways that made repeated runs and object reuse harder to reason about.

## Decision

Treat calibrated item data as domain data and keep runtime state in dedicated execution models:

- `ItemBank` holds item parameters and related derived data
- `CatSessionState` holds per-examinee runtime state
- `ExposureTracker` and `SimulationResult` hold run-level exposure information

## Consequences

- Reusing an item bank across sessions or simulations is safer.
- Exposure becomes a property of a run, not of the calibrated bank itself.
- Run metrics have a clearer ownership model and are easier to validate.
- `ItemBank.items` has exactly four columns, `[a, b, c, d]`. The fifth exposure-rate column that
  earlier versions kept for compatibility is gone, together with `ItemBank.exposure_rates`,
  `update_exposure_rate`, `reset_exposure_rates` and `reset`. An item matrix with more than four
  columns raises `ValueError` instead of having its extra column read as exposure rates.
