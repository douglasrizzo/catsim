# ADR-0021: NumPy and SciPy Are the Only Runtime Dependencies

## Status

Proposed 2026-11-25

## Context

catsim required NumPy, SciPy, tqdm and typing-extensions. tqdm drew the progress bar of a
simulation, which few runs show, and typing-extensions provided `override`, which the standard
library has from Python 3.12. matplotlib was already an optional extra (ADR-0022). An application
that embeds catsim's computations installs whatever catsim requires, and must reconcile each
required package's version range with its own; the tests cover each range at both ends, with the
newest versions and with the lowest ones allowed. catR depends on base R only.

## Decision

- catsim's runtime dependencies are NumPy and SciPy, and typing-extensions on Python versions below
  3.12 only.
- Plotting is the optional `plot` extra, with matplotlib (ADR-0022).
- Progress is reported through a callback: `simulate(..., progress=None)` takes a callable that is
  told how many sessions have finished and how many the run has (ADR-0015). catsim draws no progress
  bar itself.
- A new runtime dependency needs an ADR of its own.

## Consequences

- An application or a study environment that installs catsim gets two well-known libraries.
- A researcher who wants a progress bar passes one, for example a tqdm bar's `update`, in a line.
- On Python 3.12 and later, catsim installs nothing but NumPy and SciPy, so an application has two
  version ranges to reconcile, and the tests two to cover at both ends.
- Pandas, polars and file formats other than CSV stay out of catsim; its tables are plain NumPy
  columns that those libraries read (ADR-0016).

## Alternatives Considered

- tqdm as an optional extra: an extra to maintain for one bar that a callback already gives.
- Keeping tqdm required: every installation carries it for a feature most runs do not use.
- Dropping typing-extensions on 3.11 too: `override` would have to be redefined in catsim.
