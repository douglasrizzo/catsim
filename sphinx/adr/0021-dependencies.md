# ADR-0021: NumPy, SciPy and irtlib Are the Only Runtime Dependencies

## Status

Accepted 2026-11-25

## Context

catsim required NumPy, SciPy, tqdm and typing-extensions. tqdm drew the progress bar of a
simulation, which few runs show, and typing-extensions provided `override`, which the standard
library has from Python 3.12. matplotlib was already an optional extra (ADR-0022). An application
that embeds catsim's computations installs whatever catsim requires, and must reconcile each
required package's version range with its own; the tests cover each range at both ends, with the
newest versions and with the lowest ones allowed.

## Decision

- catsim's runtime dependencies are NumPy, SciPy and irtlib, the item response theory catsim builds
  on (ADR-0024), which itself requires only NumPy and SciPy.
- Plotting is the optional `plot` extra, with matplotlib (ADR-0022).
- Progress is reported through a callback: `simulate(..., progress=None)` takes a callable that is
  told how many sessions have finished and how many the run has (ADR-0015). catsim draws no progress
  bar itself.
- A new runtime dependency needs an ADR of its own.

## Consequences

- An application or a study environment that installs catsim gets two well-known libraries and
  irtlib, which adds no dependency of its own.
- A researcher who wants a progress bar passes one, for example a tqdm bar's `update`, in a line.
- An application has three version ranges to reconcile, and the tests cover NumPy's and SciPy's at
  both ends; irtlib's is one commit until its first release.
- Pandas, polars and file formats other than CSV stay out of catsim; its tables are plain NumPy
  columns that those libraries read (ADR-0016).

## Alternatives Considered

- tqdm as an optional extra: an extra to maintain for one bar that a callback already gives.
- Keeping tqdm required: every installation carries it for a feature most runs do not use.
- typing-extensions for `override` on Python 3.11: catsim requires 3.12, where `typing` has it
  (ADR-0024).
