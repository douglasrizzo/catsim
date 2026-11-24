# ADR-0015: Plotting as an Optional Extra

## Status

Accepted 2026-11-24

## Context

`catsim.plot` was catsim's only use of matplotlib, yet matplotlib was a required dependency. Every
install carried it and its dependency tree, including services that run live sessions and never
draw a figure, and importing it is slow.

The plot functions also had no common calling convention. `item_curve` took an item's four
parameters as separate scalars, while `item_exposure` and `test_progress` put `ax` and `title`
first and took either a `SimulationResult` or raw arrays through mutually exclusive keywords.
`test_progress` could not plot a session run directly with `CatEngine`, and three functions took a
`figsize` that applied only when they created the figure themselves.

## Decision

- matplotlib is an optional dependency, installed with the `catsim[plot]` extra.
- `catsim.plot` is the only module that imports matplotlib. Importing `catsim`, or any public module
  other than `catsim.plot`, never loads it. Two import-linter contracts in `pyproject.toml` check
  both rules.
- Importing `catsim.plot` without matplotlib raises an `ImportError` that gives the install command
  for the extra.
- A plot function takes its data first, as plain catsim objects (an item's parameter row, the
  exposure rates, a session), and its options as keyword-only arguments:

  ```python
  session_progress(session, item_bank, *, ax=None, title=None) -> Axes
  ```

- Every plot function draws on the axes the caller passes as `ax`, or on the axes of a new figure,
  and returns the axes it drew on. None takes a figure size: a caller who wants one creates the
  axes with `plt.subplots(figsize=...)` and passes them.

## Consequences

- A plain install of catsim needs numpy, scipy, tqdm and typing-extensions only. Users who plot
  install the extra; code that imports `catsim.plot` without it fails at the import, with the fix in
  the message.
- Domain, strategy and runtime modules cannot grow a plotting helper: a figure belongs in
  `catsim.plot`.
- A session plots the same way whether it comes from a simulation (`result.sessions[j]`) or from a
  live loop, because the function reads the session, not the result that holds it.
- Figures with several panels are laid out by the caller with matplotlib's own tools, and each plot
  function fills one axes. Callers that relied on a function creating a sized figure create it
  themselves.
- The test suite, the documentation build and the notebooks need matplotlib, so the development and
  documentation dependency groups include it, and the CI job that tests the built wheel installs
  the wheel with the extra.

## Alternatives Considered

- Keep matplotlib required: every deployment pays its installation and import cost for figures it
  never draws.
- Import matplotlib inside each plot function: `import catsim.plot` would succeed and the error
  would surface at the first call, far from the missing install; one guarded import at the top of
  the module fails at once.
- A separate plotting package: a second distribution to version and release for three functions.
- Plot functions that take a `SimulationResult` and a session index: they cannot plot a live
  session, and they tie the plots to the runner's result type.
- A `figsize` argument on every function: it does nothing when the caller passes `ax`, and
  `plt.subplots` already provides it.
