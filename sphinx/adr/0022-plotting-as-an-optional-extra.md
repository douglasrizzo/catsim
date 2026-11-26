# ADR-0022: Plotting Is an Optional Module With One Calling Convention

## Status

Accepted 2026-11-26

## Context

`catsim.plot` was catsim's only use of matplotlib, yet matplotlib was once a required dependency, so
every installation carried it and its import cost, including those that never draw a figure. The
plot functions also had no common calling convention: some took an item's parameters as separate
scalars, others put `ax` and `title` first and took either a result or raw arrays through mutually
exclusive keywords, and three took a `figsize` that applied only when they created the figure. They
also set a grid and an 18-point title of their own, over whatever style the caller had chosen.

## Decision

- matplotlib is an optional dependency, installed with the `catsim[plot]` extra (ADR-0021).
- `catsim.plot` is the only module that imports matplotlib. Importing `catsim`, or any other public
  module, never loads it, and import-linter contracts in `pyproject.toml` check both rules.
- Importing `catsim.plot` without matplotlib raises an `ImportError` that gives the install command.
- A plot function takes its data first, as plain catsim objects (an item bank, a session, a
  result's exposure rates), and its options as keyword-only arguments:

  ```python
  session_progress(session, item_bank, *, ax=None, title=None) -> Axes
  ```

- Every plot function draws on the axes the caller passes as `ax`, or on the axes of a new figure,
  and returns the axes it drew on. None takes a figure size.
- A plot follows the caller's matplotlib style: the style sets the fonts, colours, title size and
  grid, and the functions set none of their own. A colour a function takes from the style's cycle
  is fixed when it is called. A second y scale, on the right, draws its spine and no grid.
- Plots label items by name (ADR-0004).

## Consequences

- An installation without the extra fails at `import catsim.plot`, with the fix in the message.
- A figure belongs in `catsim.plot`: no other module can grow a plotting helper.
- A session plots the same way whether it comes from a simulation result or from an application,
  since the function reads the session.
- Figures with several panels are laid out by the caller with matplotlib's tools, and each plot
  function fills one axes.
- A caller changes how plots look with matplotlib's styles alone. Without one, plots have
  matplotlib's default look, which has no grid.
- The test suite, the documentation build and the notebooks need matplotlib, so the development and
  documentation dependency groups include it.

## Alternatives Considered

- matplotlib required: every installation pays for figures it never draws.
- Importing matplotlib inside each function: the error would surface at the first call, far from the
  missing installation.
- A separate plotting package: a second distribution to version and release for a few functions.
- Functions that take a result and a session index: they could not plot a session from elsewhere.
- A `figsize` argument: it does nothing when the caller passes `ax`, and `plt.subplots` provides it.
- Functions that set their own grid and title size: they override the caller's style, and
  `ax.grid()` without arguments toggles the grid, so it removes the grid of a style that has one.
