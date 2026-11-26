# ADR-0023: Public Import Paths and One Name per Concept

## Status

Accepted 2026-11-25

## Context

Nothing once said which import path of a class was public: documents and tests imported names from
their package or from the module file that defined them, and one module shadowed the standard
library's `random`. The same concept also had several names: the estimate was `est_theta`, `theta`
and `updated_theta`, the responses `response_vector` and `responses`, the random source `seed` and
`rng`. Users had to learn each spelling, and a module split broke their imports.

## Decision

- The public import paths are:
  - `catsim`, for the core types and functions: `CatEngine`, `Session`, `Selection`, `StopReason`,
    `StepContext`, `Estimate`, `ItemBank`, `IrtModel`, `simulate`, `SimulationResult` and the
    exceptions;
  - `catsim.selection`, `catsim.estimation`, `catsim.initialization`, `catsim.stopping` and
    `catsim.exposure`, for the strategies, their base classes and the helpers documented for custom
    strategies;
  - `catsim.irt`, `catsim.metrics`, `catsim.plot` and `catsim.typing`, modules of functions and
    type aliases, with the constants and types their functions take or return, such as
    `irt.THETA_MIN_EXTENDED`, `plot.PlotType` and `metrics.Metric`.
- Every other module is internal. Code outside catsim imports each name from its public path. A
  package's `__all__` lists every public name its modules define, and nothing else.
- A module is named after what it holds, never after its package or a standard-library module.
- One name per concept:
  - `theta` for an ability or an estimate, and `current_theta` for a session's latest estimate;
  - `item` for one item's index, `name` for one item's name, `indices` for a list of item indices
    and `names` for a list of item names (ADR-0004);
  - `administered_items` for the indices of the items given so far, in order, and `responses` for
    their responses, in the same order;
  - `items` for an item parameter matrix, one row per item;
  - `n_items` for a number of items, `model` for the logistic model and
    `rng: int | numpy.random.Generator | None` for the random source (ADR-0013);
  - `test_size` for the length a selector plans its schedule for, and `max_items` for the length at
    which `MaxLengthStopper` ends a test. A test may stop before or after its planned length.

## Consequences

- An object keeps its import path when its module is split, merged or renamed. Its `__module__`
  still names the defining module, so a pickle of a catsim object breaks when its class moves.
- No alias keeps a retired name or path working.
- Tests check that each package's `__all__` matches the public definitions of its modules and that
  no public signature uses a retired name.

## Alternatives Considered

- Public module files: every split or rename of a module would break user code.
- Aliases for old names: two names for one concept are what this decision removes.
- `item_id` for an item's index: with item names, an "identifier" reads as the name.
- One length setting for selectors and stoppers: it would tie a selector's schedule to a length
  criterion that a precision-based test lacks.
- Underscore-prefixed internal modules: noisier `__module__` names and tracebacks for what the rule
  and `__all__` already state.
