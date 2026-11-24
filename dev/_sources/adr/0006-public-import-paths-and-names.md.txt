# ADR-0006: Public Import Paths and One Name per Concept

## Status

Accepted 2026-11-24

## Context

Nothing said which import path of a class was the public one. Docs and tests imported names from
their package or from the module file that defines them, and some public names, such as `LogPrior`
and the helpers for custom selectors, existed only in a module file. Some module names repeated
their package (`selection/selection.py`), and `selection/random.py` shadowed the standard library:
any process whose `sys.path[0]` was the package directory failed on `import tempfile`.

The same concept also had several names. The ability estimate was `est_theta`, `theta` and
`updated_theta`; the items given so far were `administered_items` and `administered_item_ids`,
while `administered_items` also named a parameter matrix; the responses were `response_vector` and
`responses`; the item model `itemtype`, `item_type` and `model`; the random source `seed` and
`rng`.

## Decision

- The public import paths are:
  - `catsim`, for the core types: the engine, the session state and step results, the runner and
    its result, the item bank, `StepContext`, `Estimate` and the exceptions;
  - `catsim.selection`, `catsim.estimation`, `catsim.initialization`, `catsim.stopping` and
    `catsim.exposure`, for the strategies and the exposure controls, their base classes and the
    helpers documented for custom strategies;
  - `catsim.irt`, `catsim.metrics`, `catsim.plot` and `catsim.typing`, modules of functions and
    type aliases.
- Every other module is internal: the module files of the subpackages and the modules that define
  what `catsim` exports (`catsim.engine`, `catsim.state`, ...). Code outside catsim imports each
  name from its public path; only a test of a private helper imports a module file. A package's
  `__all__` lists every public class, function and alias its modules define, and nothing else.
- A module is named after what it holds (`selection/information.py`, `selection/randomized.py`),
  never after its package or a standard-library module.
- One name per concept:
  - `theta` for an ability or ability estimate passed to a function or held by a `StepContext`;
    `current_theta` for a session's latest estimate;
  - `administered_items` for the indices of the items given so far, in order;
  - `items` for an item parameter matrix, one row per item, and `indices` for other lists of item
    indices into a bank; `LinearSelector(items)`, the list of items it administers, is the one
    exception;
  - `responses` for the responses to the administered items, in the same order;
  - `n_items` for a number of items, `model` for the logistic model, and
    `rng: int | numpy.random.Generator | None` for the random source (ADR-0010);
  - `test_size` for the test length a selector plans its schedule for, and `max_items` for the
    length at which `MaxLengthStopper` ends a test. They are different settings: a test may stop
    before or after the length its selector planned.

## Consequences

- An object keeps its import path when its module is split, merged or renamed. Its `__module__`
  still names the defining module, so a pickle of a catsim object breaks when its class moves.
- No alias keeps a retired name or module path working.
- Tests check that each package's `__all__` matches the public definitions of its modules and that
  no public signature or attribute uses a retired name.

## Alternatives Considered

- Public module files: every split or rename of a module would break user code.
- Aliases for the old names: two names for one concept are what this decision removes.
- `est_theta` for `theta`: the `irt` functions take any ability, not only an estimate.
- One length setting for selectors and stoppers: it would tie a selector's schedule to a length
  criterion that a precision-based test lacks.
- Underscore-prefixed internal modules (`_information.py`): noisier `__module__` names and
  tracebacks for what the rule and `__all__` already state.
