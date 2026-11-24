# ADR-0001: Adopt Explicit CAT Engine Architecture

## Status

Accepted

## Context

catsim's CAT execution was centred on a simulator object. Batch simulation was the workflow the
design served: a live test, run one response at a time by an application, had to go through
interfaces coupled to state the simulator owned, and strategies reached that state through a
reference back to the simulator rather than through their arguments. No component could be used or
tested without a simulator built around it.

## Decision

catsim executes a CAT through four explicit objects:

- `CatEngine` runs one session step by step. It holds the item bank and the strategies, and each
  of its methods takes the session's state as an argument.
- `CatSessionState` holds one session's runtime state.
- `SimulationRunner` runs a batch of sessions through the same engine.
- `SimulationResult` holds the batch's aggregate outputs.

This is the only execution model. No strategy holds a reference to the engine or the runner: each
receives what it needs as arguments.

## Consequences

- A live CAT session is a first-class use: an application drives the engine one item at a time,
  through the same methods the runner calls.
- Simulation reuses the engine's execution path instead of owning a separate lifecycle, so a
  simulated session and a live one behave alike.
- Each state transition can be tested through the engine alone, without a batch around it.
- The documentation and tutorials teach the engine, session and result objects directly, and
  catsim is a CAT toolkit of which simulation is one use.
- An extension point that relied on the simulator has no counterpart: a strategy gets its inputs
  from its arguments (the strategy contract, ADR-0004).
- The import-linter contract in `pyproject.toml` keeps the runner above the engine and the engine
  above the strategies, and keeps the strategy families from importing one another, with two
  exceptions: `MEISelector` and `MLWISelector` import the estimators, for their look-ahead estimates
  and their posterior.

## Alternatives Considered

- Keep the simulator at the centre and add a step-by-step mode to it: a live session would still
  depend on simulator-owned state, and batch concerns would reach every session.
- Keep the simulator-centric interfaces beside the engine, as a compatibility layer: the package
  would have two competing execution models, each with its own lifecycle, and a lasting
  compatibility burden.
