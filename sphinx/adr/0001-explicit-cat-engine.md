# ADR-0001: Adopt Explicit CAT Engine Architecture

## Status

Accepted

## Context

The previous architecture centered CAT execution around a simulator object. That made batch
simulation the dominant workflow and forced manual or third-party usage through interfaces that
were indirectly coupled to simulator-owned state.

## Decision

Adopt an explicit execution model built around:

- `CatEngine` for stepwise CAT execution
- `CatSessionState` for per-session runtime state
- `SimulationRunner` for batch execution on top of the same engine
- `SimulationResult` for aggregate outputs

## Consequences

- Manual CAT execution becomes a first-class use case.
- Simulation reuses the same execution path instead of owning a separate lifecycle.
- State transitions become easier to test because the engine boundary is explicit.
- The package is better described as a CAT toolkit than as a simulator-only library.
