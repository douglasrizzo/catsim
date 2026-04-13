# ADR-0004: Use Explicit Strategy Contracts

## Status

Accepted

## Context

The refactor established an explicit engine boundary, but strategy interfaces still accepted
engine-managed data through generic `**kwargs`. That weakened the contract between the engine
and concrete initializers, selectors, and stoppers.

## Decision

Use explicit typed parameters for engine-managed strategy inputs:

- initializers receive an `ItemBank` and a `numpy.random.Generator`
- selectors receive explicit session inputs such as administered items, theta, RNG, and
  exposure-rate snapshots
- stoppers receive explicit item-bank, administered-item, and theta inputs

## Consequences

- Public contracts are clearer for users implementing custom strategies.
- Type checking can enforce the engine boundary more effectively.
- Some permissive legacy call patterns are removed intentionally.
