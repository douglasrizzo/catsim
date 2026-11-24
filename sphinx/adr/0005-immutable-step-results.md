# ADR-0005: Make Step Results Immutable Snapshots

## Status

Accepted 2026-11-23

## Context

A step result returned the live, mutable session object. As the session went on, every earlier
result changed with it: a list of step results showed the final state as many times as there were
steps, so step-level inspection, tracing and the tutorials that print a session step by step
reported the wrong thing without any error.

## Decision

`CatStepResult.session` is a `CatSessionSnapshot`: an immutable, hashable copy of the session as
the step left it, with the session's lists as tuples. `CatSessionState.snapshot()` builds one. The
engine mutates `CatSessionState` as the test runs; every object it hands out about a past step is a
snapshot. The response provider receives a snapshot too, so it cannot change the session it
answers for.

## Consequences

- A stored step result stays as the step left it, however many steps follow, so step-level
  debugging, audit trails and tutorials can keep and compare results.
- A snapshot can be a dictionary key or a set member, and two snapshots of the same state compare
  equal.
- Each step copies the session's items, responses and estimates, a cost proportional to the test
  length and small next to an estimate.
- Code that wants the current state reads the `CatSessionState` it passed to the engine, not a step
  result.

## Alternatives Considered

- Returning the live session and documenting that it changes: the mistake stays silent, and every
  caller that keeps results must copy them itself.
- Copying the session into another mutable `CatSessionState`: a caller could still change a result
  after the fact, and it could not be hashed.
