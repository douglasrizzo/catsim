# ADR-0002: Remove Legacy Simulator APIs

## Status

Accepted

## Context

During the refactor, keeping both the simulator-centric interfaces and the new engine-oriented
architecture would have left the codebase with two competing execution models and long-term
compatibility burden.

## Decision

Remove the legacy simulator-coupled interfaces from the final architecture, including the
simulator backreference pattern and compatibility-oriented execution helpers.

## Consequences

- The public API has one execution model instead of two.
- Internal components no longer need to support hidden simulator-mediated argument preparation.
- Some pre-refactor extension points are intentionally breaking changes.
- Documentation and tutorials must teach the engine/session/result model directly.
