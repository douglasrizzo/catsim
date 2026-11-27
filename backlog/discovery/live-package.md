---
title: Build a live-testing package on catsim's public API
kind: research
status: needs-decision
priority: 2
effort: XL
blocked_by: []
raised_by: []
---

# Build a live-testing package on catsim's public API

## Goal

Check that catsim can be the computing core of live tests, the boundary ADR-0001 draws. Build a
package with no frontend, using catsim's public API only. The package owns every serving concern:

- sessions that survive across requests and restarts;
- the item on screen;
- exposure shared across sessions;
- persistence, concurrency and audits.

Its friction changes catsim only where the change meets catsim's own requirements. Anything specific
to serving stays in the live package. The "Embedding catsim" guide is the starting point.

## Decision needed

Where the package lives: a new repository, or a folder here that is never published with catsim.
The maintainer chooses before work starts.

## Done when

The package runs a test end to end, through a minimal request loop with storage and restarts, on
catsim's public API alone. Its findings are triaged into cards.
