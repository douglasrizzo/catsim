# ADR-0001: catsim Computes and Evaluates Adaptive Tests and Does Not Serve Them

## Status

Accepted 2026-11-25

## Context

catsim promised two things: the computations and simulation studies of computerized adaptive
testing (CAT), and the running of live tests in applications. A live session was a first-class use
of the engine, so catsim's objects had to survive across requests and restarts, keep the item on
screen, share exposure counts between sessions run side by side, and support audit replay. A review
of the API found that this promise drove most of its complexity, and that every result
it found silently wrong came from catsim's state outliving one run.

The two can live apart: the Concerto testing platform, for one, serves live tests and leaves their
computations to the catR package.

## Decision

catsim is a library for computing and evaluating computerized adaptive tests:

- the IRT quantities of its models;
- the CAT strategies: initialization, item selection, ability estimation, stopping and exposure
  control, built in or written by method authors;
- simulation studies that run and compare them.

A live testing application can use catsim's computations as its core. catsim does not run, store or serve live tests: storing sessions, the item on screen, concurrency,
sharing exposure counts across sessions, persistence and audits belong to the application
(ADR-0006).

catsim does not calibrate items, and it has no graphical or web interface.

## Consequences

- Its readers are, in order: researchers who run and compare simulation studies, method authors who
  write strategies, and integrators who embed the computations in an application. Integrators get
  one page on the boundary, with no serving recipes (`sphinx/AGENTS.md`).
- Every design question is settled for the computations and the studies. A need that only serving
  has, such as resuming a test after a restart, is the application's, or a separate package's.
- A separate live package, built on catsim's public API only, checks that the boundary holds.
- Item parameters are an input, from whatever software calibrated them.

## Alternatives Considered

- Serving and computing in one package: the serving concerns shape every object,
  and they are what made catsim's results silently wrong.
- Computation only, with no simulation studies: comparing strategies is what catsim's first
  audience does, and simulation is where catsim's computations are exercised together.
- Calibration in catsim: a different field of estimation, with its own literature and software,
  that a CAT library can take as input.
