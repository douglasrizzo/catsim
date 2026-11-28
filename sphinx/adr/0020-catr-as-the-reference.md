# ADR-0020: Parity Tests Check catsim Against catR

## Status

Accepted 2026-11-25

## Context

An independent implementation of the same technique is a strong check that catsim computes it
correctly. catR, a published and widely used R package, implements many of catsim's techniques. The
technique pages claimed agreement with it, but no test compared the two, so a claim could drift from
the code unnoticed. catsim also departs from catR on purpose in places, such as the default
acceleration of the progressive schedule (ADR-0019).

## Decision

- catR is catsim's reference implementation: every catsim technique with a catR counterpart is
  checked against it by parity tests in three parts:
  - R scripts in the repository run catR on fixed cases and write their outputs, with the R and catR
    versions, to files in the repository;
  - a pytest module compares catsim's outputs with those files, in every test run, with no R
    installed;
  - one documented command regenerates the files, when the cases change or the pinned catR version
    moves. It needs R and catR.
- Agreement is checked at three levels:
  1. single computations (each estimator on fixed response patterns, each selector's scores and
     choice in a fixed context, stopping decisions, initial values; the probabilities and the
     information are irtlib's, and its parity tests check them against catR, ADR-0024):
     closed forms to a relative $10^{-10}$, iterative estimators within the sum of the two
     packages' tolerances, since their errors can point in opposite directions;
  2. whole tests on recorded responses, with no randomization: the same items in the same order,
     and the estimates as in level 1;
  3. simulations with generated responses, for the randomized selectors: bias, RMSE, mean test
     length, exposure and overlap within Monte Carlo error, since R and NumPy draw different
     numbers.
- catsim keeps its own defaults. Every departure from catR either has a setting that reproduces
  catR, which the parity tests use, or is stated on the technique page with its reason and kept in
  the parity tests as a strict expected failure with that reason, so a departure that closes fails
  the run until it is removed from the list.

## Consequences

- A claim of agreement on a technique page is backed by a test, and a change that breaks agreement
  fails it.
- Most cases are level 2, the strongest check and the cheapest, since recorded responses need no
  shared random numbers.
- A new technique with a catR counterpart comes with its parity cases.
- The parity tests follow one catR version; moving to another is a deliberate change of the fixed
  outputs.

## Alternatives Considered

- Running catR in CI: R on every run, slower and less reliable, for a package that changes rarely.
- Comparisons made by hand or by an agent when a page is written: nothing durable, so claims drift.
- catR's defaults everywhere: catsim's choices, such as its wider ability scale (ADR-0018), have
  reasons of their own.
- A second reference package: two references can disagree.
