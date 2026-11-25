# ADR-0020: catR Is the Reference Implementation, Checked by Parity Tests

## Status

Accepted 2026-11-25

## Context

catsim implements many techniques that catR implements: the logistic models, the ML, Bayes modal,
EAP and weighted likelihood estimators, the MFI, bOpt, MLWI, MEI, KL, progressive, proportional and
randomesque selection rules, and its length, precision and classification stopping rules. The
technique pages and several decisions claimed agreement with catR, but no test compared the two, so
a claim could drift from the code unnoticed. catsim also departs from catR on purpose in places,
such as the default acceleration of the progressive schedule (ADR-0019) and the default EAP grid,
121 nodes over $[-6, 6]$ against catR's 33 over $[-4, 4]$. Neither catR nor mirtCAT compares itself
with another package.

## Decision

- catR is catsim's reference implementation. Every catsim technique with a catR counterpart agrees
  with it, as tested by parity tests in three parts:
  - R scripts in the repository run catR on fixed cases and write their outputs, with the R and catR
    versions, to files in the repository;
  - a pytest module compares catsim's outputs with those files, in every test run, with no R
    installed;
  - one documented command regenerates the files, when the cases change or the pinned catR version
    moves. It needs R and catR.
- Agreement is checked at three levels:
  1. single computations (probabilities, information, each estimator on fixed response patterns,
     each selector's scores and choice in a fixed context, stopping decisions, initial values):
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
- mirtCAT is consulted when a technique has no catR counterpart, not tested against.

## Alternatives Considered

- Running catR in CI: R on every run, slower and less reliable, for a package that changes rarely.
- Comparisons made by hand or by an agent when a page is written: nothing durable, so claims drift.
- catR's defaults everywhere: catsim's choices, such as its wider ability scale (ADR-0018), have
  reasons of their own.
- mirtCAT as a second reference: two references can disagree, and mirtCAT's estimation comes from
  mirt.
