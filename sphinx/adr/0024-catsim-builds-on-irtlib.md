# ADR-0024: catsim Builds on irtlib's Item Response Theory

## Status

Accepted 2026-11-27

## Context

catsim held the item response theory its strategies compute with: the response functions and the
quantities derived from them, the item bank with its CSV format and random banks, the quadrature
grid with its priors and posterior, and the mathematics of the ability estimators. Those are
meaningful for any test, adaptive or not, and calibrating items, which catsim leaves to other
software (ADR-0001), needs the same functions.

[irtlib](https://github.com/douglasrizzo/irtlib) is a package for the item response theory of the
dichotomous one- to four-parameter logistic models: the response functions, calibration, ability
estimation for many examinees at once, model fit and the item bank. Its scope ADR draws the line
between the two packages at the fixed-form test, and its ADR-0004 makes catsim build on it. It took
catsim's item response code with its reference cases and rewrote the tests. It requires Python 3.12,
and it has not been released.

## Decision

- catsim depends on irtlib for its item response theory, and keeps no copy of it: the response
  functions, the item bank and its model names, the array types, the quadrature grid, the priors and
  the posterior are irtlib's, and callers import them from irtlib. catsim does not re-export them.
- catsim's ability estimators are strategies over irtlib's estimators. catsim keeps what only an
  adaptive test needs: the step context, the estimate before any response, and the rules for an
  estimate that does not exist yet, such as Dodd's step toward the bank's hardest item.
- Quantities that only adaptive testing uses, such as information over an interval for a selector,
  are irtlib's too when they are functions of the model alone: one package validates each formula.
- catsim requires the Python versions irtlib supports, from 3.12.
- Until irtlib's first release, catsim depends on a commit of irtlib's GitHub repository, given as a
  direct reference in its metadata. catsim's next release waits for irtlib's, because PyPI does not
  accept a dependency given as a URL.
- A change to the item response theory that catsim needs, for speed or for a new quantity, is made
  in irtlib.

## Consequences

- Each formula has one implementation, validated in the package whose subject it is, with its
  parity cases against catR and mirt there.
- Code that builds a bank or computes a response probability imports from irtlib as well as from
  catsim, as `from irtlib import ItemBank`.
- catsim does not install on Python 3.11.
- catsim's estimators run at irtlib's speed for one examinee; a slowdown that matters to a study is
  measured and taken to irtlib.
- Until irtlib publishes its documentation, catsim's pages name irtlib's objects without linking
  them.
- An installation of catsim from GitHub fetches irtlib from GitHub too, at the pinned commit.

## Alternatives Considered

- Keeping catsim's copies beside irtlib's: two implementations drift, and each is validated twice.
- Waiting for irtlib's first release: the two copies would drift in the meantime, and catsim's
  development would keep code it is about to delete.
- Re-exporting irtlib's names from catsim: two import paths for one object, which ADR-0023's one
  public path per name rules out.
- Asking irtlib to support Python 3.11: irtlib's ADR-0002 sets 3.12, and its code uses the type
  statement and `typing.override`, which 3.11 lacks.
