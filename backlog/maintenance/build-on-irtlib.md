---
title: Build catsim on irtlib's item response theory and remove catsim's copies
kind: maintenance
status: ready
priority: 3
effort: L
blocked_by: []
raised_by: [irtlib ADR-0004 and the maintainer on 2026-11-27]
---

# Build catsim on irtlib's item response theory and remove catsim's copies

## Problem

The response functions, the item bank, the quadrature grid with its priors and posterior, and the
ability estimators now exist twice: in catsim, and in
[irtlib](https://github.com/douglasrizzo/irtlib), the item response theory package whose ADR-0001
gives it everything meaningful for a fixed-form test and whose ADR-0004 makes catsim build on it.
Two copies drift, and each formula is validated twice. The copies in catsim are to go, and catsim is
to import them from irtlib.

## What irtlib has now

At irtlib's `dev`, `ae6a45b` (2026-11-27):

| catsim | irtlib | Difference |
|---|---|---|
| `catsim.irt`: `icc`, `information`, `total_information`, `var`, `see`, `confidence_interval`, `theta_to_scale`, `scale_to_theta`, `reliability`, `max_info_theta`, `info_integral`, `log_response_probabilities`, the `THETA_*` bounds | `irtlib.irt` | same signatures |
| `catsim.irt.log_likelihood`, `log_likelihood_function` | `irtlib.irt` | many examinees: a float response matrix with NaN for a missing response, one row per examinee |
| `catsim.irt.validate_response_vector` | none | a session checks its responses where they enter (ADR-0009) |
| `catsim.ItemBank`, `catsim.IrtModel` | `irtlib.ItemBank`, `irtlib.IrtModel` | same API, CSV format and random banks |
| `catsim.typing` | `irtlib.typing` | adds `BoolArray` |
| `catsim.estimation`: `QuadratureGrid`, `LogPrior`, `normal_log_prior`, `uniform_log_prior`, `posterior`, `posterior_mean`, `posterior_variance` | `irtlib.quadrature` | `posterior` and its moments take one row per examinee |
| the mathematics of `EAPEstimator`, `MAPEstimator`, `NumericalSearchEstimator`, `WarmLikelihoodEstimator` | `irtlib.ability`: `expected_a_posteriori`, `maximum_a_posteriori`, `maximum_likelihood`, `weighted_likelihood` | many examinees; ML searches 201 points before refining, where catsim runs Brent's method alone |

irtlib requires Python 3.12 (its ADR-0002), and has neither a release, a license nor a documentation
site yet. Its benchmarks time its estimators at 1.3 to 2.4 times catsim's for one examinee
(irtlib's `estimation/faster-ability-estimation` and `irt/one-examinee-speed` cards); that work is in
progress.

## Decisions (maintainer, 2026-11-27)

1. catsim requires Python 3.12 or later.
2. catsim adopts irtlib now, on `dev`. Its next release waits for irtlib's first release and license,
   since PyPI refuses a dependency given as a URL.
3. The dependency is a direct reference in `[project].dependencies`,
   `irtlib @ git+https://github.com/douglasrizzo/irtlib@<sha>`, which pip and uv both install. The
   commit is irtlib's `dev` when each PR opens, bumped when irtlib's speed work lands.
4. No re-exports: the names in the table leave catsim, and callers import them from irtlib
   (ADR-0023).
5. catsim's four estimators become strategies over `irtlib.ability`, called with the session's
   responses as one row. catsim keeps what only an adaptive test needs: the `StepContext` contract,
   the estimate before any response, and Dodd's step while every response is the same. Their names
   stay. If they stay too slow and irtlib has no faster option, the maintainer and the agent
   propose a change to irtlib.
6. The documentation writes irtlib's names as literals, with no cross-reference, and Sphinx ignores
   them in signatures, until irtlib publishes its documentation (`maintenance/irtlib-docs-links`).
7. A source in `sphinx/references.bib` or on the local paper shelf that nothing cites after the move
   (in `src/`, `sphinx/`, the notebooks, the README or a backlog card) is deleted; a PDF only after
   the maintainer sees the list, and one that irtlib cites but lacks moves to irtlib's shelf.
8. The item response plots of `catsim.plot` stay until irtlib's `discovery/irt-plots` decision.

## Proposal

A stack of four pull requests, in review order. Each updates the docstrings, Sphinx pages, notebooks,
README, tests and cards that its change touches, and deletes the `references.bib` entries that it
leaves uncited.

1. **`build!: require Python 3.12 and depend on irtlib`.** `requires-python = ">=3.12"`; the CI
   matrix, the lowest-direct job and the wheel check start at 3.12; `catsim.compat` and its test go,
   and `override` comes from `typing`. The irtlib dependency, ADR-0024 (catsim builds on irtlib),
   ADR-0021 (dependencies) and `AGENTS.md`'s summary and dependency rule. The release card gets the
   step that replaces the URL by a version range.
2. **`refactor!: take the response functions, the item bank and the array types from irtlib`.**
   Deletes `irt.py`, `item_bank.py` and `typing.py`, the API pages `irt`, `item_bank` and `typing`,
   and their tests (`test_irt*`, `test_item_bank*`, catR's `TestItemResponseFunctions`, and the `irt`
   and `item_bank` reference outputs). Every module, test, guide and notebook imports from irtlib.
   ADR-0002, 0003, 0004 and 0018 keep catsim's part and point to irtlib's ADRs for the rest;
   ADR-0005 (the CSV format) is superseded by irtlib's ADR-0009. The bank-generation card moves to
   irtlib; the import-time card keeps catsim's two `scipy.stats` imports, and irtlib gets a card for
   its own.
3. **`refactor!: take the quadrature grid, the priors and the posterior from irtlib`.** Deletes them
   from `estimation/bayesian.py` with their tests and the "Bayesian infrastructure" technique page;
   `EAPEstimator` and `MAPEstimator` take irtlib's grid and priors.
4. **`refactor!: estimate ability with irtlib's estimators`.** The four estimators call
   `irtlib.ability`. Records the times of a simulation before and after, and regenerates the
   estimation reference outputs where ML's grid search changes an estimate. The WLE standard-error
   card moves to irtlib, whose estimator computes the error. Deletes this card.

In irtlib, a separate pull request updates ADR-0004 and `AGENTS.md`, which say catsim stays as it is
until irtlib's first release.

## Tests

- The suite, the docstring examples, the notebooks and the catR parity of the four estimator
  strategies pass on Python 3.12 to 3.15.
- `uv run lint-imports` passes with the deleted modules out of the layers contract.
- `uv run sphinx-build -W` builds the documentation with no unresolved reference.
- A test that `import catsim` provides none of the removed names fails if a re-export comes back.
