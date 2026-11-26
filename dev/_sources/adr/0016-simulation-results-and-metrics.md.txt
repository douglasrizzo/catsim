# ADR-0016: A Run's Result Gives Tables and Metrics With Their Monte Carlo Error

## Status

Accepted 2026-11-25

## Context

A simulation study reports how accurate, long and exposure-balanced a design's tests are, overall
and across the ability scale, and analyses the sessions in a statistics package. Its usual measures
are the bias, RMSE, test length, standard errors and exposure, overall and conditional on deciles of
the true ability. Morris, White and Crowther (2019) recommend reporting each performance
measure of a simulation study with its Monte Carlo standard error.

`SimulationResult` gave its sessions and properties such as `final_thetas`, `rmse` and
`exposure_rates`, with no conditional metrics, no Monte Carlo error and no tabular form.

## Decision

- `SimulationResult` is immutable. It holds the sessions in examinee order, the true abilities when
  known, the exposure and proposal counts, the number of parts (ADR-0015) and the run's duration.
- It gives two tables of its sessions, each a dict of equal-length NumPy columns:
  - one row per session and step: the session, the position, the item's index and name
    (ADR-0004), the response, the estimate and its standard error, and the true ability;
  - one row per session: its length, final estimate and standard error, stop reason and true ability.
- Its metrics are the bias, the mean squared error and RMSE, the test length, the standard error,
  the exposure rates and the overlap rate, each reported with its Monte Carlo standard error, overall
  and by decile of the true ability. A table gives them by decile, and another at each distinct true
  ability, for designs that give many examinees each of a few abilities, which deciles group unevenly.
- A run is one condition. catsim has no layer for grids of conditions and replications: a study
  loops over them and gathers the results.

## Consequences

- `pandas.DataFrame(result.steps())` and polars accept the tables as they are, so neither becomes a
  dependency, and CSV writers take them too.
- A difference between two designs can be judged against its Monte Carlo error.
- With exposure feedback, sessions are not independent, so the Monte Carlo standard errors, which
  treat them as independent, are approximations; replications give the exact spread.
- Metrics that need the true abilities raise for recorded examinees whose abilities are unknown.

## Alternatives Considered

- pandas data frames: a dependency for a format pandas reads from plain columns.
- Point estimates only: a difference between designs cannot be told from noise.
- A study layer of conditions, replications and aggregation: one design among many a study may need,
  which the study's own loop expresses directly; to reconsider if research use shows the loop is
  where the work is.
