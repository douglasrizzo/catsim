# ADR-0008: Only Derived Data Is Cached, and History Caches Are Explicit

## Status

Accepted 2026-11-25

## Context

Steps are computed from explicit inputs (ADR-0006), which raises the question of what may be kept
between calls to save work. catsim already caches some values: the item bank's points of maximum
information and its model, the stratified selectors' strata for a bank, the KL selector's
Gauss-Legendre rules, and the EAP estimator's grid and prior. Each depends only on the bank and on a
strategy's settings, which never change.

Other savings depend on a session's history. The EAP posterior, about 40% of a step's time in a
simulation, is rebuilt from every answered item at each step, where adding the newest item's
log-likelihood to the previous one would do. The legacy design reached such state through a
reference from each strategy back to the simulator that called it, so a strategy's result depended
on who called it.

## Decision

catsim distinguishes three kinds of data kept between calls:

- **State** is what happened: the items given, the responses, the estimates and the exposure
  counts. Its owner, the caller, passes it in at every call (ADR-0006). catsim keeps none.
- **Derived data** depends only on inputs that never change: the item bank and a strategy's
  settings. It may be cached by the object it derives from, lazily or when that object is built,
  since losing it costs time and never changes a result. Data derived from a strategy's settings and
  a bank is cached by the strategy, keyed by the bank.
- **History caches** depend on a session's history and only save work, such as a running
  log-likelihood. One is allowed only as an optional, explicit input, passed down by the caller (the
  simulation runner or an application) and returned by the computation that updates it. A
  computation without it recomputes from the explicit inputs and gives the same result, up to
  floating-point rounding.

A computation never looks for a cache in the object that called it.

## Consequences

- A caller that keeps no cache, such as an application that stores only the session, gets the same
  results as the runner.
- Every cache is either immutable once built, so sharing it between threads and processes is safe,
  or travels with the call that uses it.
- A history cache adds an optional field to an input type and to a result type, so it is introduced
  with an ADR of its own that names the computation and the saving.
- Rounding can differ between a cached and a recomputed value, so exact ties between items
  (ADR-0017) can fall differently in rare cases.

## Alternatives Considered

- No caches at all: derived data such as strata would be rebuilt at every step.
- Caches found by looking up the caller, the simulator or engine: the legacy backreference, which
  makes a strategy's result depend on who called it.
- History caches kept on the strategy, keyed by session: per-session state on an object that serves
  every session, lost or wrong when a session moves to another process.
- Derived data stored on the item bank for every strategy: the bank would hold data it cannot
  validate, keyed by strategies it does not know.
